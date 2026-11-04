#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Tests for the Chariot STT service, through real pipelines and a local server."""

import asyncio
import time

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.transcriptions.language import Language

from pipecat_chariot import CHARIOT_DEFAULT_STT_MODEL, ChariotSTTService, ChariotSTTSettings
from pipecat_chariot.stt import chariot_language
from tests.fakes import FakeSTTServer, next_message, run_pipeline

RATE = 16000
FRAME_20MS = b"\x01\x00" * (RATE // 50)


@pytest.fixture
async def server():
    async with FakeSTTServer() as fake:
        yield fake


def _service(server, **kwargs) -> ChariotSTTService:
    kwargs.setdefault("api_key", "test-key")
    return ChariotSTTService(base_url=server.url, **kwargs)


async def _speak(worker, frames: int):
    await worker.queue_frame(VADUserStartedSpeakingFrame())
    for _ in range(frames):
        await worker.queue_frame(InputAudioRawFrame(FRAME_20MS, RATE, 1))


async def _stop_speaking(worker):
    await worker.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()))


async def _connected(server, count: int = 1):
    async with asyncio.timeout(5):
        while len(server.connections) < count:
            await asyncio.sleep(0.01)


#
# Configuration
#


def test_languages():
    assert chariot_language(None) is None
    assert chariot_language(Language.HI) == "hi"
    assert chariot_language(Language.HI_IN) == "hi"
    assert chariot_language("ta-IN") == "ta"
    with pytest.raises(ValueError):
        chariot_language(Language.FR)


def test_unsupported_language_is_refused_at_construction():
    with pytest.raises(ValueError):
        ChariotSTTService(api_key="k", settings=ChariotSTTSettings(language=Language.DE))


def test_url_carries_only_what_was_set():
    service = ChariotSTTService(api_key="k", sample_rate=16000)
    service._settings.language = None
    url = service._websocket_url().replace(f"pcm_{service.sample_rate}", "pcm_RATE")
    assert url.startswith("wss://api.chariot.in/v1/asr/ws?")
    assert f"model_type={CHARIOT_DEFAULT_STT_MODEL}" in url
    assert "commit_strategy=manual" in url
    for absent in ("language=", "mode=", "style=", "include_timestamps", "vad_threshold"):
        assert absent not in url


def test_server_vad_settings_are_sent_only_with_server_vad():
    manual = ChariotSTTService(api_key="k", silence_duration_ms=800)
    assert "silence_duration_ms" not in manual._websocket_url()
    vad = ChariotSTTService(api_key="k", commit_strategy="vad", silence_duration_ms=800)
    assert "commit_strategy=vad" in vad._websocket_url()
    assert "silence_duration_ms=800" in vad._websocket_url()


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        ChariotSTTService(api_key="k", commit_strategy="sometimes")
    with pytest.raises(ValueError):
        ChariotSTTService(api_key="k", sample_rate=11025)


#
# Pipelines
#


async def test_manual_commit_finalizes_the_utterance(server):
    service = _service(
        server,
        settings=ChariotSTTSettings(language=Language.HI, mode="balanced", style="codemix"),
    )
    async with run_pipeline(service) as (worker, down, up):
        await _connected(server)
        await _speak(worker, 15)  # 300 ms of audio
        interim = await down.until(InterimTranscriptionFrame)
        assert interim.text == "namaste"

        await _stop_speaking(worker)
        await next_message(server.received, "input.commit")
        final = await down.until(TranscriptionFrame)
        assert final.text == "namaste duniya"
        assert final.finalized is True
        assert final.language == Language.HI
        assert not up.of_type(ErrorFrame)

    conn = server.connections[0]
    assert conn.query["audio_format"] == "pcm_16000"
    assert conn.query["language"] == "hi"
    assert conn.query["commit_strategy"] == "manual"
    assert conn.headers["chariotai-api-key"] == "test-key"
    assert conn.headers["user-agent"].startswith("pipecat-chariot/")

    # 20 ms frames go out as 100 ms chunks, plus whatever was left at the commit.
    audio = [m for m in conn.messages if isinstance(m, bytes)]
    assert sum(len(a) for a in audio) == 15 * len(FRAME_20MS)
    assert all(len(a) == 5 * len(FRAME_20MS) for a in audio[:-1])


async def test_no_commit_without_new_audio(server):
    async with run_pipeline(_service(server)) as (worker, down, _up):
        await _connected(server)
        await _speak(worker, 5)
        await _stop_speaking(worker)
        await next_message(server.received, "input.commit")
        await down.until(TranscriptionFrame)

        # The user "stops" again with nothing new said: a commit would get no
        # reply, so none is sent.
        await _stop_speaking(worker)
        await asyncio.sleep(0.2)
    assert len(server.connections[0].of_type("input.commit")) == 1


async def test_server_vad_splits_utterances(server):
    server.vad_commit_after_audio = True
    service = _service(server, commit_strategy="vad", silence_duration_ms=600)
    started = asyncio.Event()

    @service.event_handler("on_speech_started")
    async def _on_started(service, event):
        started.set()

    async with run_pipeline(service) as (worker, down, _up):
        await _connected(server)
        await _speak(worker, 10)
        final = await down.until(TranscriptionFrame)
        assert final.text == "namaste duniya"
        assert final.finalized is False, "only an answer to our own commit is final"
        await asyncio.wait_for(started.wait(), 2)
        await _stop_speaking(worker)
        await asyncio.sleep(0.2)

    conn = server.connections[0]
    assert conn.query["commit_strategy"] == "vad"
    assert conn.query["silence_duration_ms"] == "600"
    assert conn.of_type("input.commit") == [], "with server VAD, Chariot commits"


async def test_exhausted_credits_stop_the_service_with_one_error(server):
    server.close_on_connect = 4402
    server.close_reason = "insufficient credits"
    service = _service(server)
    async with run_pipeline(service) as (worker, _down, up):
        error = await up.until(ErrorFrame)
        assert "credit balance exhausted" in error.error
        await _speak(worker, 10)
        await asyncio.sleep(0.5)
        assert len(up.of_type(ErrorFrame)) == 1, "the server's error and its close are one failure"
        assert service.is_usable is False
    assert len(server.connections) == 1, "no reconnect loop against a refusal"


async def test_a_busy_slot_is_retried(server):
    server.close_on_connect = 4429
    server.close_reason = "concurrency limit"
    async with run_pipeline(_service(server)) as (worker, down, up):
        await _connected(server, 2)
        await asyncio.sleep(0.2)
        await _speak(worker, 10)
        await down.until(InterimTranscriptionFrame)
        assert not [e for e in up.of_type(ErrorFrame) if "refused" in e.error]


async def test_the_turn_is_final_only_when_every_commit_is_answered(server):
    server.hold_commits = 2
    async with run_pipeline(_service(server)) as (worker, down, _up):
        await _connected(server)
        await _speak(worker, 5)
        await _stop_speaking(worker)
        await next_message(server.received, "input.commit")
        # The user carries on before the first commit is answered.
        await _speak(worker, 5)
        await _stop_speaking(worker)
        first = await down.until(TranscriptionFrame)
        second = await down.until(TranscriptionFrame)
    assert (first.text, first.finalized) == ("segment 0", False)
    assert (second.text, second.finalized) == ("segment 1", True)


def test_ttfs_default_follows_mode_and_commit_strategy():
    assert ChariotSTTService(api_key="k")._ttfs_p99_latency == 0.6
    accurate = ChariotSTTService(api_key="k", settings=ChariotSTTSettings(mode="high_accuracy"))
    assert accurate._ttfs_p99_latency == 0.8
    vad = ChariotSTTService(api_key="k", commit_strategy="vad", silence_duration_ms=800)
    assert vad._ttfs_p99_latency == pytest.approx(1.4)
    assert ChariotSTTService(api_key="k", ttfs_p99_latency=0.7)._ttfs_p99_latency == 0.7


async def test_keepalive_is_a_message_not_audio(server):
    service = _service(server)
    async with run_pipeline(service) as (_worker, _down, _up):
        await _connected(server)
        await asyncio.sleep(0.1)
        await service._send_keepalive(b"\x00\x00" * 160)
        assert await next_message(server.received, "keepalive") == {"type": "keepalive"}
    assert not any(isinstance(m, bytes) for m in server.connections[0].messages)


async def test_ending_the_pipeline_collects_the_tail(server):
    service = _service(server)
    async with run_pipeline(service) as (worker, down, _up):
        await _connected(server)
        await _speak(worker, 5)
        await down.until(InterimTranscriptionFrame)
        await service._finish_input()
        tail = await down.until(TranscriptionFrame)
        assert tail.text == "the tail"
    assert server.connections[0].of_type("input.done") == [{"type": "input.done"}]
