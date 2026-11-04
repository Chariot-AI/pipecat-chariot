#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Tests for the Chariot TTS service, through real pipelines and a local server."""

import asyncio
import dataclasses
import json

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.utils.types import is_given

from pipecat_chariot import (
    CHARIOT_DEFAULT_TTS_MODEL,
    CHARIOT_DEFAULT_VOICE,
    ChariotTTSService,
    ChariotTTSSettings,
)
from pipecat_chariot.tts import _split_text
from tests.fakes import PCM, FakeTTSServer, next_message, run_pipeline


@pytest.fixture
async def server():
    async with FakeTTSServer() as fake:
        yield fake


def _service(server, **kwargs) -> ChariotTTSService:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("voice_id", "test-voice")
    return ChariotTTSService(base_url=server.url, **kwargs)


async def _speak(worker, down, text):
    await worker.queue_frame(TTSSpeakFrame(text))
    return await down.until(TTSStoppedFrame)


#
# Configuration
#


def test_defaults():
    service = ChariotTTSService(api_key="k")
    assert service._settings.voice == CHARIOT_DEFAULT_VOICE
    assert service._settings.model == CHARIOT_DEFAULT_TTS_MODEL == "chariot-multilingual-v1"
    assert service._settings.speed is None


def test_all_settings_fields_are_initialized():
    """Pipecat validates that no settings field is left NOT_GIVEN at startup."""
    service = ChariotTTSService(api_key="k")
    unset = [
        f.name
        for f in dataclasses.fields(service._settings)
        if not is_given(getattr(service._settings, f.name))
    ]
    assert unset == []


def test_settings_win_over_arguments():
    service = ChariotTTSService(
        api_key="k",
        voice_id="direct",
        model="chariot-v0",
        settings=ChariotTTSSettings(voice="from-settings"),
    )
    assert service._settings.voice == "from-settings"
    assert service._settings.model == "chariot-v0"


def test_a_settings_delta_leaves_other_fields_alone():
    """Updating only the voice must not reset the speed."""
    service = ChariotTTSService(api_key="k", speed=1.25)
    service._settings.apply_update(ChariotTTSSettings(voice="other"))
    assert service._settings.speed == 1.25


def test_url_carries_only_what_was_set():
    service = ChariotTTSService(api_key="k")
    url = service._websocket_url()
    assert url.startswith("wss://api.chariot.in/v1/tts/ws-multi?")
    assert "model_type=chariot-multilingual-v1" in url
    assert "idle_timeout=300" in url
    assert "enable_logging" not in url
    assert "optimize_streaming_latency" not in url
    assert "voice_id" not in url, "the voice travels per context, not per connection"

    url = ChariotTTSService(
        api_key="k", enable_logging=False, optimize_streaming_latency=2
    )._websocket_url()
    assert "enable_logging=false" in url
    assert "optimize_streaming_latency=2" in url


def test_split_text_reconstructs_exactly():
    text = "All work and no play makes Jack a dull boy. " * 50
    pieces = _split_text(text, 500)
    assert "".join(pieces) == text
    assert all(len(p) <= 500 for p in pieces)
    assert _split_text("x" * 1200, 500) == ["x" * 500, "x" * 500, "x" * 200]


#
# Pipelines
#


async def test_speaks_a_turn(server):
    async with run_pipeline(_service(server, speed=1.25)) as (worker, down, up):
        stopped = await _speak(worker, down, "Namaste.")

        audio = down.of_type(TTSAudioRawFrame)
        assert b"".join(f.audio for f in audio) == PCM, (
            "an odd chunk boundary must not lose samples"
        )
        assert all(len(f.audio) % 2 == 0 for f in audio)
        assert all(f.sample_rate == 44100 and f.num_channels == 1 for f in audio)
        assert all(f.context_id == stopped.context_id for f in audio)
        assert down.of_type(TTSStartedFrame)[0].context_id == stopped.context_id
        assert not up.of_type(ErrorFrame)
        metrics = [m for f in down.of_type(MetricsFrame) for m in f.data]
        assert any(isinstance(m, TTFBMetricsData) and m.value > 0 for m in metrics)

    conn = server.connections[0]
    assert conn.path == "/v1/tts/ws-multi"
    assert conn.headers["chariotai-api-key"] == "test-key"
    assert conn.headers["user-agent"].startswith("pipecat-chariot/")
    assert " pipecat/" in conn.headers["user-agent"]
    opened = conn.of_type("input.text")[0]
    assert opened == {
        "type": "input.text",
        "context_id": stopped.context_id,
        "text": "",
        "voice_id": "test-voice",
        "speed": 1.25,
    }
    assert [m["text"] for m in conn.of_type("input.text")[1:]] == ["Namaste."]
    assert conn.of_type("context.close") == [
        {"type": "context.close", "context_id": stopped.context_id}
    ]


async def test_turns_share_one_connection(server):
    async with run_pipeline(_service(server)) as (worker, down, up):
        first = await _speak(worker, down, "First.")
        second = await _speak(worker, down, "Second.")
        assert first.context_id != second.context_id
        assert not up.of_type(ErrorFrame)
    assert len(server.connections) == 1


async def test_interruption_cancels_the_context_and_keeps_the_connection(server):
    server.stall = True
    async with run_pipeline(_service(server)) as (worker, down, up):
        await worker.queue_frame(TTSSpeakFrame("Interrupted."))
        first = await down.until(TTSAudioRawFrame)

        await worker.queue_frame(InterruptionFrame())
        cancel = await next_message(server.received, "context.cancel")
        assert cancel["context_id"] == first.context_id

        server.stall = False
        second = await _speak(worker, down, "Next turn.")
        assert second.context_id != first.context_id
        assert not up.of_type(ErrorFrame)
    assert len(server.connections) == 1


async def test_a_pause_in_generation_does_not_cut_the_turn_short(server):
    """Generation can pause for seconds mid segment under load.

    Pipecat's default gives up on a turn after 3 seconds without audio, which
    dropped the rest of it and never ended the turn. The turn must complete,
    with all its audio and exactly one stop frame.
    """
    server.pause_between_chunks = 3.5
    async with run_pipeline(_service(server)) as (worker, down, up):
        await worker.queue_frame(TTSSpeakFrame("A slow one."))
        stopped = await down.until(TTSStoppedFrame, timeout=10)
        audio = [f for f in down.of_type(TTSAudioRawFrame) if f.context_id == stopped.context_id]
        assert b"".join(f.audio for f in audio) == PCM
        await asyncio.sleep(0.2)
        assert len(down.of_type(TTSStoppedFrame)) == 1
        assert not up.of_type(ErrorFrame)


async def test_voice_and_speed_changes_apply_to_the_next_turn_without_reconnecting(server):
    async with run_pipeline(_service(server)) as (worker, down, _up):
        await _speak(worker, down, "Before.")
        await worker.queue_frame(
            TTSUpdateSettingsFrame(delta=ChariotTTSSettings(voice="new-voice", speed=0.75))
        )
        await _speak(worker, down, "After.")
    assert len(server.connections) == 1
    opens = [m for m in server.connections[0].of_type("input.text") if "voice_id" in m]
    assert [(m["voice_id"], m.get("speed")) for m in opens] == [
        ("test-voice", None),
        ("new-voice", 0.75),
    ]


async def test_a_model_change_reconnects(server):
    async with run_pipeline(_service(server)) as (worker, down, _up):
        await _speak(worker, down, "Before.")
        await worker.queue_frame(
            TTSUpdateSettingsFrame(delta=ChariotTTSSettings(model="chariot-v0"))
        )
        await _speak(worker, down, "After.")
    assert [c.query["model_type"] for c in server.connections] == [
        "chariot-multilingual-v1",
        "chariot-v0",
    ]


async def test_exhausted_credits_stop_the_service_with_one_error(server):
    server.close_on_text = 4402
    service = _service(server)
    async with run_pipeline(service) as (worker, down, up):
        await worker.queue_frame(TTSSpeakFrame("Hello."))
        error = await up.until(ErrorFrame)
        assert "credit balance exhausted" in error.error
        await down.until(TTSStoppedFrame)
        await asyncio.sleep(0.3)
        chariot_errors = [e for e in up.of_type(ErrorFrame) if "Chariot" in e.error]
        assert len(chariot_errors) == 1, "the server's error and its close are one failure"
        assert service.is_usable is False, "an empty balance refuses every retry the same way"
    assert len(server.connections) == 1


async def test_a_failed_context_ends_the_turn_with_an_error(server):
    server.context_error = ("voice_unavailable", "voice is not available on this model")
    async with run_pipeline(_service(server)) as (worker, down, up):
        await worker.queue_frame(TTSSpeakFrame("Hello."))
        error = await up.until(ErrorFrame)
        assert "voice is not available" in error.error
        await down.until(TTSStoppedFrame)


async def test_a_refused_context_ends_the_turn(server):
    server.reject_open = "context_limit"
    async with run_pipeline(_service(server)) as (worker, down, up):
        await worker.queue_frame(TTSSpeakFrame("Hello."))
        error = await up.until(ErrorFrame)
        assert "context_limit" in error.error
        await down.until(TTSStoppedFrame)


async def test_long_text_is_sent_in_fragments(server):
    long_text = "All work and no play makes Jack a dull boy. " * 30
    async with run_pipeline(_service(server)) as (worker, down, _up):
        await _speak(worker, down, long_text)
    texts = [m["text"] for m in server.connections[0].of_type("input.text") if m["text"]]
    assert all(len(t) <= 500 for t in texts)
    assert "".join(texts).strip() == long_text.strip()


async def test_keepalive_touches_only_live_contexts(server):
    service = _service(server)
    async with run_pipeline(service) as (worker, down, _up):
        await _speak(worker, down, "Connect.")
        # The current turn, mid reply; one whose close was already sent (a
        # closed context refuses input); and one Pipecat has let go of, which
        # must be left to expire rather than hold a slot forever.
        service._turn_context_id = "mid-reply"
        service._open_contexts.update({"mid-reply", "closing", "orphan"})
        service._closing_contexts.add("closing")
        while not server.received.empty():
            server.received.get_nowait()

        await service._send_keepalive()
        assert await next_message(server.received, "ping") == {"type": "ping"}
        keepalive = await next_message(server.received, "input.text")
        assert keepalive == {"type": "input.text", "context_id": "mid-reply", "text": ""}
        await asyncio.sleep(0.1)
        assert server.received.empty()


async def test_token_streaming_flushes_only_between_sentences():
    from pipecat.services.tts_service import TextAggregationMode
    from websockets.protocol import State

    class Recorder:
        state = State.OPEN

        def __init__(self):
            self.sent = []

        async def send(self, message):
            self.sent.append(json.loads(message))

    service = ChariotTTSService(api_key="k", text_aggregation_mode=TextAggregationMode.TOKEN)
    service._websocket = Recorder()
    tokens = [
        "It",
        " costs",
        " 3",
        ".",
        "5",
        " lakh",
        "।",
        " Dr",
        ".",
        " Rao",
        " agreed",
        ".",
        " Next",
    ]
    for token in tokens:
        await service._send_text(token, "turn")
    sent = service._websocket.sent[1:]
    flushed_before = [sent[i + 1]["text"] for i, m in enumerate(sent) if m["type"] == "input.flush"]
    # "3.5" stays whole: a flush needs the dot to be followed by whitespace.
    # An abbreviation like "Dr." still splits, as any one token lookahead must.
    assert flushed_before == [" Dr", " Rao", " Next"]
