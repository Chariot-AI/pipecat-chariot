#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Tests for the Chariot TTS service."""

import dataclasses
import json

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from websockets.protocol import State

from pipecat_chariot.tts import (
    CHARIOT_DEFAULT_VOICE,
    CHARIOT_SAMPLE_RATE,
    ChariotTTSService,
    ChariotTTSSettings,
)


def _service(**kwargs) -> ChariotTTSService:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("voice_id", "test-voice")
    return ChariotTTSService(**kwargs)


class _FakeWebsocket:
    """Collects sent messages; optionally yields scripted incoming messages."""

    def __init__(self, incoming=None, raise_after=None):
        self.state = State.OPEN
        self.sent: list[str] = []
        self._incoming = list(incoming or [])
        self._raise_after = raise_after

    async def send(self, message: str):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._incoming:
            return self._incoming.pop(0)
        if self._raise_after is not None:
            raise self._raise_after
        raise StopAsyncIteration


def test_default_voice_when_none_given():
    service = ChariotTTSService(api_key="test-key")
    assert service._settings.voice == CHARIOT_DEFAULT_VOICE


def test_settings_voice_wins_over_voice_id():
    service = _service(
        voice_id="direct-voice",
        settings=ChariotTTSSettings(voice="settings-voice"),
    )
    assert service._settings.voice == "settings-voice"


def test_websocket_url():
    service = _service(optimize_streaming_latency=2)
    url = service._websocket_url()
    assert url.startswith("wss://api.chariot.in/v1/tts/ws?")
    assert "voice_id=test-voice" in url
    assert "response_format=pcm" in url
    assert "idle_timeout=300" in url
    assert "optimize_streaming_latency=2" in url


def test_base_url_override():
    service = _service(base_url="wss://example.test/")
    assert service._websocket_url().startswith("wss://example.test/v1/tts/ws?")


def test_default_sample_rate_is_chariot_rate():
    service = _service()
    assert service._output_sample_rate == CHARIOT_SAMPLE_RATE


@pytest.mark.asyncio
async def test_send_text_appends_then_flushes():
    service = _service()
    ws = _FakeWebsocket()
    service._websocket = ws

    await service._send_text("Namaste!")

    assert [json.loads(m)["type"] for m in ws.sent] == ["input.text", "input.flush"]
    assert json.loads(ws.sent[0])["text"] == "Namaste!"


@pytest.mark.asyncio
async def test_receive_routes_audio_to_context():
    service = _service()
    appended = []
    removed = []

    service.get_active_audio_context_id = lambda: "ctx-1"
    service.audio_context_available = lambda context_id: True

    async def _append(context_id, frame):
        appended.append((context_id, frame))

    async def _remove(context_id):
        removed.append(context_id)

    async def _noop():
        pass

    service.append_to_audio_context = _append
    service.remove_audio_context = _remove
    service.stop_ttfb_metrics = _noop

    ws = _FakeWebsocket(
        incoming=[
            json.dumps({"type": "audio.start", "sample_rate": 22050}),
            b"\x00\x01" * 32,
            b"\x02\x03" * 32,
            json.dumps({"type": "audio.done"}),
        ]
    )
    service._get_websocket = lambda: ws

    await service._receive_messages()

    audio_frames = [f for _, f in appended if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 2
    # The server-declared rate from audio.start is applied to every frame.
    assert all(f.sample_rate == 22050 for f in audio_frames)
    assert all(context_id == "ctx-1" for context_id, _ in appended)


@pytest.mark.asyncio
async def test_segment_boundary_does_not_end_the_turn():
    """A multi segment turn must not be cut short by the first audio.done.

    Pipecat aggregates a turn into sentences and a long utterance is split
    into several flushed segments, so one turn produces several audio.done
    events. Ending the audio context on the first one drops the rest of the
    reply and stops the bot mid sentence.
    """
    service = _service()
    appended = []
    removed = []
    alive = {"turn-1": True}

    service.get_active_audio_context_id = lambda: "turn-1"
    service.audio_context_available = lambda context_id: alive.get(context_id, False)

    async def _append(context_id, frame):
        appended.append((context_id, frame, alive.get(context_id, False)))

    async def _remove(context_id):
        removed.append(context_id)
        alive[context_id] = False

    async def _noop():
        pass

    service.append_to_audio_context = _append
    service.remove_audio_context = _remove
    service.stop_ttfb_metrics = _noop

    ws = _FakeWebsocket(
        incoming=[
            json.dumps({"type": "audio.start", "sample_rate": 44100}),
            b"\x00\x01" * 64,
            json.dumps({"type": "audio.done"}),
            json.dumps({"type": "audio.start", "sample_rate": 44100}),
            b"\x02\x03" * 64,
            json.dumps({"type": "audio.done"}),
        ]
    )
    service._get_websocket = lambda: ws

    await service._receive_messages()

    audio = [a for a in appended if isinstance(a[1], TTSAudioRawFrame)]
    assert len(audio) == 2, "both segments must reach the context"
    assert all(was_alive for _, _, was_alive in audio), "audio landed on a closed context"
    assert removed == [], "the segment boundary must not end the audio context"
    stopped = [a for a in appended if isinstance(a[1], TTSStoppedFrame)]
    assert stopped == [], "end of turn belongs to the base class, not audio.done"


@pytest.mark.asyncio
async def test_receive_reports_exhausted_credits():
    service = _service()
    errors = []

    async def _push_error(error_msg=None, **kwargs):
        errors.append(error_msg)

    service.push_error = _push_error
    service.get_active_audio_context_id = lambda: None

    closed = ConnectionClosedError(Close(4402, "no credits"), None)
    ws = _FakeWebsocket(raise_after=closed)
    service._get_websocket = lambda: ws

    await service._receive_messages()

    assert len(errors) == 1
    assert "credit balance exhausted" in errors[0]


@pytest.mark.asyncio
async def test_receive_reraises_other_close_errors():
    service = _service()
    service.get_active_audio_context_id = lambda: None

    closed = ConnectionClosedError(Close(1011, "server error"), None)
    ws = _FakeWebsocket(raise_after=closed)
    service._get_websocket = lambda: ws

    with pytest.raises(ConnectionClosedError):
        await service._receive_messages()


@pytest.mark.asyncio
async def test_flush_audio_sends_input_flush():
    service = _service()
    ws = _FakeWebsocket()
    service._websocket = ws

    await service.flush_audio()

    assert [json.loads(m)["type"] for m in ws.sent] == ["input.flush"]


def test_split_text_reconstructs_exactly():
    from pipecat_chariot.tts import _split_text

    text = "All work and no play makes Jack a dull boy. " * 50
    pieces = _split_text(text, 500)
    assert "".join(pieces) == text
    assert all(len(p) <= 500 for p in pieces)
    # A single over-long token is hard-split rather than looping forever.
    assert _split_text("x" * 1200, 500) == ["x" * 500, "x" * 500, "x" * 200]


@pytest.mark.asyncio
async def test_long_text_is_chunked_before_flush():
    service = _service()
    ws = _FakeWebsocket()
    service._websocket = ws

    long_text = "All work and no play makes Jack a dull boy. " * 50  # 2200 chars
    await service._send_text(long_text)

    messages = [json.loads(m) for m in ws.sent]
    # Strict alternation: every buffered segment is flushed before the next.
    assert [m["type"] for m in messages] == ["input.text", "input.flush"] * (len(messages) // 2)
    texts = [m["text"] for m in messages if m["type"] == "input.text"]
    assert len(texts) > 1  # actually segmented
    assert all(len(t) <= 500 for t in texts)
    assert "".join(texts) == long_text


def test_all_settings_fields_are_initialized():
    """Pipecat validates that no settings field is left NOT_GIVEN at startup."""
    from pipecat.services.settings import is_given

    service = _service()
    unset = [
        f.name
        for f in dataclasses.fields(service._settings)
        if not is_given(getattr(service._settings, f.name))
    ]
    assert unset == [], f"settings fields left NOT_GIVEN: {unset}"


def _context_harness(service, context_id="turn-1"):
    """Wire a service to a fake audio context, returning its event log."""
    log = {"appended": [], "removed": [], "alive": {context_id: True}}

    service.get_active_audio_context_id = lambda: context_id
    service.audio_context_available = lambda cid: log["alive"].get(cid, False)

    async def _append(cid, frame):
        log["appended"].append((cid, frame))

    async def _remove(cid):
        log["removed"].append(cid)
        log["alive"][cid] = False

    async def _noop(*args, **kwargs):
        pass

    service.append_to_audio_context = _append
    service.remove_audio_context = _remove
    service.stop_ttfb_metrics = _noop
    return log


@pytest.mark.asyncio
async def test_turn_completion_waits_for_segments_still_generating():
    """The turn must not close while a flushed segment is still coming back."""
    service = _service()
    log = _context_harness(service)

    ws = _FakeWebsocket()
    service._websocket = ws
    await service._send_text("Two segments. " * 60)  # > 500 chars, so segmented
    assert service._segments_in_flight > 1

    # The LLM turn finishes while audio is still streaming.
    await service.on_turn_context_completed()
    stopped = [f for _, f in log["appended"] if isinstance(f, TTSStoppedFrame)]
    assert stopped == [], "closed the turn while segments were still generating"
    assert log["removed"] == []


@pytest.mark.asyncio
async def test_last_segment_closes_a_completed_turn():
    """Once the turn is done, the final audio.done finalizes it."""
    service = _service()
    log = _context_harness(service)

    ws = _FakeWebsocket()
    service._websocket = ws
    await service._send_text("One short sentence.")
    assert service._segments_in_flight == 1

    await service.on_turn_context_completed()
    assert log["removed"] == [], "nothing had come back yet"

    incoming = _FakeWebsocket(incoming=[json.dumps({"type": "audio.done"})])
    service._get_websocket = lambda: incoming
    await service._receive_messages()

    stopped = [f for _, f in log["appended"] if isinstance(f, TTSStoppedFrame)]
    assert len(stopped) == 1, "exactly one stop frame per turn"
    assert log["removed"] == ["turn-1"]
    assert service._segments_in_flight == 0
    assert service._turn_closing is False


@pytest.mark.asyncio
async def test_interruption_clears_turn_accounting():
    service = _service()
    _context_harness(service)
    ws = _FakeWebsocket()
    service._websocket = ws
    await service._send_text("Interrupted mid reply.")
    service._turn_closing = True

    await service.on_audio_context_interrupted("turn-1")

    assert service._segments_in_flight == 0
    assert service._turn_closing is False


def test_model_and_speed_are_omitted_when_unset():
    """No model or speed means the account default applies, server side."""
    url = _service()._websocket_url()
    assert "model_type=" not in url
    assert "speed=" not in url


def test_model_and_speed_travel_in_the_url():
    service = _service(model="v2", speed=1.25)
    url = service._websocket_url()
    assert "model_type=v2" in url
    assert "speed=1.25" in url


def test_settings_carry_model_and_speed():
    service = _service(model="v2", speed=0.75)
    assert service._settings.model == "v2"
    assert service._settings.speed == 0.75
