#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Chariot text-to-speech service implementation.

This module provides a TTS service on Chariot's multi-context WebSocket API.
Chariot serves lifelike voices for Indian languages and English.

See https://docs.chariot.in/api-reference/introduction for full API details.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from pipecat_chariot._common import (
    CHARIOT_BASE_URL,
    CLOSE_CODE_NO_CREDITS,
    close_code,
    close_reason,
    user_agent,
)

# Chariot always synthesizes 16-bit mono PCM at 44.1 kHz. The server also
# declares the rate when the session opens and per generation segment, and
# this service trusts that declaration over the constant.
CHARIOT_SAMPLE_RATE = 44100

# Matt, a published Chariot stock voice. Used when no voice is given so the
# service works out of the box with just an API key.
CHARIOT_DEFAULT_VOICE = "bac7d666-094d-4698-91fa-741d60fce662"

# The model used when none is given.
CHARIOT_DEFAULT_TTS_MODEL = "chariot-multilingual-v1"

# The longest single input.text the smallest model accepts. Longer text is sent
# as several fragments; the server flushes its buffer by itself whenever the
# next fragment would overflow it, so no text is ever dropped.
_MAX_FRAGMENT_CHARS = 500

# The longest the server keeps a quiet session open (its maximum). The
# keepalive below keeps a live session well inside it.
_IDLE_TIMEOUT_SECS = 300

# The server ends a context that has nothing generating and has heard nothing
# for 20 seconds, which a slow LLM can exceed mid-turn. The keepalive touches
# every open context more often than that.
_KEEPALIVE_SECS = 10

# Error codes for a rejected message naming a context that is not open. No
# context.done follows these, so the turn has to be closed here.
_CONTEXT_NOT_OPEN_CODES = {"bad_request", "context_limit", "session_closing", "unknown_context"}

# Characters that end a sentence, including the Devanagari danda. In token
# streaming mode a flush is sent once a token ending in one of these is
# followed by whitespace, so each generated segment is a whole sentence, and
# "3.5" or "Dr. Rao" is not cut at the dot.
_SENTENCE_ENDINGS = (".", "!", "?", "\u0964", "\u0965")

# Server error codes that come just before the server closes the connection.
# They are reported once, with the close, rather than twice.
_CLOSING_ERROR_CODES = {"insufficient_credits", "idle_timeout"}

# How long a turn may go without audio before Pipecat gives up on it. The
# server always ends a context explicitly with context.done, so this is only a
# backstop; Pipecat's default of 3 seconds would cut a turn short whenever
# generation pauses longer than that, losing the rest of its audio.
_STOP_FRAME_TIMEOUT_SECS = 10.0

# Close codes after which reconnecting cannot help: the session was refused
# for a reason the next connection would hit too.
_CLOSE_CODE_POLICY = 1008


def _split_text(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most ``limit`` characters.

    Cuts at the last space inside the limit when there is one, so words stay
    whole; a single over-long token is hard-split. Concatenating the pieces
    reproduces the input exactly, which is what the server's buffer sees.
    """
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 1, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


@dataclass
class ChariotTTSSettings(TTSSettings):
    """Settings for ChariotTTSService.

    Parameters:
        voice: Chariot voice UUID, from ``GET https://api.chariot.in/v1/voices``.
            Any voice speaks any supported language, so the language of the
            audio follows the text rather than the voice. Changing it applies
            from the next turn, on the same connection.
        model: Chariot TTS model, ``chariot-multilingual-v1`` (the default) or
            ``chariot-v0``. Fixed for a connection, so changing it reconnects.
        speed: Speaking rate. Rates are pre-encoded per voice and per model,
            so this is a fixed set rather than a range, and a rate a voice
            does not have is refused. ``GET /v1/voices`` lists what each voice
            offers under each model as ``speeds``. ``None`` uses the voice as
            recorded. Changing it applies from the next turn.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ChariotTTSService(WebsocketTTSService):
    """Chariot real-time text-to-speech over the multi-context WebSocket API.

    One WebSocket carries the whole conversation. Each bot turn is a Chariot
    context: sentences are generated as they arrive, and audio streams back
    as 16-bit mono PCM at 44.1 kHz while the server is still generating. An
    interruption cancels the turn's context and the connection stays open,
    so the next turn starts without a new handshake.

    Example::

        tts = ChariotTTSService(
            api_key="your-api-key",
            settings=ChariotTTSService.Settings(voice="your-voice-uuid"),
        )
    """

    Settings = ChariotTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        base_url: str = CHARIOT_BASE_URL,
        sample_rate: int | None = CHARIOT_SAMPLE_RATE,
        optimize_streaming_latency: int | None = None,
        enable_logging: bool | None = None,
        extra_headers: dict[str, str] | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Chariot TTS service.

        Args:
            api_key: Chariot API key, from https://platform.chariot.in.
            voice_id: Chariot voice UUID from ``GET /v1/voices``.
                ``settings.voice`` takes precedence. Defaults to Matt, a
                Chariot stock voice, when neither is given.
            model: Chariot TTS model. Defaults to ``chariot-multilingual-v1``.
                ``settings.model`` takes precedence.
            speed: Speaking rate, one of the speeds this voice supports under
                this model. See ``GET /v1/voices``. ``None`` uses the voice as
                recorded. ``settings.speed`` takes precedence.
            base_url: Base WebSocket URL for the Chariot API.
            sample_rate: Audio sample rate in Hz. Chariot outputs 44100; the
                server declares its rate and that declaration is applied to
                every audio frame.
            optimize_streaming_latency: Latency optimization level, 0 to 4.
                Higher values trade quality passes for lower latency. ``None``
                uses the server default.
            enable_logging: Whether Chariot may store the generated audio.
                Not sent unless set. Unset or ``True``, your workspace's data
                retention setting decides; ``False`` keeps the audio out of
                storage for this connection.
            extra_headers: Additional headers sent with the WebSocket handshake.
            settings: Runtime-updatable settings; values here win over the
                direct arguments.
            **kwargs: Additional arguments passed to WebsocketTTSService.
        """
        if sample_rate is not None and sample_rate != CHARIOT_SAMPLE_RATE:
            logger.warning(
                f"Chariot TTS streams at {CHARIOT_SAMPLE_RATE} Hz; "
                f"configured sample_rate={sample_rate}"
            )

        # Every inherited field is set, including language, which the API
        # does not take: the base class validates that none is left unset.
        default_settings = self.Settings(
            voice=voice_id or CHARIOT_DEFAULT_VOICE,
            model=model or CHARIOT_DEFAULT_TTS_MODEL,
            speed=speed,
            language=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        kwargs.setdefault("stop_frame_timeout_s", _STOP_FRAME_TIMEOUT_SECS)
        super().__init__(
            push_text_frames=True,
            # The stop frame normally comes from context.done; this makes sure
            # a turn that times out instead still ends with one.
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._optimize_streaming_latency = optimize_streaming_latency
        self._enable_logging = enable_logging
        self._extra_headers = dict(extra_headers or {})

        # The rate stamped on audio frames, from session.ready and audio.start.
        self._output_sample_rate = sample_rate or CHARIOT_SAMPLE_RATE

        self._receive_task = None

        # Contexts this connection has opened on the server (their first
        # input.text was sent) and, of those, the ones already asked to close.
        # A closed context refuses further input.text, keepalives included.
        self._open_contexts: set[str] = set()
        self._closing_contexts: set[str] = set()
        # Contexts cancelled by an interruption. Chunks already on the wire
        # for them still arrive, and are dropped.
        self._cancelled_contexts: set[str] = set()
        # In token streaming mode, contexts whose last token ended a sentence;
        # the flush waits to see that the next token starts a new one.
        self._sentence_pending: set[str] = set()
        # The message of a server error announcing that it is about to close
        # the connection, reported together with the close.
        self._closing_error: str | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True, as Chariot service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Chariot's language format.

        The API takes no language parameter: any voice speaks any supported
        language and the audio follows the text, so there is nothing to map.

        Args:
            language: The language to convert.

        Returns:
            None, always.
        """
        return None

    def _websocket_url(self) -> str:
        params: dict[str, Any] = {
            "model_type": self._settings.model,
            "idle_timeout": _IDLE_TIMEOUT_SECS,
        }
        if self._optimize_streaming_latency is not None:
            params["optimize_streaming_latency"] = self._optimize_streaming_latency
        if self._enable_logging is not None:
            params["enable_logging"] = str(self._enable_logging).lower()
        return f"{self._base_url}/v1/tts/ws-multi?{urlencode(params)}"

    async def start(self, frame: StartFrame):
        """Start the Chariot TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Voice and speed are sent when a context opens, so a change applies
        from the next turn without touching the connection. The model is
        fixed for a connection, so changing it reconnects.
        """
        changed = await super()._update_settings(delta)
        if "model" in changed and self._websocket and self._websocket.state is State.OPEN:
            await self._disconnect()
            await self._connect()
        self._warn_unhandled_updated_settings(changed.keys() - {"voice", "model", "speed"})
        return changed

    #
    # Connection
    #

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        # A receive loop that ended (after a refused session) is replaced, so
        # a reconnect on the next turn is read again.
        if self._websocket and (not self._receive_task or self._receive_task.done()):
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket:
            self._create_keepalive_task(interval=_KEEPALIVE_SECS)

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._cancel_keepalive_task()

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Chariot TTS")
            # Contexts belong to a connection, so a new one starts clean.
            self._clear_context_state()
            self._websocket = await self._websocket_connect(
                self._websocket_url(),
                additional_headers={**self._extra_headers, "chariotai-api-key": self._api_key},
                user_agent_header=user_agent(),
                max_size=None,
            )
            await self._call_event_handler("on_connected")
        except Exception as e:  # noqa: BLE001
            self._websocket = None
            await self.push_error(
                error_msg=f"Error connecting to Chariot TTS WebSocket: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket.

        Closing ends every context still open on the server, which stops
        anything it is generating. ``session.close`` is not sent: it would
        wait for open contexts to finish, and teardown should be prompt.
        """
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Chariot TTS")
                await self._websocket.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            self._clear_context_state()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise RuntimeError("Websocket not connected")

    def _clear_context_state(self) -> None:
        self._open_contexts.clear()
        self._closing_contexts.clear()
        self._cancelled_contexts.clear()
        self._sentence_pending.clear()
        self._closing_error = None

    async def _send(self, message: dict) -> None:
        await self._get_websocket().send(json.dumps(message))

    async def _send_keepalive(self):
        """Keep the connection and every open context alive.

        A ping resets the connection's idle timer. An empty ``input.text``
        resets a context's, which matters between sentences of a turn when
        the LLM is slow to produce the next one. Only the current turn and
        contexts Pipecat is still playing are kept alive: one it has let go of
        is left to expire on the server, so it never holds one of the
        connection's slots.
        """
        await self._send({"type": "ping"})
        for context_id in self._open_contexts - self._closing_contexts:
            if context_id == self._turn_context_id or self.audio_context_available(context_id):
                await self._send({"type": "input.text", "context_id": context_id, "text": ""})

    #
    # Contexts
    #

    async def flush_audio(self, context_id: str | None = None):
        """Generate any text the server is still buffering for a context.

        Args:
            context_id: The context to flush. ``None`` uses the active one.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or flush_id not in self._open_contexts:
            return
        if flush_id in self._closing_contexts:
            return
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._send({"type": "input.flush", "context_id": flush_id})
        except Exception as e:  # noqa: BLE001
            await self.push_error(error_msg=f"Error flushing Chariot TTS: {e}", exception=e)

    async def _close_context(self, context_id: str | None) -> None:
        """Ask the server to finish a context: generate what is left, then end it."""
        if not context_id or context_id not in self._open_contexts:
            return
        if context_id in self._closing_contexts:
            return
        self._closing_contexts.add(context_id)
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._send({"type": "context.close", "context_id": context_id})
            except Exception as e:  # noqa: BLE001
                await self.push_error(
                    error_msg=f"Error closing Chariot TTS context: {e}", exception=e
                )

    async def _end_context(self, context_id: str | None) -> None:
        """Close a turn locally: emit the stop frame and release its audio context."""
        if context_id and self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    def _forget_context(self, context_id: str) -> None:
        self._open_contexts.discard(context_id)
        self._closing_contexts.discard(context_id)
        self._cancelled_contexts.discard(context_id)
        self._sentence_pending.discard(context_id)

    async def on_turn_context_completed(self):
        """Close the turn's context once the LLM has finished the turn.

        The server generates anything still buffered, then answers with
        ``context.done``, which closes the turn here.
        """
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        await self._close_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the turn's context on the server when the user barges in.

        Generation stops at once and the connection stays open for the next
        turn. Audio already in flight for the context is dropped on arrival.
        Any other context still open on the server, such as one Pipecat had
        already let go of during a long pause, is cancelled too.
        """
        await self.stop_all_metrics()
        stale = {
            cid
            for cid in self._open_contexts - self._closing_contexts
            if not self.audio_context_available(cid)
        }
        for cid in {context_id} | stale:
            await self._cancel_context(cid)
        await super().on_audio_context_interrupted(context_id)

    async def _cancel_context(self, context_id: str) -> None:
        if context_id not in self._open_contexts or not self._websocket:
            return
        self._closing_contexts.add(context_id)
        self._cancelled_contexts.add(context_id)
        self._sentence_pending.discard(context_id)
        try:
            await self._send({"type": "context.cancel", "context_id": context_id})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self} could not cancel context {context_id}: {e}")

    #
    # Receiving
    #

    async def _receive_messages(self):
        """Receive audio and control events from the Chariot WebSocket."""
        try:
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    continue
                await self._handle_event(json.loads(message))
        except ConnectionClosed as e:
            code = close_code(e)
            reason = self._closing_error or close_reason(e)
            if code == CLOSE_CODE_NO_CREDITS:
                await self._handle_refusal(
                    "Chariot TTS: credit balance exhausted. Top up at "
                    "https://platform.chariot.in and start a new session."
                )
                return
            if code == _CLOSE_CODE_POLICY and "idle" not in reason.lower():
                await self._handle_refusal(
                    f"Chariot TTS refused the session: {reason or 'policy violation'}"
                )
                return
            raise

    async def _handle_refusal(self, message: str) -> None:
        """Report a refusal that reconnecting would only repeat.

        A bad key, a bad setting or an empty balance would refuse every new
        connection the same way, so the service stops for this session
        rather than retrying.
        """
        for context_id in self.get_audio_contexts():
            await self._end_context(context_id)
        self._clear_context_state()
        self._disconnecting = True
        await self.push_error(error_msg=message, force_treat_as_permanent=True)

    async def _handle_event(self, event: dict) -> None:
        event_type = event.get("type")
        context_id = event.get("context_id")

        if event_type == "audio.chunk":
            if context_id in self._cancelled_contexts:
                # Chunks already on the wire when the turn was cancelled.
                return
            await self.stop_ttfb_metrics()
            audio = base64.b64decode(event.get("audio", ""))
            # Pipecat realigns chunks that end mid sample, and recreates the
            # turn's audio context if it timed out while audio was delayed.
            if audio:
                frame = TTSAudioRawFrame(audio, self._output_sample_rate, 1, context_id=context_id)
                await self.append_to_audio_context(context_id, frame)
        elif event_type in ("session.ready", "audio.start"):
            self._output_sample_rate = event.get("sample_rate") or self._output_sample_rate
        elif event_type == "audio.done":
            if event.get("error") and self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, ErrorFrame(error="Chariot TTS: a segment failed to generate")
                )
        elif event_type == "context.done":
            await self._on_context_done(context_id, event)
        elif event_type == "error":
            await self._on_error(context_id, event)

    async def _on_context_done(self, context_id: str | None, event: dict) -> None:
        """The last message for a context, whatever ended it."""
        if not context_id:
            return
        self._forget_context(context_id)
        reason = event.get("reason")
        if reason == "error" and self.audio_context_available(context_id):
            message = event.get("message") or event.get("code") or "context failed"
            await self.append_to_audio_context(
                context_id, ErrorFrame(error=f"Chariot TTS error: {message}")
            )
        if reason == "expired":
            logger.warning(f"{self} context {context_id} expired before the turn completed")
        await self._end_context(context_id)

    async def _on_error(self, context_id: str | None, event: dict) -> None:
        """A rejected message. The connection stays open."""
        code = event.get("code")
        message = event.get("message", "unknown error")
        if code in _CLOSING_ERROR_CODES:
            self._closing_error = message
            return
        await self.push_error(error_msg=f"Chariot TTS error ({code}): {message}")
        if context_id and code in _CONTEXT_NOT_OPEN_CODES:
            self._forget_context(context_id)
            await self._end_context(context_id)

    #
    # Synthesis
    #

    async def _open_context(self, context_id: str) -> None:
        """Open a context on the server with the turn's voice and speed."""
        message: dict[str, Any] = {
            "type": "input.text",
            "context_id": context_id,
            "text": "",
            "voice_id": self._settings.voice,
        }
        speed = self._settings.speed
        if is_given(speed) and speed is not None:
            message["speed"] = speed
        await self._send(message)
        self._open_contexts.add(context_id)

    async def _send_text(self, text: str, context_id: str) -> None:
        """Send text for a turn and generate it as soon as a sentence is whole.

        By default Pipecat hands the service one sentence at a time, and each
        is flushed straight away, so its audio starts while the LLM is still
        writing the next. In token streaming mode text arrives a token at a
        time, so the flush waits until a sentence ending is followed by the
        start of the next sentence; whatever is left at the end of the turn is
        generated when the context closes.
        """
        if context_id not in self._open_contexts:
            await self._open_context(context_id)

        if self._is_streaming_tokens:
            if context_id in self._sentence_pending and text[:1].isspace():
                await self._send({"type": "input.flush", "context_id": context_id})
            self._sentence_pending.discard(context_id)

        for piece in _split_text(text, _MAX_FRAGMENT_CHARS):
            await self._send({"type": "input.text", "context_id": context_id, "text": piece})

        if not self._is_streaming_tokens:
            await self._send({"type": "input.flush", "context_id": context_id})
        elif text.rstrip().endswith(_SENTENCE_ENDINGS) or text.endswith("\n"):
            self._sentence_pending.add(context_id)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech audio frames from input text using Chariot TTS.

        Sends text over the WebSocket; audio arrives on the receive task and
        is routed to the turn's audio context.

        Args:
            text: The text to synthesize.
            context_id: The turn's context ID.

        Yields:
            Frame objects, or None when audio is delivered via the context.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if self._websocket is None:
                yield ErrorFrame(error="Chariot TTS: not connected")
                return

            try:
                await self._send_text(text, context_id)
                await self.start_tts_usage_metrics(text)
            except Exception as e:  # noqa: BLE001
                yield ErrorFrame(error=f"Chariot TTS: could not send text: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                return
            yield None
        except Exception as e:  # noqa: BLE001
            yield ErrorFrame(error=f"Chariot TTS: unexpected error: {e}")
