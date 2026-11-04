#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Chariot speech-to-text service implementation.

This module provides a streaming STT service on Chariot's real-time ASR
WebSocket API, for English and Indian languages, including code-mixed speech.

See https://docs.chariot.in/api-reference/introduction for full API details.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from pipecat_chariot._common import (
    CHARIOT_BASE_URL,
    CLOSE_CODE_NO_CREDITS,
    close_code,
    close_reason,
    user_agent,
)

# The real-time model, and the only one the streaming endpoint serves.
CHARIOT_DEFAULT_STT_MODEL = "chariot-1-asr-realtime"

# PCM rates the server accepts. Anything above 16 kHz is resampled server side.
CHARIOT_STT_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})

# Languages the server accepts. No language means automatic detection.
CHARIOT_STT_LANGUAGES = frozenset(
    {"en", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "sa", "ta", "te"}
)

CommitStrategy = Literal["manual", "vad"]

# Audio is sent in chunks of this length rather than Pipecat's 10 to 20 ms
# frames. The server queues a bounded number of messages per session, so
# larger messages leave far more headroom when it is briefly behind.
_CHUNK_SECS = 0.1

# The server closes with 1008 both for refusals (a bad key or parameter) and
# for limits a fresh session clears. These reasons are the second kind, so the
# service reconnects for them; any other 1008 stops it.
_CLOSE_CODE_POLICY = 1008
_RECOVERABLE_POLICY_REASONS = ("idle", "pending audio", "pending commit")

# How long to wait for the final transcript when the pipeline ends.
_FINAL_TRANSCRIPT_TIMEOUT_SECS = 5.0

# P99 time from the end of speech to the final transcript, in seconds, by mode,
# measured as Pipecat measures it (VAD stop_secs=0.2, commit on VAD stop).
# Measured medians were 0.34 s (balanced) and 0.48 s (high_accuracy), with
# worst cases of 0.55 s and 0.59 s; these defaults leave headroom for network
# distance. Pipecat uses the value to time turn ends when no finalized
# transcript arrives. Override it with a measurement from your deployment.
CHARIOT_STT_TTFS_P99 = {"balanced": 0.6, "high_accuracy": 0.8}


def chariot_language(language: Language | str | None) -> str | None:
    """Convert a Pipecat language to the code Chariot expects.

    Regional variants map to their base language (``hi-IN`` becomes ``hi``).

    Args:
        language: A Language enum, a language code, or None.

    Returns:
        The Chariot language code, or None for automatic detection.

    Raises:
        ValueError: If Chariot does not transcribe the language.
    """
    if language is None:
        return None
    value = language.value if isinstance(language, Language) else str(language)
    code = value.split("-")[0].lower()
    if code == "none":
        return None
    if code not in CHARIOT_STT_LANGUAGES:
        supported = ", ".join(sorted(CHARIOT_STT_LANGUAGES))
        raise ValueError(f"Chariot STT does not support language {value!r}; supported: {supported}")
    return code


@dataclass
class ChariotSTTSettings(STTSettings):
    """Settings for ChariotSTTService.

    Changing any of them reconnects, since they are fixed for a session.

    Parameters:
        model: Chariot ASR model. ``chariot-1-asr-realtime`` is the real-time
            model and the default.
        language: Language of the speech. ``None`` (the default) detects it
            automatically.
        mode: ``balanced`` (the default) or ``high_accuracy``, which is more
            accurate (it writes numbers as digits, for example). Its interim
            transcripts update less often, and the final one takes about
            0.15 s longer.
        style: ``codemix`` (the default) writes each word in the script it is
            usually written in, so English words in Hindi speech come out in
            Latin script; ``native`` writes everything in the language's own
            script.
    """

    mode: Literal["balanced", "high_accuracy"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    style: Literal["codemix", "native"] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ChariotSTTService(WebsocketSTTService):
    """Chariot real-time speech-to-text over the streaming WebSocket API.

    Streams the user's audio to Chariot and pushes interim transcripts while
    they speak and a final transcript per utterance.

    With the default ``commit_strategy="manual"``, Pipecat's VAD decides
    when an utterance ends: the service asks Chariot to finalize it and the
    resulting transcript is marked final, which lets the turn end promptly.
    With ``commit_strategy="vad"``, Chariot's server side voice detection
    splits utterances instead, while Pipecat's VAD still drives turn taking.

    Example::

        stt = ChariotSTTService(
            api_key="your-api-key",
            settings=ChariotSTTService.Settings(language=Language.HI),
        )

    Event handlers available (in addition to WebsocketSTTService events):

    - on_speech_started(service, event): server VAD heard speech start
      (``commit_strategy="vad"`` only).
    - on_speech_stopped(service, event): server VAD heard speech stop
      (``commit_strategy="vad"`` only).
    - on_session_expiring(service, event): the session is about a minute from
      its maximum length; the service reconnects when it ends.
    """

    Settings = ChariotSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = CHARIOT_BASE_URL,
        sample_rate: int | None = None,
        commit_strategy: CommitStrategy = "manual",
        include_timestamps: bool = False,
        vad_threshold: float | None = None,
        silence_duration_ms: int | None = None,
        min_speech_duration_ms: int | None = None,
        prefix_padding_ms: int | None = None,
        extra_headers: dict[str, str] | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = None,
        **kwargs,
    ):
        """Initialize the Chariot STT service.

        Args:
            api_key: Chariot API key, from https://platform.chariot.in.
            base_url: Base WebSocket URL for the Chariot API.
            sample_rate: Audio sample rate in Hz. Defaults to the pipeline's
                input rate. Must be one of 8000, 16000, 22050, 24000, 32000,
                44100 or 48000.
            commit_strategy: ``manual`` (default) finalizes each utterance when
                Pipecat's VAD hears the user stop. ``vad`` lets Chariot's server
                side voice detection split utterances.
            include_timestamps: Include word timestamps. They arrive in each
                transcription frame's ``result`` under ``words``.
            vad_threshold: Server VAD speech probability threshold, 0.05 to
                0.95. ``commit_strategy="vad"`` only; ``None`` uses the server
                default.
            silence_duration_ms: Server VAD silence that ends an utterance, 100
                to 10000 ms. ``commit_strategy="vad"`` only.
            min_speech_duration_ms: Server VAD minimum speech length, 0 to 5000
                ms. ``commit_strategy="vad"`` only.
            prefix_padding_ms: Server VAD audio kept before detected speech, 0
                to 2000 ms. ``commit_strategy="vad"`` only.
            extra_headers: Additional headers sent with the WebSocket handshake.
            settings: Runtime-updatable settings.
            ttfs_p99_latency: P99 latency from the end of speech to the final
                transcript, in seconds. ``None`` picks a default for the mode,
                plus the server's silence window when ``commit_strategy`` is
                ``vad``. Override it with a measurement from your deployment.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        if commit_strategy not in ("manual", "vad"):
            raise ValueError("commit_strategy must be 'manual' or 'vad'")
        if sample_rate is not None and sample_rate not in CHARIOT_STT_SAMPLE_RATES:
            raise ValueError(f"Chariot STT does not accept sample_rate={sample_rate}")

        default_settings = self.Settings(
            model=CHARIOT_DEFAULT_STT_MODEL,
            language=None,
            mode=None,
            style=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        default_settings.language = chariot_language(default_settings.language)

        if ttfs_p99_latency is None:
            ttfs_p99_latency = CHARIOT_STT_TTFS_P99.get(default_settings.mode or "balanced", 0.6)
            if commit_strategy == "vad":
                ttfs_p99_latency += (silence_duration_ms or 500) / 1000

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=20,
            keepalive_interval=5,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._commit_strategy = commit_strategy
        self._include_timestamps = include_timestamps
        self._extra_headers = dict(extra_headers or {})
        self._server_vad = {
            key: value
            for key, value in {
                "vad_threshold": vad_threshold,
                "silence_duration_ms": silence_duration_ms,
                "min_speech_duration_ms": min_speech_duration_ms,
                "prefix_padding_ms": prefix_padding_ms,
            }.items()
            if value is not None
        }
        if self._server_vad and commit_strategy != "vad":
            logger.warning(f"{self}: server VAD settings are ignored unless commit_strategy='vad'")

        self._receive_task = None
        self._audio_buffer = bytearray()
        self._chunk_bytes = 0  # set once the sample rate is known
        # Audio sent since the last commit. A commit with nothing new gets no
        # reply, so one is only sent (and awaited) when there is audio.
        self._uncommitted = False
        # Commits sent and not yet answered. The turn's transcript is final
        # only once every one of them has come back.
        self._pending_commits = 0
        # The message of a server error that precedes its closing the
        # connection, reported together with the close.
        self._closing_error: str | None = None
        self._partial = ""
        self._final_transcript = asyncio.Event()

        self._register_event_handler("on_speech_started")
        self._register_event_handler("on_speech_stopped")
        self._register_event_handler("on_session_expiring")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True, as Chariot service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Chariot's language code.

        Args:
            language: The language to convert.

        Returns:
            The Chariot language code, or None for automatic detection.
        """
        return chariot_language(language)

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting if anything changed.

        The utterance in progress is finalized first, so it is not lost to the
        reconnect. A language Chariot does not transcribe is refused and the
        current one kept.
        """
        language = getattr(delta, "language", NOT_GIVEN)
        if not isinstance(language, NotGiven):
            try:
                chariot_language(language)
            except ValueError as e:
                await self.push_error(error_msg=str(e))
                delta.language = NOT_GIVEN
        changed = await super()._update_settings(delta)
        if changed and self._websocket:
            await self._commit()
            await self._disconnect()
            await self._connect()
        return changed

    def _websocket_url(self) -> str:
        params: dict[str, Any] = {
            "model_type": self._settings.model,
            "audio_format": f"pcm_{self.sample_rate}",
            "commit_strategy": self._commit_strategy,
        }
        if self._settings.language:
            params["language"] = self._settings.language
        if self._settings.mode:
            params["mode"] = self._settings.mode
        if self._settings.style:
            params["style"] = self._settings.style
        if self._include_timestamps:
            params["include_timestamps"] = "true"
        if self._commit_strategy == "vad":
            params.update(self._server_vad)
        return f"{self._base_url}/v1/asr/ws?{urlencode(params)}"

    async def start(self, frame):
        """Start the service and connect.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        if self.sample_rate not in CHARIOT_STT_SAMPLE_RATES:
            await self.push_error(
                error_msg=f"Chariot STT does not accept the pipeline's {self.sample_rate} Hz audio",
                force_treat_as_permanent=True,
            )
            return
        self._chunk_bytes = int(self.sample_rate * _CHUNK_SECS) * 2
        await self._connect()

    async def stop(self, frame):
        """Finish the session gracefully, waiting briefly for the last transcript.

        Args:
            frame: The end frame.
        """
        await self._finish_input()
        await super().stop(frame)

    async def _finish_input(self) -> None:
        """Send buffered audio and input.done, then wait for transcript.done."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._send_buffered_audio()
            self._final_transcript.clear()
            # The server closes once the final transcript is out; that close
            # must not look like a dropped connection to reconnect.
            self._disconnecting = True
            await self._websocket.send(json.dumps({"type": "input.done"}))
            await asyncio.wait_for(self._final_transcript.wait(), _FINAL_TRANSCRIPT_TIMEOUT_SECS)
        except TimeoutError:
            logger.debug(f"{self}: no final transcript within {_FINAL_TRANSCRIPT_TIMEOUT_SECS}s")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self}: could not finish the session cleanly: {e}")

    #
    # Audio
    #

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Buffer audio and send it in chunks.

        Args:
            audio: Raw 16-bit mono PCM at the service's sample rate.

        Yields:
            None; transcripts arrive on the receive task.
        """
        self._audio_buffer.extend(audio)
        if len(self._audio_buffer) >= self._chunk_bytes:
            try:
                await self._send_buffered_audio()
            except Exception as e:  # noqa: BLE001
                yield ErrorFrame(error=f"Chariot STT: could not send audio: {e}")
        yield None

    async def _send_buffered_audio(self) -> None:
        if not self._audio_buffer:
            return
        # While the connection is down the receive loop reconnects; audio that
        # arrives meanwhile is dropped rather than queued indefinitely.
        if self._websocket and self._websocket.state is State.OPEN:
            await self._websocket.send(bytes(self._audio_buffer))
            self._uncommitted = True
        self._audio_buffer.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Finalize the utterance when Pipecat's VAD hears the user stop.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self._commit_strategy == "manual":
            await self._commit()

    async def _commit(self) -> None:
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._send_buffered_audio()
            if not self._uncommitted:
                return
            self.request_finalize()
            await self._websocket.send(json.dumps({"type": "input.commit"}))
            self._uncommitted = False
            self._pending_commits += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self}: could not finalize the utterance: {e}")

    async def _send_keepalive(self, silence: bytes):
        """Keep a quiet session open with a keepalive message rather than audio.

        Pipecat sends this only when no audio has flowed for a while, for
        example while the input is muted. A message costs nothing, whereas
        silence would be transcribed and billed.
        """
        if self._websocket is None:
            return
        await self._websocket.send(json.dumps({"type": "keepalive"}))

    #
    # Connection
    #

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        # Started even when the connection failed: with no socket the receive
        # loop goes straight to its reconnect path.
        if not self._receive_task or self._receive_task.done():
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Chariot STT")
            self._audio_buffer.clear()
            self._uncommitted = False
            self._pending_commits = 0
            self._closing_error = None
            self._partial = ""
            self._websocket = await self._websocket_connect(
                self._websocket_url(),
                additional_headers={**self._extra_headers, "chariotai-api-key": self._api_key},
                user_agent_header=user_agent(),
            )
            await self._call_event_handler("on_connected")
        except Exception as e:  # noqa: BLE001
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Chariot STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Chariot STT")
                await self._websocket.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise RuntimeError("Websocket not connected")

    #
    # Receiving
    #

    async def _receive_messages(self):
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
                    "Chariot STT: credit balance exhausted. Top up at "
                    "https://platform.chariot.in and start a new session."
                )
                return
            if code == _CLOSE_CODE_POLICY and not any(
                r in reason.lower() for r in _RECOVERABLE_POLICY_REASONS
            ):
                await self._handle_refusal(
                    f"Chariot STT refused the session: {reason or 'policy violation'}"
                )
                return
            # Anything else (all sessions busy, the session's maximum length,
            # an idle or backlog limit, a server error) clears on a new
            # session, so the base class reconnects with backoff.
            if reason:
                logger.warning(f"{self}: connection closed ({code}): {reason}")
            raise

    async def _handle_refusal(self, message: str) -> None:
        """Report a refusal that reconnecting would only repeat, and stop.

        A bad key, a bad setting or an empty balance would refuse every new
        session the same way, so the service stops for this session.
        """
        self._final_transcript.set()
        self._disconnecting = True
        await self.push_error(error_msg=message, force_treat_as_permanent=True)

    async def _handle_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "transcript.delta":
            await self._on_delta(event)
        elif event_type == "transcript.committed":
            await self._on_final(event, from_commit=True)
        elif event_type == "transcript.done":
            await self._on_final(event, from_commit=False)
            self._final_transcript.set()
        elif event_type == "session.started":
            logger.debug(f"{self}: session {event.get('session_id')} started")
        elif event_type == "input_audio_buffer.speech_started":
            await self._call_event_handler("on_speech_started", event)
        elif event_type == "input_audio_buffer.speech_stopped":
            await self._call_event_handler("on_speech_stopped", event)
        elif event_type == "session.expiring":
            logger.info(f"{self}: session ends in {event.get('seconds_remaining')}s")
            await self._call_event_handler("on_session_expiring", event)
        elif event_type == "error":
            message = event.get("message", "unknown error")
            if event.get("code"):
                # A coded error ends the session: the server closes right
                # after it, and the close reports it once.
                self._closing_error = message
                return
            await self.push_error(error_msg=f"Chariot STT error: {message}")

    def _language(self) -> Language | None:
        code = self._settings.language
        try:
            return Language(code) if code else None
        except ValueError:
            return None

    async def _on_delta(self, event: dict) -> None:
        # Deltas only ever append to the utterance in progress.
        self._partial += event.get("text", "")
        text = self._partial.strip()
        if text:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), self._language(), result=event
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record a final transcript for tracing."""

    async def _on_final(self, event: dict, *, from_commit: bool) -> None:
        """A finalized utterance. Its text replaces the interim text so far."""
        self._partial = ""
        if from_commit and self._commit_strategy == "manual":
            # An answer to one of our commits. Once every commit has come
            # back the transcript is complete, so the turn can end without
            # waiting out a timeout.
            self._pending_commits = max(0, self._pending_commits - 1)
            if self._pending_commits == 0:
                self.confirm_finalize()
        text = event.get("text", "").strip()
        if not text:
            return
        language = self._language()
        await self._handle_transcription(text, True, language)
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(text, self._user_id, time_now_iso8601(), language, result=event)
        )
