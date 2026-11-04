#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Local stand-ins for the Chariot WebSocket APIs, and a pipeline harness.

The fakes follow the public protocols closely enough to exercise the
services end to end through real Pipecat pipelines, without a network or an
API key.
"""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner
from websockets.asyncio.server import serve

PCM = b"\x01\x00\xff\x7f" * 200


@dataclass
class Connection:
    """One client connection as the fake server saw it."""

    path: str
    query: dict
    headers: dict
    messages: list = field(default_factory=list)

    def of_type(self, message_type: str) -> list:
        return [m for m in self.messages if isinstance(m, dict) and m.get("type") == message_type]


class FakeTTSServer:
    """A minimal ``/v1/tts/ws-multi``.

    Each flush of a context with buffered text produces one segment:
    ``audio.start``, the PCM as ``audio.chunk`` messages and ``audio.done``.
    ``context.close`` answers with ``context.done`` (completed) once buffered
    text is generated, and ``context.cancel`` with ``context.done``
    (cancelled).
    """

    def __init__(self):
        self.connections: list[Connection] = []
        self.chunks: list[bytes] = [PCM[:401], PCM[401:]]  # odd split on purpose
        self.stall = False  # send one chunk per segment, then hold the rest
        self.pause_between_chunks = 0.0  # seconds; a slow generation
        self.close_on_text: int | None = None  # close with this code on first text
        self.context_error: tuple[str, str] | None = None  # end contexts with this error
        self.reject_open: str | None = None  # reject a context's first message
        self.received = asyncio.Queue()
        self._server = None

    async def __aenter__(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def _handle(self, ws):
        parts = urlsplit(ws.request.path)
        conn = Connection(
            path=parts.path,
            query={k: v[0] for k, v in parse_qs(parts.query).items()},
            headers=dict(ws.request.headers),
        )
        self.connections.append(conn)
        contexts: dict[str, dict] = {}

        async def send(message):
            await ws.send(json.dumps(message))

        await send(
            {
                "type": "session.ready",
                "session_id": f"s{len(self.connections)}",
                "max_contexts": 5,
                "sample_rate": 44100,
                "encoding": "pcm_s16le",
            }
        )

        async def generate(cid):
            ctx = contexts[cid]
            ctx["segment"] += 1
            seg = ctx["segment"]
            ctx["buffer"] = ""
            await send(
                {"type": "audio.start", "context_id": cid, "segment": seg, "sample_rate": 44100}
            )
            chunks = self.chunks[:1] if self.stall else self.chunks
            for n, chunk in enumerate(chunks):
                if n and self.pause_between_chunks:
                    await asyncio.sleep(self.pause_between_chunks)
                await send(
                    {
                        "type": "audio.chunk",
                        "context_id": cid,
                        "segment": seg,
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            if self.stall:
                return
            await send(
                {
                    "type": "audio.done",
                    "context_id": cid,
                    "segment": seg,
                    "credits_utilized": 1,
                    "error": False,
                }
            )

        async def done(cid, reason, **extra):
            ctx = contexts.pop(cid, None)
            if ctx is None:
                return
            await send(
                {
                    "type": "context.done",
                    "context_id": cid,
                    "reason": reason,
                    "segments": ctx["segment"],
                    "credits_utilized": ctx["segment"],
                    **extra,
                }
            )

        try:
            async for raw in ws:
                msg = json.loads(raw)
                conn.messages.append(msg)
                self.received.put_nowait(msg)
                kind = msg.get("type")
                cid = msg.get("context_id", "default")
                if kind == "ping":
                    await send({"type": "pong"})
                elif kind == "input.text":
                    if cid not in contexts:
                        if self.reject_open:
                            await send(
                                {
                                    "type": "error",
                                    "code": self.reject_open,
                                    "message": "no",
                                    "context_id": cid,
                                }
                            )
                            continue
                        contexts[cid] = {
                            "voice": msg.get("voice_id"),
                            "speed": msg.get("speed"),
                            "buffer": "",
                            "segment": 0,
                        }
                    if msg.get("text") and self.close_on_text is not None:
                        # Like the real server: an error event, then the close.
                        await send(
                            {
                                "type": "error",
                                "code": "insufficient_credits",
                                "message": "Workspace has 0 credits",
                            }
                        )
                        await ws.close(self.close_on_text, "insufficient credits")
                        return
                    contexts[cid]["buffer"] += msg.get("text", "")
                elif kind == "input.flush" and cid in contexts:
                    if self.context_error:
                        code, message = self.context_error
                        await done(cid, "error", code=code, message=message)
                    elif contexts[cid]["buffer"]:
                        await generate(cid)
                elif kind == "context.close" and cid in contexts:
                    if contexts[cid]["buffer"]:
                        await generate(cid)
                    if not self.stall:
                        await done(cid, "completed")
                elif kind == "context.cancel":
                    await done(cid, "cancelled")
        except Exception:  # noqa: BLE001, S110
            # The client went away mid conversation, which tests do on purpose.
            pass


class FakeSTTServer:
    """A minimal ``/v1/asr/ws``.

    Audio gets one ``transcript.delta`` per utterance, ``input.commit``
    answers with ``transcript.committed``, and ``input.done`` with
    ``transcript.done`` followed by a normal close.
    """

    def __init__(self):
        self.connections: list[Connection] = []
        self.close_on_connect: int | None = None
        self.close_reason = "refused"
        self.vad_commit_after_audio = False  # server VAD: commit on its own
        self.hold_commits = 1  # answer commits only once this many are pending
        self.received = asyncio.Queue()
        self._server = None

    async def __aenter__(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def _handle(self, ws):
        parts = urlsplit(ws.request.path)
        conn = Connection(
            path=parts.path,
            query={k: v[0] for k, v in parse_qs(parts.query).items()},
            headers=dict(ws.request.headers),
        )
        self.connections.append(conn)
        if self.close_on_connect is not None:
            code = self.close_on_connect
            if code == 4429:
                self.close_on_connect = None  # the next session finds a free slot
            # Like the real server: a coded error event, then the close.
            await ws.send(
                json.dumps({"type": "error", "code": "refused", "message": self.close_reason})
            )
            await ws.close(code, self.close_reason)
            return

        async def send(message):
            await ws.send(json.dumps(message))

        await send(
            {
                "type": "session.started",
                "session_id": f"a{len(self.connections)}",
                "sample_rate": int(conn.query["audio_format"].split("_")[1]),
                "commit_strategy": conn.query.get("commit_strategy"),
            }
        )
        segment = 0
        spoke = False
        held = 0
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    conn.messages.append(raw)
                    self.received.put_nowait(raw)
                    if not spoke:
                        spoke = True
                        if self.vad_commit_after_audio:
                            await send(
                                {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0}
                            )
                        await send({"type": "transcript.delta", "text": "namaste"})
                        if self.vad_commit_after_audio:
                            await send(
                                {"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 500}
                            )
                            await send(
                                {
                                    "type": "transcript.committed",
                                    "segment_id": segment,
                                    "text": "namaste duniya",
                                }
                            )
                            segment += 1
                    continue
                msg = json.loads(raw)
                conn.messages.append(msg)
                self.received.put_nowait(msg)
                kind = msg.get("type")
                if kind == "input.commit":
                    held += 1
                    spoke = False
                    if held < self.hold_commits:
                        continue
                    for _ in range(held):
                        text = f"segment {segment}" if self.hold_commits > 1 else "namaste duniya"
                        await send(
                            {"type": "transcript.committed", "segment_id": segment, "text": text}
                        )
                        segment += 1
                    held = 0
                elif kind == "input.done":
                    await send(
                        {"type": "transcript.done", "text": "the tail", "audio_seconds": 1.0}
                    )
                    await ws.close(1000)
                    return
        except Exception:  # noqa: BLE001, S110
            # The client went away mid conversation, which tests do on purpose.
            pass


class Capture(FrameProcessor):
    """Records the frames passing in one direction."""

    def __init__(self, direction: FrameDirection):
        super().__init__()
        self.direction = direction
        self.frames = []
        self.queue = asyncio.Queue()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == self.direction:
            self.frames.append(frame)
            self.queue.put_nowait(frame)
        await self.push_frame(frame, direction)

    async def until(self, frame_type, timeout: float = 5.0):
        async with asyncio.timeout(timeout):
            while True:
                frame = await self.queue.get()
                if isinstance(frame, frame_type):
                    return frame

    def of_type(self, frame_type) -> list:
        return [f for f in self.frames if isinstance(f, frame_type)]


@asynccontextmanager
async def run_pipeline(service, *, audio_in_sample_rate: int = 16000):
    """Run ``service`` between two captures; yields (worker, downstream, upstream)."""
    upstream = Capture(FrameDirection.UPSTREAM)
    downstream = Capture(FrameDirection.DOWNSTREAM)
    worker = PipelineWorker(
        Pipeline([upstream, service, downstream]),
        params=PipelineParams(
            audio_in_sample_rate=audio_in_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), 5)
        yield worker, downstream, upstream
    finally:
        await worker.cancel()
        await asyncio.wait_for(running, 5)


async def next_message(queue: asyncio.Queue, message_type: str, timeout: float = 5.0):
    """The next message of a type the fake server received."""
    async with asyncio.timeout(timeout):
        while True:
            msg = await queue.get()
            if isinstance(msg, dict) and msg.get("type") == message_type:
                return msg
