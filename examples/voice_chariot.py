#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""A browser voice agent that speaks with a Chariot voice.

Pipeline: browser microphone -> OpenAI STT -> OpenAI LLM -> Chariot TTS ->
browser speakers, over WebRTC, with Silero VAD driving interruptions.

Run it, open the printed URL, and talk. Interrupt it mid sentence and it stops
immediately: Chariot's protocol has no cancel message, so the service closes
the socket, which stops server side generation.

Install:
    uv pip install "pipecat-ai[openai,silero,webrtc,runner]" pipecat-chariot

Environment (see .env.example):
    CHARIOT_API_KEY   required
    CHARIOT_VOICE_ID  optional, defaults to a Chariot stock voice
    OPENAI_API_KEY    required, used for speech to text and the LLM

Usage:
    python examples/voice_chariot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_chariot import ChariotTTSService

load_dotenv(override=True)

# Browser WebRTC, which the runner serves with a prebuilt UI, so the example
# needs no third party account. ChariotTTSService itself is transport agnostic:
# it sits between the LLM and transport.output() and never sees the transport.
# To demo another one, add its key here and install its extra, for example
# "daily": lambda: DailyParams(...) with pipecat-ai[daily].
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = OpenAISTTService(api_key=os.environ["OPENAI_API_KEY"])

    # Voice UUIDs come from GET https://api.chariot.in/v1/voices. Any voice
    # speaks any supported language, so there is no language setting: the
    # audio follows the text. Defaults to a stock voice when unset.
    tts = ChariotTTSService(
        api_key=os.environ["CHARIOT_API_KEY"],
        voice_id=os.getenv("CHARIOT_VOICE_ID"),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice call. Keep replies "
                "short. Your output is spoken aloud, so avoid bullet points, "
                "markdown, and emojis."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "user", "content": "Greet the user and invite them to talk."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point used by the Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
