#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

The example connects between client and server using a P2P WebRTC connection.

Run the bot using::

    python bot.py
"""

import os
import asyncio
import aiohttp

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading AI models (30-40 seconds first run, <2 seconds after)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transcriptions.language import Language

logger.info("✅ Pipeline components loaded")

logger.info("Loading WebRTC transport...")
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    session = aiohttp.ClientSession()
    try:
        logger.info(f"Starting bot")

        stt = WhisperSTTService(model=WhisperModel.SMALL, language=Language.ES, device="cuda") #device="cuda"

        # Make sure Piper TTS server is running: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
        # https://huggingface.co/rhasspy/piper-voices/tree/main/
        # $python -m piper.download_voices es_MX-claude-high
        # $python -m piper.http_server -m es_MX-claude-high
        tts = PiperTTSService(
            base_url=os.getenv("PIPER_BASE_URL", "http://localhost:5000"),
            aiohttp_session=session,
            sample_rate=int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
        )

        llm = OLLamaLLMService(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            params=OLLamaLLMService.InputParams(
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000"))
            )
        )

        messages = [
            {
                "role": "system",
                "content": "Eres un asistente AI amigable. Responde de manera natural y mantén tus respuestas conversacionales.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI processor
                stt,
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Di hola y preséntate brevemente. No pongas headers."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)
    finally:
        # ensure aiohttp session closed
        await session.close()
        logger.info("Session closed, bot shutdown complete.")


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()