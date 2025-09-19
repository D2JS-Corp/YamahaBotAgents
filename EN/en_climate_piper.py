#!/usr/bin/env python3
# bot.py
"""
Pipecat voice bot example using:
- Whisper (local) for STT
- Ollama (local) for LLM
- Piper (local HTTP server) for TTS
"""
import os
import sys
import asyncio
import aiohttp
from loguru import logger

# Pipecat core
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

# VAD (Silero)
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Services (STT / LLM / TTS)
from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService

# Context & tools
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# RTVI (optional - keep if you need client/server communications helpers)
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

logger.remove()  # default handler
logger.add(sys.stderr, level="INFO")

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot...")
    session = aiohttp.ClientSession()
    try:
        # --- STT (Whisper) ---
        # pick the whisper model you want, e.g. WhisperModel.DISTIL_LARGE_V2 or others
        stt = WhisperSTTService(model=Model.TINY)

        # --- LLM (Ollama) ---
        llm = OLLamaLLMService(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            params=OLLamaLLMService.InputParams(
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000"))
            )
        )

        # --- TTS (Piper) ---
        # Make sure Piper TTS server is running: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
        tts = PiperTTSService(
            base_url=os.getenv("PIPER_BASE_URL", "http://localhost:5000"),
            aiohttp_session=session,
            sample_rate=int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
        )

        # Example: simple function schema / tools
        weather_function = FunctionSchema(
            name="get_current_weather",
            description="Get current weather information",
            properties={
                "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
                "format": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
            },
            required=["location", "format"],
        )
        tools = ToolsSchema(standard_tools=[weather_function])

        # Create the assistant context (initial system message)
        context = OpenAILLMContext(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, concise assistant running locally."
                }
            ],
            tools=tools
        )

        # Create context aggregator tied to the chosen LLM service
        context_aggregator = llm.create_context_aggregator(context)

        # Register a local function handler for function-calls from the LLM
        async def fetch_weather(params):
            # Example local stub: implement your real lookup here
            location = params.arguments.get("location")
            fmt = params.arguments.get("format", "celsius")
            # return via callback expected shape
            await params.result_callback({"conditions": "sunny", "temperature": "22°C"})

        llm.register_function("get_current_weather", fetch_weather)

        # Optional: RTVI (if you want the extra client handshake/observability)
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Build pipeline - order matters
        pipeline = Pipeline(
            [
                transport.input(),               # incoming audio frames from client
                rtvi,                            # optional RTVI processing
                stt,                             # speech -> text
                context_aggregator.user(),       # add user messages to context
                llm,                             # generate assistant response
                tts,                             # text -> audio
                transport.output(),              # send audio back to client
                context_aggregator.assistant(),  # store assistant spoken responses in context
            ]
        )

        # Create task and runner
        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport_obj, client):
            logger.info("Client connected - starting initial prompt.")
            # Add a system/user starter message to context and kick off the pipeline
            context.messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
            # queue initial frame so the pipeline speaks the opener
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport_obj, client):
            logger.info("Client disconnected - cancelling task.")
            await task.cancel()

        # Runner: run until cancelled / SIGINT handled according to runner_args
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)

    finally:
        # ensure aiohttp session closed
        await session.close()
        logger.info("Session closed, bot shutdown complete.")


async def bot(runner_args: RunnerArguments):
    """Entry point used by the pipecat development runner."""
    # Configure a Silero VAD analyzer (sample_rate must match your audio upstream)
    vad = SileroVADAnalyzer(sample_rate=16000)

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=vad,
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    # pipecat runner will call `bot(...)` for you when you run `pipecat` runner entry
    from pipecat.runner.run import main
    main()
