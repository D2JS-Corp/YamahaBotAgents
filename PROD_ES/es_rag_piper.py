import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot with RAG (information.txt only)...\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import Frame, TranscriptionFrame

logger.info("Pipeline components loaded")

logger.info("Loading WebRTC transport...")
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

load_dotenv(override=True)

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8700")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "3"))
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

async def call_retriever(session: aiohttp.ClientSession, query: str, k: int = RETRIEVER_TOP_K):
    """Call the retriever service to get relevant context"""
    url = f"{RETRIEVER_URL}/search"
    params = {"q": query, "k": k}
    try:
        async with session.get(url, params=params, timeout=10) as r:
            r.raise_for_status()
            data = await r.json()
            logger.info(f"🔍 Retrieved {len(data)} chunks for query: '{query[:60]}...'")
            return data
    except Exception as e:
        logger.error(f"❌ Retriever error: {e}")
        return []


class RAGProcessor(FrameProcessor):
    """Processor that intercepts transcriptions and injects RAG context"""
    
    def __init__(self, session: aiohttp.ClientSession, llm_context: OpenAILLMContext, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.llm_context = llm_context  # This is the actual OpenAILLMContext object
        self.last_query = ""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Intercept transcription frames (user speech)
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            
            if user_text and user_text != self.last_query:
                self.last_query = user_text
                logger.info(f"💬 User query: '{user_text}'")
                
                # Call retriever
                results = await call_retriever(self.session, user_text, k=RETRIEVER_TOP_K)
                
                # Build context message
                if results:
                    context_parts = []
                    for i, r in enumerate(results, 1):
                        text = r.get("text", "").strip()
                        score = r.get("score", 0)
                        if text:
                            context_parts.append(f"[Fragmento {i}] (relevancia: {score:.3f})\n{text}")
                    
                    context_text = "\n\n".join(context_parts)
                    
                    rag_system_msg = {
                        "role": "system",
                        "content": f"""CONTEXTO RECUPERADO del archivo information.txt:

{context_text}

INSTRUCCIÓN CRÍTICA: 
- Usa EXCLUSIVAMENTE la información del contexto anterior para responder
- Si la respuesta NO está en el contexto, responde: "Lo siento, esa información no está disponible en mis documentos."
- NO inventes, NO asumas, NO agregues información externa
- Responde de forma natural pero SOLO con lo que dice el contexto
"""
                    }
                    
                    # Inject into context BEFORE the user message gets processed by LLM
                    self.llm_context.messages.append(rag_system_msg)
                    logger.info(f"📚 Injected {len(results)} context chunks into LLM context")
                    
                else:
                    # No results - tell LLM explicitly
                    no_context_msg = {
                        "role": "system",
                        "content": 'No se encontró información relevante. Responde exactamente: "Lo siento, esa información no está disponible en mis documentos."'
                    }
                    self.llm_context.messages.append(no_context_msg)
                    logger.warning("⚠️  No relevant context found in retriever")
        
        # Pass frame downstream
        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    session = aiohttp.ClientSession()
    try:
        logger.info("Starting bot with RAG system")

        # STT: Whisper for Spanish
        stt = WhisperSTTService(
            model=WhisperModel.SMALL, 
            language=Language.ES
        )

        # TTS: Piper
        # Make sure Piper TTS server is running: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
        # https://huggingface.co/rhasspy/piper-voices/tree/main/
        # $python -m piper.download_voices es_MX-claude-high
        # $python -m piper.http_server -m es_MX-claude-high
        tts = PiperTTSService(
            base_url=os.getenv("PIPER_BASE_URL", "http://localhost:5000"),
            aiohttp_session=session,
            sample_rate=int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
        )

        # LLM: Ollama with strict temperature
        llm = OLLamaLLMService(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=OLLAMA_BASE,
            params=OLLamaLLMService.InputParams(
                temperature=1,  # Zero temperature for deterministic responses
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "400"))
            )
        )

        # System prompt that enforces strict RAG behavior
        system_prompt = """Eres un asistente virtual que SOLAMENTE responde usando información del archivo information.txt.

REGLAS ABSOLUTAS:
1. Si recibes contexto recuperado del archivo, usa SOLO esa información
2. Si NO recibes contexto relevante, di: "Lo siento, esa información no está disponible en mis documentos."
3. NUNCA uses tu conocimiento general
4. NUNCA inventes información
5. Sé breve y directo

Responde en español de forma natural."""

        messages = [{"role": "system", "content": system_prompt}]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Create RAG processor with access to the actual context object
        # IMPORTANT: Pass 'context', not 'context_aggregator'
        rag_processor = RAGProcessor(
            session=session,
            llm_context=context
        )

        # Pipeline: RAG processor between STT and context aggregator
        pipeline = Pipeline([
            transport.input(),
            rtvi,
            stt,
            rag_processor,  # Intercepts transcriptions and injects context
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True, 
                enable_usage_metrics=True
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("✅ Client connected")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("❌ Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)

    finally:
        await session.close()
        logger.info("🔚 Bot shutdown complete")


async def bot(runner_args: RunnerArguments):
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