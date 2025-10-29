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
                
                # Build context message (solo bloque <context>…</context>)
                if results:
                    plain_chunks = []
                    for r in results:
                        text = (r.get("text") or "").strip()
                        if text:
                            plain_chunks.append(text)
                    context_text = "\n".join(plain_chunks).strip()

                    rag_system_msg = {
                        "role": "system",
                        "content": f"<context>\n{context_text}\n</context>"
                    }
                    
                    # Inject into context BEFORE the user message gets processed by LLM
                    self.llm_context.messages.append(rag_system_msg)
                    logger.info(f"📚 Injected {len(plain_chunks)} context chunks into LLM context")
                    
                else:
                    # Sin resultados: inyecta <context> vacío; el system_prompt decidirá la respuesta
                    empty_context_msg = {
                        "role": "system",
                        "content": "<context>\n\n</context>"
                    }
                    self.llm_context.messages.append(empty_context_msg)
                    logger.warning("⚠️  No context found: injected empty <context>")
        
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
        tts = PiperTTSService(
            base_url=os.getenv("PIPER_BASE_URL", "http://localhost:5000"),
            aiohttp_session=session,
            sample_rate=int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
        )

        # LLM: Ollama with strict temperature (se mantiene como en tu código)
        llm = OLLamaLLMService(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=OLLAMA_BASE,
            params=OLLamaLLMService.InputParams(
                temperature=1,  # se mantiene tu configuración original
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "400"))
            )
        )

        # System prompt minimalista: responde SOLO con lo que haya entre <context>...</context>
        system_prompt = """Eres un asistente que SOLO responde usando el TEXTO DE CONTEXTO entre <context>...</context>.
REGLAS:
1) Si <context> NO está vacío, ASUME que es relevante y responde usando SOLO ese contenido. No evalúes relevancia.
2) Solo puedes decir: "Lo siento, esa información no está en el contexto." si y solo si <context> está vacío.
3) No agregues datos externos ni inventes.
4) Responde en español, breve y directo (80–120 palabras por defecto), parafraseando el contexto.
5) Si hay varias piezas, combínalas y resume sin listas largas por defecto.
"""

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
