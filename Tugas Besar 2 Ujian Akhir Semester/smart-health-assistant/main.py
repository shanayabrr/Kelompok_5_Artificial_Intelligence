"""
main.py
=======
Backend FastAPI untuk Smart Health Assistant.

Menyediakan tiga endpoint utama:
- GET  /health       : cek status koneksi Ollama & ChromaDB
- POST /chat          : jawaban RAG lengkap beserta metadata sumber
- POST /chat/stream   : jawaban RAG secara streaming (Server-Sent Events)

Menjalankan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from modules.config import OLLAMA_BASE_URL, OLLAMA_MODEL, PERSIST_DIR
from modules.retriever import load_vectorstore, get_retriever, get_collection_count
from modules.rag_chain import generate_answer, retrieve_context, format_docs, RAG_PROMPT, get_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# State aplikasi (dimuat sekali saat startup agar efisien)
# --------------------------------------------------------------------------
app_state = {"vectorstore": None, "retriever": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle handler FastAPI: memuat vector store ChromaDB persisten
    dan menyiapkan retriever saat aplikasi start, agar tidak perlu
    reload berulang kali di setiap request.
    """
    logger.info("Memulai Smart Health Assistant backend...")
    try:
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)
        app_state["vectorstore"] = vectorstore
        app_state["retriever"] = retriever
        logger.info("Vector store & retriever berhasil dimuat saat startup.")
    except Exception as exc:  # noqa: BLE001
        # Aplikasi tetap dijalankan, tapi endpoint /chat akan menolak request
        # dengan pesan error yang jelas sampai vector store berhasil dimuat.
        logger.error("Gagal memuat vector store saat startup: %s", exc)
        app_state["vectorstore"] = None
        app_state["retriever"] = None

    yield

    logger.info("Menghentikan Smart Health Assistant backend...")


app = FastAPI(
    title="Smart Health Assistant API",
    description=(
        "Backend RAG Chatbot berbasis dokumen resmi Kemenkes RI. "
        "Menggunakan LangChain + ChromaDB + Ollama (Phi-3-mini)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Skema Request/Response
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Skema body request untuk endpoint /chat dan /chat/stream."""

    query: str = Field(..., min_length=1, description="Pertanyaan pengguna")


class SourceDocument(BaseModel):
    """Skema metadata dokumen sumber yang digunakan sebagai referensi jawaban."""

    source: str = Field(..., description="Nama file dokumen sumber")
    content: str = Field(..., description="Potongan teks (chunk) yang relevan")


class ChatResponse(BaseModel):
    """Skema response lengkap untuk endpoint /chat."""

    answer: str
    sources: List[SourceDocument]


class HealthResponse(BaseModel):
    """Skema response untuk endpoint /health."""

    status: str
    ollama: dict
    chromadb: dict


# --------------------------------------------------------------------------
# Helper internal
# --------------------------------------------------------------------------
def _ensure_retriever_ready():
    """Memastikan retriever sudah siap sebelum memproses chat request."""
    retriever = app_state.get("retriever")
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector store ChromaDB belum siap. Pastikan Anda sudah menjalankan "
                "proses ingestion (ingest.py) untuk mengindex dokumen di data/documents/."
            ),
        )
    return retriever


async def _check_ollama() -> dict:
    """Mengecek apakah runtime Ollama lokal aktif dan model tersedia."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            model_names = [m.get("name", "") for m in data.get("models", [])]
            model_available = any(OLLAMA_MODEL in name for name in model_names)
            return {
                "connected": True,
                "base_url": OLLAMA_BASE_URL,
                "model": OLLAMA_MODEL,
                "model_available": model_available,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama tidak dapat diakses: %s", exc)
        return {
            "connected": False,
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
            "error": str(exc),
        }


def _check_chromadb() -> dict:
    """Mengecek apakah ChromaDB persisten dapat diakses dan berisi data."""
    count = get_collection_count()
    if count is None:
        return {"connected": False, "persist_directory": str(PERSIST_DIR), "document_count": 0}
    return {"connected": True, "persist_directory": str(PERSIST_DIR), "document_count": count}


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Mengecek status kesehatan sistem:
    - Koneksi ke runtime Ollama lokal (dan ketersediaan model phi3)
    - Koneksi ke ChromaDB persisten (dan jumlah dokumen ter-index)
    """
    ollama_status = await _check_ollama()
    chroma_status = _check_chromadb()

    overall_status = (
        "healthy" if ollama_status["connected"] and chroma_status["connected"] else "degraded"
    )

    return HealthResponse(status=overall_status, ollama=ollama_status, chromadb=chroma_status)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Menerima query pengguna, menjalankan pipeline RAG (retrieval + generation),
    dan mengembalikan jawaban LENGKAP beserta metadata dokumen sumber.
    """
    retriever = _ensure_retriever_ready()

    try:
        answer, docs = await generate_answer(request.query, retriever)
        sources = [
            SourceDocument(
                source=doc.metadata.get("source", "tidak diketahui"),
                content=doc.page_content.strip(),
            )
            for doc in docs
        ]
        return ChatResponse(answer=answer, sources=sources)

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Kesalahan tak terduga pada /chat: %s", exc)
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {exc}") from exc


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Menerima query pengguna dan mengembalikan jawaban secara STREAMING
    (token-by-token) menggunakan format Server-Sent Events (SSE).

    Urutan event yang dikirim ke klien:
    1. event "sources" -> daftar dokumen sumber (dikirim di awal)
    2. event "token"   -> potongan teks jawaban (berulang kali)
    3. event "done"    -> penanda bahwa streaming telah selesai
    4. event "error"   -> dikirim jika terjadi kegagalan di tengah proses
    """
    retriever = _ensure_retriever_ready()

    async def event_generator():
        try:
            docs = retrieve_context(request.query, retriever)
            sources_payload = [
                {
                    "source": doc.metadata.get("source", "tidak diketahui"),
                    "content": doc.page_content.strip(),
                }
                for doc in docs
            ]
            yield {"event": "sources", "data": json.dumps({"sources": sources_payload})}

            context = format_docs(docs)
            llm = get_llm()
            chain = RAG_PROMPT | llm

            async for chunk in chain.astream({"context": context, "question": request.query}):
                token = getattr(chunk, "content", "")
                if token:
                    yield {"event": "token", "data": json.dumps({"content": token})}

            yield {"event": "done", "data": json.dumps({"status": "completed"})}

        except Exception as exc:  # noqa: BLE001
            logger.error("Kesalahan saat streaming untuk query '%s': %s", request.query, exc)
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}

    return EventSourceResponse(event_generator())


@app.get("/", tags=["Monitoring"])
async def root():
    """Endpoint dasar untuk memverifikasi bahwa server backend aktif."""
    return {
        "message": "Smart Health Assistant API aktif.",
        "docs": "/docs",
        "health": "/health",
    }
