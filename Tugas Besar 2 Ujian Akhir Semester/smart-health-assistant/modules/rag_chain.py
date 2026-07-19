"""
rag_chain.py
============
Pipeline inti Retrieval-Augmented Generation (RAG) untuk Smart Health
Assistant. Modul ini menghubungkan retriever ChromaDB dengan LLM lokal
(Ollama - Phi-3-mini) menggunakan LangChain Expression Language (LCEL),
serta menyediakan fungsi untuk mode jawaban penuh maupun streaming.
"""

import logging
from functools import lru_cache
from typing import AsyncGenerator, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

from modules.config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """
    Membuat (atau mengambil dari cache) instance LLM Ollama (Phi-3-mini)
    yang terhubung ke runtime Ollama lokal.

    Returns:
        Instance ChatOllama siap pakai untuk invoke/astream.

    Raises:
        RuntimeError: Jika instance LLM gagal dibuat (bukan berarti Ollama
                      sudah pasti hidup, itu dicek terpisah lewat /health).
    """
    try:
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
        )
        logger.info("LLM Ollama '%s' berhasil diinisialisasi (%s)", OLLAMA_MODEL, OLLAMA_BASE_URL)
        return llm
    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal menginisialisasi LLM Ollama: %s", exc)
        raise RuntimeError(f"Gagal menginisialisasi LLM Ollama: {exc}") from exc


def format_docs(docs: List[Document]) -> str:
    """
    Menggabungkan potongan dokumen hasil retrieval menjadi satu blok teks
    konteks yang siap disisipkan ke dalam prompt.

    Args:
        docs: List Document hasil retrieval.

    Returns:
        String konteks gabungan, masing-masing chunk diberi label sumber.
    """
    if not docs:
        return "Tidak ada dokumen relevan yang ditemukan."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "tidak diketahui")
        parts.append(f"[Kutipan {i} - Sumber: {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def retrieve_context(query: str, retriever: VectorStoreRetriever) -> List[Document]:
    """
    Mengambil dokumen-dokumen relevan dari vector store berdasarkan query pengguna.

    Args:
        query: Pertanyaan/pesan pengguna.
        retriever: Objek retriever dari modules.retriever.get_retriever().

    Returns:
        List Document relevan (bisa kosong jika tidak ada yang relevan).

    Raises:
        RuntimeError: Jika proses retrieval gagal (misal ChromaDB tidak dapat diakses).
    """
    try:
        docs = retriever.invoke(query)
        logger.info("Retrieval untuk query '%s' mengembalikan %d dokumen", query, len(docs))
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal melakukan retrieval untuk query '%s': %s", query, exc)
        raise RuntimeError(f"Gagal melakukan retrieval dokumen: {exc}") from exc


async def generate_answer(
    query: str, retriever: VectorStoreRetriever
) -> Tuple[str, List[Document]]:
    """
    Menghasilkan jawaban LENGKAP (non-streaming) berdasarkan pipeline RAG:
    retrieval -> pembentukan prompt -> generation via LLM.

    Args:
        query: Pertanyaan pengguna.
        retriever: Objek retriever ChromaDB.

    Returns:
        Tuple berisi (jawaban_teks, list_dokumen_sumber).

    Raises:
        RuntimeError: Jika retrieval atau generation gagal.
    """
    docs = retrieve_context(query, retriever)
    context = format_docs(docs)

    try:
        llm = get_llm()
        chain = RAG_PROMPT | llm
        response = await chain.ainvoke({"context": context, "question": query})
        answer = response.content
        return answer, docs

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal menghasilkan jawaban untuk query '%s': %s", query, exc)
        raise RuntimeError(
            f"Gagal menghasilkan jawaban dari LLM Ollama ('{OLLAMA_MODEL}'). "
            f"Pastikan Ollama berjalan dan model sudah di-pull. Detail: {exc}"
        ) from exc


async def stream_answer(
    query: str, retriever: VectorStoreRetriever
) -> AsyncGenerator[str, None]:
    """
    Menghasilkan jawaban secara STREAMING (token demi token) berdasarkan
    pipeline RAG. Cocok digunakan untuk endpoint SSE `/chat/stream`.

    Args:
        query: Pertanyaan pengguna.
        retriever: Objek retriever ChromaDB.

    Yields:
        Potongan teks (token/chunk) jawaban secara berurutan.

    Raises:
        RuntimeError: Jika retrieval gagal sebelum streaming dimulai.
    """
    docs = retrieve_context(query, retriever)
    context = format_docs(docs)

    try:
        llm = get_llm()
        chain = RAG_PROMPT | llm

        async for chunk in chain.astream({"context": context, "question": query}):
            token = getattr(chunk, "content", "")
            if token:
                yield token

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal melakukan streaming jawaban untuk query '%s': %s", query, exc)
        raise RuntimeError(
            f"Gagal melakukan streaming jawaban dari LLM Ollama ('{OLLAMA_MODEL}'). "
            f"Pastikan Ollama berjalan dan model sudah di-pull. Detail: {exc}"
        ) from exc
