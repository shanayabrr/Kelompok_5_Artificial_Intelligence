"""
retriever.py
============
Modul untuk mengelola vector store ChromaDB: membangun index dari dokumen
(ingestion), memuat index yang sudah persisten, serta menyediakan objek
retriever untuk pipeline RAG.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma

from modules.config import PERSIST_DIR, COLLECTION_NAME, TOP_K
from modules.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def build_vectorstore(chunks: List[Document], batch_size: int = 4000) -> Chroma:
    """
    Membangun ulang vector store ChromaDB dari nol menggunakan potongan
    dokumen (chunks) yang diberikan, lalu menyimpannya secara persisten.

    Proses insert dilakukan secara BER-BATCH karena ChromaDB membatasi
    jumlah maksimum dokumen yang dapat dimasukkan dalam satu kali panggilan
    (biasanya sekitar 5000-an, tergantung versi). Batching juga membuat
    proses lebih tangguh untuk dataset besar.

    Args:
        chunks: List Document hasil dari modules.splitter.split_documents().
        batch_size: Jumlah maksimum chunk yang di-insert per batch. Default 4000
                    (aman di bawah limit umum ChromaDB ~5461).

    Returns:
        Instance Chroma yang sudah terisi dan tersimpan di disk.

    Raises:
        ValueError: Jika `chunks` kosong.
        RuntimeError: Jika proses indexing ke ChromaDB gagal.
    """
    if not chunks:
        raise ValueError("Tidak ada chunk untuk di-index ke ChromaDB.")

    try:
        embedding_model = get_embedding_model()
        logger.info(
            "Membangun vector store ChromaDB (%d chunk, batch_size=%d)...",
            len(chunks), batch_size,
        )

        # Batch pertama: inisialisasi vector store baru
        first_batch = chunks[:batch_size]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=str(PERSIST_DIR),
        )
        logger.info("Batch 1: %d/%d chunk ter-index", len(first_batch), len(chunks))

        # Batch selanjutnya: tambahkan ke vector store yang sama
        total_indexed = len(first_batch)
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            vectorstore.add_documents(batch)
            total_indexed += len(batch)
            logger.info("Batch berikutnya: %d/%d chunk ter-index", total_indexed, len(chunks))

        logger.info("Vector store berhasil dibangun & disimpan di %s", PERSIST_DIR)
        return vectorstore

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal membangun vector store: %s", exc)
        raise RuntimeError(f"Gagal membangun vector store ChromaDB: {exc}") from exc


def load_vectorstore() -> Chroma:
    """
    Memuat vector store ChromaDB yang sudah tersimpan secara persisten
    di disk (tanpa melakukan re-indexing).

    Returns:
        Instance Chroma yang terhubung ke direktori persisten.

    Raises:
        RuntimeError: Jika koneksi ke ChromaDB gagal (misal folder korup
                      atau model embedding tidak dapat dimuat).
    """
    try:
        embedding_model = get_embedding_model()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(PERSIST_DIR),
        )
        logger.info("Vector store berhasil dimuat dari %s", PERSIST_DIR)
        return vectorstore

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal memuat vector store dari %s: %s", PERSIST_DIR, exc)
        raise RuntimeError(f"Gagal memuat vector store ChromaDB: {exc}") from exc


def get_retriever(vectorstore: Chroma, k: int = TOP_K) -> VectorStoreRetriever:
    """
    Membuat objek retriever dari vector store untuk digunakan pada
    pipeline RAG (pencarian dokumen relevan berdasarkan similarity).

    Args:
        vectorstore: Instance Chroma yang sudah dimuat/dibangun.
        k: Jumlah dokumen teratas yang diambil per query. Default dari config (TOP_K).

    Returns:
        VectorStoreRetriever siap pakai.
    """
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_collection_count() -> Optional[int]:
    """
    Mengecek jumlah dokumen/chunk yang tersimpan di dalam koleksi ChromaDB.
    Berguna untuk endpoint `/health` guna memverifikasi koneksi ChromaDB.

    Returns:
        Jumlah item dalam koleksi, atau None jika ChromaDB tidak dapat diakses.
    """
    try:
        vectorstore = load_vectorstore()
        count = vectorstore._collection.count()  # noqa: SLF001
        return count
    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal mengecek jumlah koleksi ChromaDB: %s", exc)
        return None