"""
embeddings.py
=============
Modul untuk menyediakan model embedding berbasis HuggingFace
(sentence-transformers/all-MiniLM-L6-v2) yang digunakan untuk mengubah
teks menjadi representasi vektor, baik saat indexing maupun saat query.

Model di-cache (singleton) agar hanya dimuat sekali ke memori selama
siklus hidup aplikasi, karena proses loading model cukup mahal.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from modules.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Membuat (atau mengambil dari cache) instance model embedding HuggingFace.

    Menggunakan `lru_cache` sehingga model hanya di-load satu kali per proses,
    mempercepat startup ulang dan menghemat memori.

    Returns:
        Instance HuggingFaceEmbeddings yang siap digunakan oleh ChromaDB.

    Raises:
        RuntimeError: Jika model embedding gagal dimuat (misal karena
                      masalah koneksi internet saat pertama kali download
                      atau dependensi sentence-transformers tidak terpasang).
    """
    try:
        logger.info("Memuat model embedding: %s", EMBEDDING_MODEL_NAME)
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Model embedding berhasil dimuat.")
        return embedding_model

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal memuat model embedding '%s': %s", EMBEDDING_MODEL_NAME, exc)
        raise RuntimeError(
            f"Tidak dapat memuat model embedding '{EMBEDDING_MODEL_NAME}'. "
            "Pastikan paket sentence-transformers terpasang dan koneksi internet "
            "tersedia untuk unduhan model pertama kali."
        ) from exc
