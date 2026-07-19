"""
ingest.py
=========
Script CLI untuk melakukan proses INGESTION: memuat seluruh dokumen
Kemenkes RI dari `data/documents/`, memecahnya menjadi chunk, lalu
meng-index-nya ke dalam ChromaDB persisten di folder `vectorstore/`.

Jalankan SEKALI (atau setiap kali dokumen sumber berubah) sebelum
menjalankan backend FastAPI:

    python ingest.py

Setelah proses ini selesai, `main.py` dan `app.py` akan otomatis membaca
vector store yang sudah ter-index tanpa perlu upload manual dari UI.
"""

import logging
import sys

from modules.loader import load_documents
from modules.splitter import split_documents
from modules.retriever import build_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Menjalankan pipeline ingestion end-to-end.

    Returns:
        Kode exit: 0 jika sukses, 1 jika gagal.
    """
    logger.info("=== Memulai proses ingestion Smart Health Assistant ===")

    try:
        documents = load_documents()
        if not documents:
            logger.error(
                "Tidak ada dokumen ditemukan di data/documents/. "
                "Letakkan minimal 1 file PDF/TXT (idealnya 10 dokumen Kemenkes RI) lalu coba lagi."
            )
            return 1

        chunks = split_documents(documents)
        build_vectorstore(chunks)

        logger.info(
            "=== Ingestion selesai: %d dokumen -> %d chunk berhasil di-index ===",
            len(documents), len(chunks),
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.error("Proses ingestion gagal: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
