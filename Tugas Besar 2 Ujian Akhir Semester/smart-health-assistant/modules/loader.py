"""
loader.py
=========
Modul untuk memuat dokumen sumber (PDF & TXT) dari folder `data/documents/`.

Setiap dokumen yang berhasil dimuat akan memiliki metadata `source`
yang berisi nama file asli, sehingga dapat dilacak kembali saat
proses retrieval (untuk fitur "Dokumen Sumber" di frontend).
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from modules.config import DATA_DIR

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _load_single_file(file_path: Path) -> List[Document]:
    """
    Memuat satu file (PDF atau TXT) menjadi daftar objek Document LangChain.

    Args:
        file_path: Path lengkap menuju file yang akan dimuat.

    Returns:
        List objek Document. Mengembalikan list kosong jika gagal dimuat
        atau format file tidak didukung.
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            logger.warning("Format file tidak didukung, dilewati: %s", file_path.name)
            return []

        documents = loader.load()

        # Normalisasi metadata: pastikan setiap chunk punya nama file asli
        for doc in documents:
            doc.metadata["source"] = file_path.name

        logger.info("Berhasil memuat %s (%d halaman/bagian)", file_path.name, len(documents))
        return documents

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal memuat file %s: %s", file_path.name, exc)
        return []


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Memuat seluruh dokumen PDF/TXT yang ada di dalam folder `data_dir`.

    Args:
        data_dir: Direktori tempat dokumen Kemenkes RI disimpan.
                  Default mengambil dari konfigurasi global (DATA_DIR).

    Returns:
        List gabungan seluruh objek Document dari semua file yang berhasil dimuat.

    Raises:
        FileNotFoundError: Jika direktori data tidak ditemukan sama sekali.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Direktori dokumen tidak ditemukan: {data_dir}")

    files = [
        f for f in sorted(data_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(
            "Tidak ada dokumen PDF/TXT ditemukan di %s. "
            "Pastikan 10 dokumen Kemenkes RI sudah diletakkan di folder ini.",
            data_dir,
        )
        return []

    all_documents: List[Document] = []
    for file_path in files:
        all_documents.extend(_load_single_file(file_path))

    logger.info(
        "Total %d dokumen berhasil dimuat dari %d file di %s",
        len(all_documents), len(files), data_dir,
    )
    return all_documents
