"""
splitter.py
===========
Modul untuk memecah (chunking) dokumen panjang menjadi potongan-potongan
teks yang lebih kecil menggunakan RecursiveCharacterTextSplitter,
sehingga cocok untuk proses embedding dan retrieval yang presisi.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Memecah daftar dokumen menjadi potongan-potongan (chunks) teks yang lebih kecil.

    Args:
        documents: List dokumen hasil dari modules.loader.load_documents().
        chunk_size: Ukuran maksimum tiap chunk (jumlah karakter). Default 500.
        chunk_overlap: Jumlah karakter overlap antar chunk. Default 50.

    Returns:
        List Document baru yang sudah terpotong, tetap membawa metadata
        `source` dari dokumen asalnya.

    Raises:
        ValueError: Jika `documents` kosong.
    """
    if not documents:
        raise ValueError("Daftar dokumen kosong, tidak ada yang bisa di-chunk.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)

        logger.info(
            "Chunking selesai: %d dokumen -> %d chunk (chunk_size=%d, overlap=%d)",
            len(documents), len(chunks), chunk_size, chunk_overlap,
        )
        return chunks

    except Exception as exc:  # noqa: BLE001
        logger.error("Gagal melakukan chunking dokumen: %s", exc)
        raise
