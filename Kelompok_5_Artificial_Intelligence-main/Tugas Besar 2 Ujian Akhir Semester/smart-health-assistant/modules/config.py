"""
config.py
=========
Konfigurasi terpusat untuk seluruh aplikasi Smart Health Assistant.
Semua path, nama model, dan parameter chunking didefinisikan di sini
agar mudah diubah tanpa menyentuh kode logika bisnis.
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Path Proyek
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
PERSIST_DIR = BASE_DIR / "vectorstore"

# Pastikan folder wajib tersedia
DATA_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Konfigurasi Embedding
# ------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------------
# Konfigurasi Chunking
# ------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ------------------------------------------------------------------
# Konfigurasi ChromaDB
# ------------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "smart_health_assistant")
TOP_K = int(os.getenv("TOP_K", 4))

# ------------------------------------------------------------------
# Konfigurasi Ollama (LLM lokal)
# ------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

# ------------------------------------------------------------------
# Prompt Sistem
# ------------------------------------------------------------------
SYSTEM_PROMPT = """Anda adalah "Smart Health Assistant", asisten AI kesehatan \
yang membantu masyarakat memahami informasi kesehatan resmi berdasarkan \
dokumen-dokumen dari Kementerian Kesehatan Republik Indonesia (Kemenkes RI).

Aturan yang WAJIB dipatuhi:
1. Jawablah HANYA berdasarkan konteks dokumen yang diberikan di bawah ini.
2. Jika informasi tidak ditemukan dalam konteks, katakan dengan jujur bahwa \
Anda tidak memiliki informasi tersebut dalam dokumen rujukan, jangan mengarang jawaban.
3. Gunakan bahasa Indonesia yang jelas, sopan, dan mudah dipahami masyarakat umum.
4. Selalu sertakan disclaimer bahwa jawaban ini bersifat informatif dan BUKAN \
pengganti konsultasi/diagnosis dari tenaga medis profesional.
5. Jangan memberikan resep obat atau dosis spesifik tanpa dasar yang jelas dari dokumen.

Konteks dokumen:
{context}

Pertanyaan pengguna:
{question}

Jawaban Anda:"""
