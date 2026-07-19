# 🏥 Smart Health Assistant — RAG Chatbot Kemenkes RI

Sistem RAG (Retrieval-Augmented Generation) Chatbot production-ready yang menjawab
pertanyaan kesehatan berdasarkan dokumen resmi Kemenkes RI, menggunakan:

- **FastAPI** — backend REST + streaming (SSE)
- **Streamlit** — frontend chat UI
- **LangChain** — orkestrasi pipeline RAG
- **ChromaDB** — vector database lokal (persisten)
- **sentence-transformers/all-MiniLM-L6-v2** — model embedding
- **Ollama + Phi-3-mini (phi3)** — LLM lokal untuk generation

## 📁 Struktur Proyek

```
smart-health-assistant/
├── main.py                 # FastAPI backend
├── app.py                  # Streamlit frontend
├── ingest.py                # Script indexing dokumen -> ChromaDB
├── requirements.txt
├── modules/
│   ├── config.py            # Konfigurasi terpusat
│   ├── loader.py             # Pemuatan dokumen PDF/TXT
│   ├── splitter.py           # Chunking teks
│   ├── embeddings.py         # Model embedding HuggingFace
│   ├── retriever.py          # ChromaDB vector store
│   └── rag_chain.py          # Pipeline RAG LangChain
├── data/documents/           # 10 dokumen Kemenkes RI (PDF/TXT) diletakkan di sini
└── vectorstore/               # Folder ChromaDB persisten (auto-generated)
```

## 🚀 Cara Menjalankan

### 1. Persiapan environment

```bash
python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install & jalankan Ollama + model Phi-3

```bash
# Install Ollama sesuai OS: https://ollama.com/download
ollama pull phi3
ollama serve        # pastikan runtime aktif di http://localhost:11434
```

### 3. Letakkan dokumen sumber

Salin 10 dokumen Kemenkes RI (format `.pdf` atau `.txt`) ke dalam folder:

```
data/documents/
```

### 4. Jalankan proses ingestion (index dokumen ke ChromaDB)

```bash
python ingest.py
```

Jalankan ulang perintah ini setiap kali dokumen di `data/documents/` berubah.

### 5. Jalankan backend FastAPI

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Cek dokumentasi API interaktif di: `http://localhost:8000/docs`

### 6. Jalankan frontend Streamlit (terminal terpisah)

```bash
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## 🔌 Endpoint API

| Method | Endpoint       | Deskripsi                                              |
|--------|----------------|----------------------------------------------------------|
| GET    | `/health`      | Status koneksi Ollama & ChromaDB                          |
| POST   | `/chat`        | Jawaban RAG lengkap + metadata dokumen sumber              |
| POST   | `/chat/stream` | Jawaban RAG streaming token-by-token (SSE)                 |

Contoh request `/chat`:

```json
{ "query": "Apa saja gejala demam berdarah?" }
```

## ⚠️ Disclaimer

Jawaban chatbot ini bersifat informatif berdasarkan dokumen resmi Kemenkes RI
dan **bukan pengganti** diagnosis atau konsultasi tenaga medis profesional.
