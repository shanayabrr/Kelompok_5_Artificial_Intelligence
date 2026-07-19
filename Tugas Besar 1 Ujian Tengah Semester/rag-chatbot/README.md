# RAG Chatbot Berbasis SLMs (Small Language Models) dengan Ollama

Proyek ini adalah implementasi sistem Chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang sepenuhnya berjalan secara **lokal**. Sistem ini digunakan untuk melakukan studi komparatif efisiensi dan akurasi antara dua model bahasa kecil (SLMs), yaitu **Phi-3** dan **Gemma 2B**.

---

## 🛠️ Tools dan Teknologi yang Digunakan

1. **Bahasa Pemrograman:** Python 3.11+
2. **Framework RAG:** LangChain
3. **Local LLM Engine:** Ollama (http://localhost:11434)
4. **Vector Database:** ChromaDB (Persisten di lokal)
5. **Model Embedding:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
6. **Frontend & Dashboard:** Streamlit & Plotly
7. **Framework Evaluasi:** Ragas (Untuk mengukur Context Precision, Answer Relevancy, dan Faithfulness)

---

## ⚙️ Persiapan & Instalasi

Sebelum menjalankan aplikasi, pastikan Anda telah menginstal dua perangkat lunak berikut di komputer Anda:
1. **Python (Minimal versi 3.10 atau 3.11)**
2. **Ollama** (Unduh dan instal dari [https://ollama.com](https://ollama.com))

### Langkah-Langkah Instalasi:

**1. Clone atau Buka Folder Proyek**
Pastikan Anda berada di dalam folder proyek di terminal (Command Prompt / PowerShell / Terminal VSCode):
```bash
cd "d:/Pa Hendri/UTs/rag-chatbot"
```

**2. Instal Library Python**
Instal seluruh dependensi (termasuk LangChain, Streamlit, ChromaDB, dll) dengan perintah:
```bash
pip install -r requirements.txt
```

**3. Unduh Model SLMs via Ollama**
Pastikan aplikasi Ollama Anda sudah menyala di *background*, lalu jalankan perintah ini di terminal untuk mengunduh model:
```bash
ollama pull phi3
ollama pull gemma:2b
```

---

## 🚀 Cara Menjalankan & Mengetes Aplikasi

Aplikasi ini dibagi menjadi 2 antarmuka: **Aplikasi Chat** dan **Dashboard Evaluasi**.

### Tahap 1: Menjalankan Aplikasi Chatbot (app.py)
Jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```
**Langkah Pengetesan:**
1. Buka URL yang muncul di terminal (biasanya `http://localhost:8501`).
2. Di **Sidebar**, pilih model yang ingin Anda tes (Phi-3 atau Gemma 2B).
3. Unggah beberapa file PDF atau TXT.
4. Klik tombol **Index Documents**. Tunggu hingga muncul notifikasi sukses.
5. Cobalah ketik pertanyaan di kolom chat terkait isi dokumen yang baru saja diunggah.
6. Chatbot akan memberikan jawaban. *(Catatan: Karena sistem ini mengukur efisiensi sistem dan Ragas secara langsung, respons mungkin membutuhkan waktu ekstra).*

### Tahap 2: Melihat Hasil Benchmark (compare_models.py)
Setelah Anda mencoba beberapa pertanyaan di Tahap 1, semua hasil log kecepatan dan akurasinya akan tersimpan di file `results.csv`.

Buka terminal baru, dan jalankan:
```bash
streamlit run compare_models.py
```
**Langkah Pengetesan:**
1. Buka URL yang muncul (biasanya `http://localhost:8502`).
2. Dashboard ini akan menampilkan grafik interaktif dari Plotly.
3. Anda bisa membandingkan **Response Time**, **Penggunaan RAM**, **Tokens per Second**, dan **Skor Akurasi Ragas** dari kedua model tersebut secara komprehensif.

---

## 📁 Struktur Folder Utama
- `app.py`: File utama untuk antarmuka chatbot dan unggah dokumen.
- `compare_models.py`: File utama untuk dashboard komparasi hasil.
- `modules/`: Kumpulan modul backend berbasis OOP (Loader, Splitter, Retriever, RAG Chain, Benchmark, dll).
- `data/documents/`: Folder tempat file-file PDF/TXT diunggah.
- `vectorstore/`: Folder tempat database vektor ChromaDB disimpan secara lokal.
- `results.csv`: File otomatis yang menyimpan log pengujian (RAM, waktu, skor Ragas).
