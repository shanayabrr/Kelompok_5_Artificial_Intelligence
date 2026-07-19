import streamlit as st
import os
import traceback

# Import Modules
from modules.loader import DocumentLoader
from modules.splitter import DocumentSplitter
from modules.embeddings import EmbeddingManager
from modules.retriever import VectorStoreManager
from modules.ollama_models import OllamaModelManager
from modules.rag_chain import RAGChain
from modules.benchmark import ModelBenchmark

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="RAG Chatbot SLMs", page_icon="🤖", layout="wide")

# CSS untuk UI Modern
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .css-1d391kg {
        background-color: #1e1e2e;
    }
    h1, h2, h3 {
        color: #cdd6f4;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #89b4fa;
        color: #1e1e2e;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #b4befe;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False

# Sidebar: Konfigurasi dan Upload Dokumen
with st.sidebar:
    st.image("https://ollama.com/public/icon-64x64.png", width=50)
    st.title("⚙️ Konfigurasi RAG")
    
    st.header("1. Pilih Model")
    selected_model = st.selectbox(
        "Pilih Small Language Model (SLM):",
        options=["phi3", "gemma:2b"],
        index=0
    )
    
    st.header("2. Upload Dokumen")
    uploaded_files = st.file_uploader(
        "Upload PDF atau TXT untuk Knowledge Base",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if st.button("🗂️ Index Documents"):
        if not uploaded_files:
            st.error("Silakan unggah dokumen terlebih dahulu.")
        else:
            with st.spinner("Memproses dan mengindeks dokumen..."):
                try:
                    # Inisialisasi Modul
                    loader = DocumentLoader()
                    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
                    embedding_mgr = EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")
                    embeddings = embedding_mgr.get_embeddings()
                    vector_mgr = VectorStoreManager(embeddings=embeddings)
                    
                    # Kosongkan vector db lama agar clean
                    vector_mgr.clear_database()
                    
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        file_path = loader.save_uploaded_file(uploaded_file)
                        docs = loader.load_document(file_path)
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        vector_mgr.add_documents(all_chunks)
                        st.session_state.is_indexed = True
                        st.success(f"Berhasil mengindeks {len(all_chunks)} chunk dari {len(uploaded_files)} dokumen!")
                    else:
                        st.warning("Tidak ada teks yang dapat diekstrak dari dokumen.")
                        
                except Exception as e:
                    st.error(f"Gagal mengindeks dokumen: {e}")
                    st.error(traceback.format_exc())

    st.markdown("---")
    st.markdown("Cek Dashboard Evaluasi di `compare_models.py`")

# Main Area: Chat Interface
st.title("🤖 Chatbot RAG Berbasis Ollama")
st.markdown(f"**Model Aktif:** `{selected_model}` | **Status Knowledge Base:** `{'✅ Indexed' if st.session_state.is_indexed else '❌ Kosong'}`")

# Menampilkan Riwayat Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Chat
if prompt := st.chat_input("Tanyakan sesuatu berdasarkan dokumen..."):
    # Cek koneksi Ollama
    ollama_mgr = OllamaModelManager()
    if not ollama_mgr.check_connection():
        st.error("⚠️ Ollama tidak berjalan. Pastikan aplikasi Ollama aktif (http://localhost:11434).")
    elif not st.session_state.is_indexed:
        st.warning("⚠️ Dokumen belum diindeks. Silakan unggah dan klik 'Index Documents' di sidebar.")
    else:
        # Tambahkan pertanyaan ke riwayat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Menghasilkan jawaban menggunakan {selected_model}..."):
                try:
                    # Inisialisasi komponen RAG
                    embedding_mgr = EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")
                    vector_mgr = VectorStoreManager(embeddings=embedding_mgr.get_embeddings())
                    retriever = vector_mgr.get_retriever(top_k=3)
                    
                    llm = ollama_mgr.get_llm(model_name=selected_model)
                    rag_chain = RAGChain(retriever=retriever, llm=llm)
                    
                    # Setup evaluator untuk Benchmark (menggunakan model yang sama)
                    evaluator_llm = ollama_mgr.get_chat_model(model_name=selected_model)
                    
                    # Eksekusi RAG dan Logging Benchmark
                    benchmark = ModelBenchmark()
                    # Menjalankan evaluasi Ragas memakan waktu cukup lama, diimplementasikan secara langsung sesuai request
                    answer = benchmark.measure_and_log(
                        model_name=selected_model,
                        question=prompt,
                        rag_chain=rag_chain,
                        retriever=retriever,
                        evaluator_llm=evaluator_llm,
                        evaluator_embeddings=embedding_mgr.get_embeddings()
                    )
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Terjadi kesalahan saat memproses pertanyaan: {e}"
                    st.error(error_msg)
                    st.error(traceback.format_exc())
