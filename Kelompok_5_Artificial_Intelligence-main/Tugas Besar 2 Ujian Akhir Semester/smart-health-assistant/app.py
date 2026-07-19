"""
app.py
======
Frontend Streamlit untuk Smart Health Assistant (Desain Elegan & Minimalis).

Menyediakan antarmuka chat modern yang terhubung ke backend FastAPI
(`main.py`) melalui HTTP, dengan efek streaming jawaban token-by-token
serta panel "Rujukan Dokumen" di bawah setiap jawaban asisten.
"""

import json
from typing import List, Dict, Any

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Konfigurasi Halaman
# --------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"
CHAT_STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT = 120

st.set_page_config(
    page_title="Smart Health Assistant · Kemenkes RI",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="collapsed", # Sidebar disembunyikan untuk tampilan clean
)

# --------------------------------------------------------------------------
# Elemen Gaya / Desain Elegan
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Reset font dasar global agar elegan */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        .main .block-container {
            padding-top: 1.5rem;
            max-width: 780px;
        }

        /* ---------- Navbar Atas Elegan ---------- */
        .shs-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 1.2rem;
            background: var(--background-color, #FFFFFF);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 16px;
            box-shadow: rgba(0, 0, 0, 0.04) 0px 4px 16px 0px;
            margin-bottom: 2rem;
        }
        .shs-nav-brand {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 700;
            font-size: 1.05rem;
            color: #0A443E;
        }
        .shs-nav-meta {
            font-size: 0.8rem;
            font-weight: 500;
            color: #627D77;
            background: rgba(10, 68, 62, 0.1);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
        }

        /* ---------- Header Konten ---------- */
        .shs-header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        .shs-title {
            font-weight: 700;
            font-size: 2.2rem;
            letter-spacing: -0.03em;
            /* Menggunakan warna teks bawaan tema agar kontras di dark/light mode */
            color: var(--text-color, #1A2E2A); 
            margin-bottom: 0.5rem;
        }
        .shs-subtitle {
            font-size: 0.95rem;
            color: var(--text-color, #627D77);
            opacity: 0.8;
            line-height: 1.6;
            max-width: 580px;
            margin: 0 auto;
        }

        /* ---------- Panel Dokumen Rujukan ---------- */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            border-radius: 14px !important;
            margin-top: 0.6rem;
        }
        .shs-source-card {
            background: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.1);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }
        .shs-source-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: #B37A2B;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .shs-source-body {
            font-size: 0.85rem;
            color: var(--text-color);
            opacity: 0.9;
            line-height: 1.5;
        }

        /* ---------- Footer & Disclaimer ---------- */
        .shs-footer {
            font-size: 0.78rem;
            color: var(--text-color);
            opacity: 0.7;
            text-align: center;
            margin-top: 3rem;
            padding: 1rem 0;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* Menghilangkan menu bawaan streamlit agar lebih bersih */
        #MainMenu, header, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Helper / Fungsi Pendukung
# --------------------------------------------------------------------------
def check_backend_health() -> Dict[str, Any]:
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}

def stream_chat_response(query: str):
    try:
        with requests.post(
            CHAT_STREAM_ENDPOINT,
            json={"query": query},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()
            current_event = None
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("event:"):
                    current_event = raw_line.replace("event:", "", 1).strip()
                elif raw_line.startswith("data:"):
                    raw_data = raw_line.replace("data:", "", 1).strip()
                    try:
                        payload = json.loads(raw_data)
                    except json.JSONDecodeError:
                        payload = {}

                    if current_event == "sources":
                        yield {"type": "sources", "data": payload.get("sources", [])}
                    elif current_event == "token":
                        yield {"type": "token", "data": payload.get("content", "")}
                    elif current_event == "done":
                        yield {"type": "done"}
                    elif current_event == "error":
                        yield {"type": "error", "data": payload.get("detail", "Error")}
    except Exception as exc:
        yield {"type": "error", "data": f"Koneksi gagal: {exc}"}

def render_sources(sources: List[Dict[str, str]]):
    if not sources:
        return
    with st.expander(f"📚 Lihat Rujukan Resmi ({len(sources)} Kutipan Document)"):
        for src in sources:
            content = src.get("content", "")
            snippet = content[:450] + ("..." if len(content) > 450 else "")
            st.markdown(
                f"""
                <div class="shs-source-card">
                    <div class="shs-source-meta">📁 {src.get('source', 'Kemenkes RI')}</div>
                    <div class="shs-source-body">"{snippet}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# Tampilan Komponen Navbar Atas
# --------------------------------------------------------------------------
health_data = check_backend_health()
is_healthy = health_data.get("status") == "healthy"
status_text = "Sistem Aktif" if is_healthy else "Sistem Offline"

st.markdown(
    f"""
    <div class="shs-nav">
        <div class="shs-nav-brand">⚕️ Smart Health Assistant</div>
        <div class="shs-nav-meta">{status_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Tampilan Header Utama
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="shs-header">
        <h1 class="shs-title">Bagaimana kami bisa membantu Anda hari ini?</h1>
        <p class="shs-subtitle">
            Asisten AI berbasis dokumen regulasi dan panduan resmi Kementerian Kesehatan RI. 
            Membantu verifikasi informasi medis Anda dengan cepat.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tombol bersihkan chat diletakkan secara subtil di pojok kanan bawah/atas jika dibutuhkan
col1, col2 = st.columns([6, 1.5])
with col2:
    if st.button("🗑️ Bersihkan Obrolan", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --------------------------------------------------------------------------
# Area Percakapan (Chat History)
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    avatar = "🩺" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

# --------------------------------------------------------------------------
# Input Obrolan Pengguna
# --------------------------------------------------------------------------
user_query = st.chat_input("Tulis pertanyaan kesehatan Anda di sini...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🩺"):
        placeholder = st.empty()
        accumulated_text = ""
        collected_sources: List[Dict[str, str]] = []
        error_message = None

        with st.spinner("Menganalisis dokumen resmi..."):
            for event in stream_chat_response(user_query):
                if event["type"] == "sources":
                    collected_sources = event["data"]
                elif event["type"] == "token":
                    accumulated_text += event["data"]
                    placeholder.markdown(accumulated_text + " ▌")
                elif event["type"] == "error":
                    error_message = event["data"]
                    break
                elif event["type"] == "done":
                    break

        if error_message:
            placeholder.error(f"⚠️ {error_message}")
            final_content = f"⚠️ {error_message}"
        else:
            placeholder.markdown(accumulated_text if accumulated_text else "_Gagal memuat jawaban._")
            final_content = accumulated_text
            render_sources(collected_sources)

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": final_content,
                "sources": collected_sources,
            }
        )

# Footer Disclaimer
st.markdown(
    '<div class="shs-footer">Pemberitahuan: Informasi ini bertujuan edukatif berdasarkan dokumen resmi, bukan pengganti diagnosis medis profesional.</div>',
    unsafe_allow_html=True,
)