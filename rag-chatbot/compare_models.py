import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dashboard Benchmark SLMs", page_icon="📊", layout="wide")

st.title("📊 Dashboard Evaluasi: Phi-3 vs Gemma 2B")
st.markdown("Membandingkan efisiensi dan akurasi (Ragas Framework) dari model Small Language Models.")

log_file = "results.csv"

if not os.path.exists(log_file):
    st.warning("Belum ada data evaluasi. Silakan lakukan percakapan dengan chatbot di `app.py` terlebih dahulu.")
else:
    df = pd.read_csv(log_file)
    
    if df.empty:
        st.warning("Data evaluasi kosong.")
    else:
        # Menampilkan Dataset
        st.subheader("Data Riwayat Percakapan")
        st.dataframe(df)

        st.markdown("---")
        
        # Agregasi Rata-rata Metrik per Model
        avg_metrics = df.groupby('model').mean(numeric_only=True).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Rata-rata Response Time (Detik)")
            fig_time = px.bar(
                avg_metrics, x='model', y='response_time', 
                color='model', text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_time.update_layout(showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)

        with col2:
            st.subheader("💾 Rata-rata RAM Usage (MB)")
            fig_ram = px.bar(
                avg_metrics, x='model', y='ram_usage', 
                color='model', text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_ram.update_layout(showlegend=False)
            st.plotly_chart(fig_ram, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Perbandingan Akurasi (Skor Ragas 0-1)")

        col3, col4 = st.columns(2)

        with col3:
            fig_acc = px.bar(
                avg_metrics, x='model', y=['context_precision', 'answer_relevancy'], 
                barmode='group',
                title="Context Precision & Answer Relevancy",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col4:
            fig_faith = px.bar(
                avg_metrics, x='model', y='faithfulness_score',
                color='model', text_auto='.4f',
                title="Faithfulness Score (Tidak berhalusinasi)",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_faith, use_container_width=True)
            
        st.markdown("---")
        st.subheader("⚙️ Tokens per Second")
        fig_tps = px.box(
            df, x='model', y='tokens_per_second',
            color='model', points="all",
            title="Distribusi Kecepatan Token per Detik"
        )
        st.plotly_chart(fig_tps, use_container_width=True)
