"""
Package `modules` berisi seluruh komponen inti pipeline RAG
untuk Smart Health Assistant:

- config       : konfigurasi & konstanta global
- loader       : pemuatan dokumen PDF/TXT
- splitter     : chunking teks
- embeddings   : model embedding HuggingFace
- retriever    : vector store ChromaDB
- rag_chain    : pipeline RAG (retrieval + generation via Ollama/Phi-3)
"""
