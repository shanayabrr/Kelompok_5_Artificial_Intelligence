import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreManager:
    """
    Kelas untuk mengelola penyimpanan vektor menggunakan ChromaDB.
    """
    def __init__(self, embeddings: HuggingFaceEmbeddings, persist_directory: str = "vectorstore"):
        """
        Inisialisasi VectorStoreManager.
        
        Args:
            embeddings (HuggingFaceEmbeddings): Objek model embedding untuk mengubah teks menjadi vektor.
            persist_directory (str): Lokasi direktori untuk menyimpan database vektor secara persisten.
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        # Memuat vectorstore yang sudah ada jika tersedia
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def add_documents(self, documents: list[Document]):
        """
        Menyimpan chunk dokumen ke dalam vector database ChromaDB.
        
        Args:
            documents (list[Document]): Daftar chunk dokumen yang akan diindeks.
        """
        if not documents:
            raise ValueError("Tidak ada dokumen yang diberikan untuk diindeks.")
            
        if self.vectorstore is None:
            # Membuat vectorstore baru dan langsung menyimpan dokumen
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Menambahkan ke vectorstore yang sudah ada
            self.vectorstore.add_documents(documents)

    def clear_database(self):
        """
        Menghapus seluruh isi database vektor.
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None

    def get_retriever(self, top_k: int = 3):
        """
        Mendapatkan objek retriever untuk melakukan pencarian semantik (semantic similarity search).
        
        Args:
            top_k (int): Jumlah dokumen paling relevan yang akan diambil.
            
        Returns:
            Retriever: Objek LangChain retriever.
        """
        if self.vectorstore is None:
            raise Exception("Database vektor kosong. Harap unggah dan indeks dokumen terlebih dahulu.")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})
