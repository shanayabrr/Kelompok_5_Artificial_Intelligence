from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    """
    Kelas untuk menangani pembuatan embedding dari teks menggunakan model HuggingFace.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inisialisasi EmbeddingManager.
        
        Args:
            model_name (str): Nama model HuggingFace yang akan digunakan.
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Mengembalikan objek HuggingFaceEmbeddings yang siap digunakan.
        
        Returns:
            HuggingFaceEmbeddings: Objek embedding dari LangChain.
        """
        return self.embeddings
