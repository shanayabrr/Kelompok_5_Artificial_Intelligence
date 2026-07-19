from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentSplitter:
    """
    Kelas untuk membagi dokumen menjadi potongan-potongan kecil (chunks).
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Inisialisasi DocumentSplitter.
        
        Args:
            chunk_size (int): Ukuran maksimal setiap chunk (karakter).
            chunk_overlap (int): Jumlah karakter yang tumpang tindih antar chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Memotong sekumpulan dokumen menjadi chunk.
        
        Args:
            documents (list[Document]): Daftar dokumen yang akan dipotong.
            
        Returns:
            list[Document]: Daftar chunk hasil pemotongan.
        """
        if not documents:
            return []
        return self.text_splitter.split_documents(documents)
