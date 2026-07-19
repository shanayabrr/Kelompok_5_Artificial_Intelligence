import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    Kelas untuk menangani pemuatan dokumen PDF dan TXT.
    Mendukung pemuatan dari direktori maupun dari file tunggal.
    """
    def __init__(self, data_dir: str = "data/documents/"):
        """
        Inisialisasi DocumentLoader.
        
        Args:
            data_dir (str): Direktori default tempat menyimpan dokumen.
        """
        self.data_dir = data_dir
        # Memastikan direktori ada
        os.makedirs(self.data_dir, exist_ok=True)

    def load_document(self, file_path: str) -> list[Document]:
        """
        Memuat dokumen berdasarkan tipe file (PDF atau TXT).
        
        Args:
            file_path (str): Path absolut atau relatif dari file.
            
        Returns:
            list[Document]: Daftar halaman/bagian dokumen yang dimuat.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        else:
            raise ValueError(f"Ekstensi file {ext} tidak didukung. Harap gunakan PDF atau TXT.")

    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Menyimpan file yang diunggah dari Streamlit ke direktori lokal.
        
        Args:
            uploaded_file: Objek file dari Streamlit (UploadedFile).
            
        Returns:
            str: Path tempat file disimpan.
        """
        file_path = os.path.join(self.data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
