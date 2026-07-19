import requests
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import ChatOllama

class OllamaModelManager:
    """
    Kelas untuk menangani koneksi dan inisialisasi model LLM lokal via Ollama.
    """
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Inisialisasi koneksi ke server Ollama.
        
        Args:
            base_url (str): URL dari API lokal Ollama.
        """
        self.base_url = base_url

    def check_connection(self) -> bool:
        """
        Mengecek apakah server Ollama sedang berjalan.
        
        Returns:
            bool: True jika server berjalan, False jika tidak.
        """
        try:
            response = requests.get(self.base_url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def get_llm(self, model_name: str, temperature: float = 0.0) -> Ollama:
        """
        Mendapatkan instance LLM LangChain untuk model yang dipilih.
        
        Args:
            model_name (str): Nama model (contoh: 'phi3' atau 'gemma:2b').
            temperature (float): Parameter temperatur untuk mengontrol kreativitas respon.
            
        Returns:
            Ollama: Instance dari LangChain Ollama LLM.
        """
        if not self.check_connection():
            raise ConnectionError(
                f"Tidak dapat terhubung ke Ollama di {self.base_url}. "
                "Pastikan aplikasi Ollama sudah berjalan."
            )
            
        return Ollama(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature
        )

    def get_chat_model(self, model_name: str, temperature: float = 0.0) -> ChatOllama:
        """
        Mendapatkan instance Chat Model LangChain yang sering dibutuhkan untuk evaluasi ragas.
        
        Args:
            model_name (str): Nama model (contoh: 'phi3' atau 'gemma:2b').
            temperature (float): Parameter temperatur.
            
        Returns:
            ChatOllama: Instance dari LangChain ChatOllama.
        """
        if not self.check_connection():
            raise ConnectionError(
                f"Tidak dapat terhubung ke Ollama di {self.base_url}."
            )
            
        return ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature
        )
