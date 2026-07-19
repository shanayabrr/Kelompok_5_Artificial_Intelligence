from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGChain:
    """
    Kelas untuk merangkai komponen Retriever dan LLM menjadi sebuah Retrieval-Augmented Generation (RAG) chain.
    """
    def __init__(self, retriever, llm):
        """
        Inisialisasi RAGChain.
        
        Args:
            retriever: Objek retriever dari vector database (ChromaDB).
            llm: Objek model bahasa (Ollama).
        """
        self.retriever = retriever
        self.llm = llm
        
        # Template prompt augmentasi sesuai spesifikasi
        self.prompt_template = """Jawablah hanya berdasarkan konteks berikut.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:"""
        
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        """
        Menggabungkan teks dari beberapa dokumen hasil retrieval menjadi satu string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """
        Membangun LangChain runnable chain (LCEL).
        """
        chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        """
        Menjalankan RAG chain untuk mendapatkan jawaban dari LLM berdasarkan dokumen.
        
        Args:
            question (str): Pertanyaan dari pengguna.
            
        Returns:
            str: Jawaban dari LLM.
        """
        return self.chain.invoke(question)
