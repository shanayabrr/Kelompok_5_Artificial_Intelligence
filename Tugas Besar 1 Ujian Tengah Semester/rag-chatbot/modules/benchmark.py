import os
import time
import psutil
import pandas as pd
from datetime import datetime

# Impor Ragas untuk evaluasi akurasi
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness
)

class ModelBenchmark:
    """
    Kelas untuk melakukan benchmark dan logging efisiensi serta akurasi model SLMs.
    """
    def __init__(self, log_file: str = "results.csv"):
        """
        Inisialisasi benchmark logger.
        
        Args:
            log_file (str): Path file CSV untuk menyimpan hasil logging.
        """
        self.log_file = log_file
        self._initialize_csv()

    def _initialize_csv(self):
        """
        Membuat file CSV dengan header yang sesuai jika belum ada.
        """
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                "timestamp",
                "model",
                "question",
                "response_time",
                "first_token_time",
                "tokens_per_second",
                "ram_usage",
                "cpu_usage",
                "context_precision",
                "answer_relevancy",
                "faithfulness_score"
            ])
            df.to_csv(self.log_file, index=False)

    def _get_system_usage(self):
        """
        Mengambil penggunaan RAM (dalam MB) dan CPU (persentase).
        """
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024) # Konversi ke MB
        cpu_usage = psutil.cpu_percent(interval=0.1)
        return ram_usage, cpu_usage

    def evaluate_accuracy_ragas(self, question: str, answer: str, contexts: list[str], evaluator_llm, evaluator_embeddings):
        """
        Menghitung metrik akurasi menggunakan framework Ragas.
        Memerlukan LLM dan Embeddings untuk bertindak sebagai juri (evaluator).
        """
        # Ragas mengharapkan input dalam bentuk Dataset HuggingFace
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truths": [[""]]
        }
        dataset = Dataset.from_dict(data)
        
        # Mengeksekusi evaluasi
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision, # Mirip context relevance
                    answer_relevancy,
                    faithfulness
                ],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )
            return {
                "context_precision": result.get("context_precision", 0.0),
                "answer_relevancy": result.get("answer_relevancy", 0.0),
                "faithfulness_score": result.get("faithfulness", 0.0)
            }
        except Exception as e:
            print(f"Error pada evaluasi Ragas: {e}")
            return {
                "context_precision": 0.0,
                "answer_relevancy": 0.0,
                "faithfulness_score": 0.0
            }

    def measure_and_log(self, model_name: str, question: str, rag_chain, retriever, evaluator_llm, evaluator_embeddings):
        """
        Mengukur efisiensi (waktu, memori) dan akurasi (Ragas), lalu menyimpannya ke results.csv.
        
        Args:
            model_name (str): Nama model yang sedang dites.
            question (str): Pertanyaan user.
            rag_chain (RAGChain): Objek RAGChain untuk menghasilkan jawaban.
            retriever (Retriever): Objek retriever untuk mendapatkan context.
            evaluator_llm: Chat model (LangChain ChatOllama) untuk evaluator Ragas.
            evaluator_embeddings: Embedding model untuk evaluator Ragas.
            
        Returns:
            str: Jawaban dari LLM.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Mulai pengukuran waktu & memori
        start_time = time.time()
        ram_before, _ = self._get_system_usage()
        
        # 1. Retrieval
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # 2. Generation (Asumsi mendapatkan jawaban penuh sekaligus untuk simplicity)
        # Jika butuh first_token_time yang akurat, harus menggunakan streaming. 
        # Di sini kita simulasikan first token sebagai fraksi dari total waktu untuk menghindari kompleksitas streaming Langchain standar,
        # Namun untuk aplikasi real, kita bisa ukur via streaming callback.
        answer = rag_chain.invoke(question)
        
        end_time = time.time()
        ram_after, cpu_usage = self._get_system_usage()
        
        # Kalkulasi metrik efisiensi
        response_time = end_time - start_time
        ram_usage = abs(ram_after - ram_before) # Selisih RAM atau penggunaan aktual
        if ram_usage == 0: 
            ram_usage = ram_after # Fallback ke total penggunaan proses jika selisih tidak terlihat
            
        # Estimasi First Token & Token/sec (Aproksimasi karena tidak streaming)
        # 1 token ~ 4 karakter
        token_count = len(answer) / 4
        tokens_per_second = token_count / response_time if response_time > 0 else 0
        first_token_time = response_time * 0.2 if response_time > 0 else 0 # Estimasi kasar
        
        # 3. Evaluasi Akurasi (Ragas)
        # Perhatian: Proses ini bisa sangat memakan waktu karena menggunakan LLM lokal
        metrics = self.evaluate_accuracy_ragas(
            question=question,
            answer=answer,
            contexts=contexts,
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings
        )
        
        # 4. Simpan ke CSV
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "model": model_name,
            "question": question,
            "response_time": round(response_time, 2),
            "first_token_time": round(first_token_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "ram_usage": round(ram_usage, 2),
            "cpu_usage": round(cpu_usage, 2),
            "context_precision": round(metrics["context_precision"], 4),
            "answer_relevancy": round(metrics["answer_relevancy"], 4),
            "faithfulness_score": round(metrics["faithfulness_score"], 4)
        }])
        
        new_row.to_csv(self.log_file, mode='a', header=False, index=False)
        
        return answer
