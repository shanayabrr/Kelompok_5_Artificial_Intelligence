"""
verify_index.py
================
Script kecil untuk memverifikasi jumlah chunk yang berhasil ter-index
di ChromaDB persisten. Jalankan setelah `python ingest.py` selesai
untuk memastikan data benar-benar tersimpan sebelum menjalankan server.

Jalankan:
    python verify_index.py
"""

from modules.retriever import get_collection_count

if __name__ == "__main__":
    count = get_collection_count()
    if count is None:
        print("❌ Tidak dapat terhubung ke ChromaDB.")
    elif count == 0:
        print("⚠️  ChromaDB terhubung tapi KOSONG (0 chunk). Jalankan ulang: python ingest.py")
    else:
        print(f"✅ ChromaDB terhubung dan berisi {count} chunk. Siap digunakan.")