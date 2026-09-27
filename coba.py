"""
Debug script - lihat chunk apa saja yang benar-benar di-retrieve untuk
suatu pertanyaan. Jalankan ini di folder project yang sama dengan
rag_chatbot.py:

    python debug_retrieval.py
"""

from rag_chatbot import load_documents, build_vectorstore, KNOWLEDGE_DIR, TOP_K

dokumen = load_documents(KNOWLEDGE_DIR)
vectorstore = build_vectorstore(dokumen)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

pertanyaan_list = [
    "What projects have you worked on?",
    "ceritain project yang pernah kamu kerjain",
]

for q in pertanyaan_list:
    print("=" * 70)
    print("QUERY:", q)
    hasil = retriever.invoke(q)
    for i, dok in enumerate(hasil):
        preview = dok.page_content.strip().split("\n")[0][:70]
        section = dok.metadata.get("section")
        print(f"  {i + 1}. [{section}] {preview}")
