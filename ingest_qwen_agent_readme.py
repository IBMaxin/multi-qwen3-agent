"""
Script to ingest a single markdown file (Qwen-Agent-README.md) into your local vector store, with step-by-step explanations.
"""

import os
from production.qwen_pipeline.tools_custom import LocalVectorSearch


def explain(msg):
    print(f"\n[STEP] {msg}\n")


def main():
    # 1. Select the file to ingest
    file_path = r"c:\Users\bobby\Downloads\Qwen-Agent-README.md"
    store_name = "qwen_agent_readme"
    explain(f"Reading file: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    # 2. Chunk the document
    explain("Chunking the document into smaller pieces for efficient retrieval...")
    chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens=500, overlap_tokens=50)
    print(f"Total chunks created: {len(chunks)}")
    print(f"First chunk preview:\n{chunks[0][:300]}\n...")

    # 3. Store the chunks in the vector store
    explain("Storing chunks in the FAISS vector store using Qwen3-embedding:4b via Ollama...")
    vector_tool = LocalVectorSearch(cfg={"embedding_model": "qwen3-embedding:4b"})
    result_json = vector_tool.store_documents(chunks, store_name)
    print(f"Ingestion result: {result_json}")

    explain("Done! You can now query this store using your RAG pipeline.")


if __name__ == "__main__":
    main()
