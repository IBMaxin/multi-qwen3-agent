"""
CLI to ingest local documentation files into the local vector store using Qwen3 4b embedding (Ollama).

Usage:
    python ingest_local_docs.py --docs_folder ./docs --store_name my_docs_store

This script will:
- Read all .md and .txt files in the specified folder
- Chunk the text using LocalVectorSearch's chunker
- Store the chunks in a FAISS vector store using Ollama embeddings
"""

import argparse
import glob
import os
from production.qwen_pipeline.tools_custom import LocalVectorSearch


def load_and_chunk_docs(folder, max_chunk_tokens=500, overlap_tokens=50):
    docs = []
    for ext in ("*.md", "*.txt"):
        for file in glob.glob(os.path.join(folder, ext)):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                # Use the official chunker
                chunks = LocalVectorSearch.chunk_text(text, max_chunk_tokens, overlap_tokens)
                docs.extend(chunks)
    return docs


def main():
    parser = argparse.ArgumentParser(description="Ingest local docs into vector store.")
    parser.add_argument("--docs_folder", type=str, required=True, help="Folder with .md/.txt docs")
    parser.add_argument("--store_name", type=str, required=True, help="Name for the vector store")
    args = parser.parse_args()

    print(f"Loading and chunking docs from {args.docs_folder} ...")
    chunks = load_and_chunk_docs(args.docs_folder)
    print(f"Total chunks: {len(chunks)}")

    print("Storing chunks in vector store using Qwen3 4b embedding ...")
    vector_tool = LocalVectorSearch(cfg={"embedding_model": "qwen3-embedding:4b"})
    result_json = vector_tool.store_documents(chunks, args.store_name)
    print(f"Ingestion result: {result_json}")


if __name__ == "__main__":
    main()
