"""
Embed preprocessed chunks and store in ChromaDB.

Usage:
    python src/embeddings.py --input data/processed/cleaned_chunks.json --db-path data/chroma_db
"""

try:
    from src.env_bootstrap import ensure_valid_thread_env
except ImportError:
    from env_bootstrap import ensure_valid_thread_env

ensure_valid_thread_env()

import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer


def build_index(input_path: str, db_path: str,
                collection_name: str = "bank_knowledge",
                model_name: str = "all-MiniLM-L6-v2"):
    """Load chunks, embed them, and store in a ChromaDB collection."""
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {input_path}")

    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=db_path)

    # Recreate collection from scratch
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Flatten metadata values to ChromaDB-compatible types (str/int/float/bool)
    metadatas = []
    for c in chunks:
        meta = {}
        for k, v in c.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        metadatas.append(meta)

    # Embed and upsert in batches
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        batch_texts = texts[i:end]
        batch_embeddings = model.encode(batch_texts).tolist()

        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"  Indexed batch {i}-{end}")

    print(f"Done – {len(texts)} chunks indexed into '{collection_name}' at {db_path}")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index.")
    parser.add_argument("--input", default="data/processed/cleaned_chunks.json")
    parser.add_argument("--db-path", default="data/chroma_db")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    build_index(args.input, args.db_path, model_name=args.model)
