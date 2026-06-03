"""
build_indexes.py
----------------
Builds two indexes over the chunked corpus:
  1. FAISS dense index    — for semantic similarity search (Hybrid RAG + Graph RAG)
  2. BM25 sparse index    — for keyword-based search (Hybrid RAG)

Why two indexes?
    Dense (FAISS): Embeds text into vectors using all-MiniLM-L6-v2.
    Good at finding semantically similar text even with different wording.
    E.g. "heart disease treatment" matches "cardiac therapy management".

    Sparse (BM25): Classic TF-IDF style keyword matching.
    Good at finding exact terms, medical codes, drug names, hospital names.
    E.g. "CLABSI infection rate" matches documents containing those exact words.

    Hybrid RAG combines both — reciprocal rank fusion merges the two result
    lists to get the best of semantic + keyword retrieval.

Output:
    data/cache/faiss_index/     — FAISS index + id mapping
    data/cache/bm25_index.pkl   — serialized BM25 index
    data/cache/chunks_map.json  — chunk_id → chunk metadata lookup

Usage:
    python indexing/build_indexes.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_PATH     = Path("data/cache/chunks.json")
FAISS_DIR       = Path("data/cache/faiss_index")
BM25_PATH       = Path("data/cache/bm25_index.pkl")
CHUNKS_MAP_PATH = Path("data/cache/chunks_map.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 64        # encode 64 chunks at a time to avoid OOM


# ── FAISS Index ───────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict]) -> None:
    """
    Embeds all chunks with all-MiniLM-L6-v2 and stores them in a FAISS
    IndexFlatIP (inner product = cosine similarity on normalized vectors).

    Why IndexFlatIP?
        Exact search — no approximation. For ~3000 chunks this is fast enough.
        For 1M+ chunks you'd switch to IndexIVFFlat for approximate search.
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]

    print(f"Encoding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize for cosine via inner product
        convert_to_numpy=True,
    )

    # Build FAISS index
    dim   = embeddings.shape[1]      # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)   # inner product on normalized = cosine
    index.add(embeddings.astype(np.float32))

    # Save index
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    # Save id mapping: position → chunk_id
    with open(FAISS_DIR / "id_map.json", "w") as f:
        json.dump(ids, f)

    print(f"  FAISS index: {index.ntotal} vectors (dim={dim}) → {FAISS_DIR}")


# ── BM25 Index ────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> None:
    """
    Builds a BM25Okapi index over tokenized chunk text.

    Why BM25Okapi?
        BM25 (Best Match 25) is the industry standard sparse retrieval algorithm.
        Okapi variant adds document length normalization.
        Used in production search systems (Elasticsearch default until v8).
    """
    print("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"  BM25 index: {len(tokenized)} documents → {BM25_PATH}")


# ── Chunks Map ────────────────────────────────────────────────────────────────

def build_chunks_map(chunks: list[dict]) -> None:
    """
    Saves chunk_id → full chunk metadata for fast lookup during retrieval.
    Both FAISS and BM25 return chunk_ids — this map resolves them to text + metadata.
    """
    chunks_map = {c["chunk_id"]: c for c in chunks}
    with open(CHUNKS_MAP_PATH, "w") as f:
        json.dump(chunks_map, f, indent=2)
    print(f"  Chunks map: {len(chunks_map)} entries → {CHUNKS_MAP_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading chunks...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded.")

    build_faiss_index(chunks)
    build_bm25_index(chunks)
    build_chunks_map(chunks)

    print(f"\nIndexing complete.")
    print(f"  FAISS: {FAISS_DIR}/index.faiss")
    print(f"  BM25:  {BM25_PATH}")
    print(f"  Map:   {CHUNKS_MAP_PATH}")


if __name__ == "__main__":
    main()
