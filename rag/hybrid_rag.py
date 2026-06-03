"""
hybrid_rag.py
-------------
Hybrid RAG: BM25 (sparse) + FAISS (dense) + BGE Cross-Encoder Reranker

How it works:
    1. Query comes in
    2. BM25 retrieves top-20 chunks by keyword matching
    3. FAISS retrieves top-20 chunks by semantic similarity
    4. Reciprocal Rank Fusion (RRF) merges both lists into one ranked list
    5. BGE cross-encoder reranks the top-20 merged results to top-5
    6. Top-5 chunks are passed to Groq (llama-3.3-70b) as context
    7. Groq generates a grounded answer

Why this order?
    BM25 + FAISS (retrieve 20 each) → RRF (merge to 20) → Reranker (compress to 5) → LLM
    
    Reranking is the key quality step. The bi-encoder (all-MiniLM) is fast but imprecise.
    The cross-encoder (BGE) is slower but much more accurate — it reads query + chunk
    together, like a human would. We can't run cross-encoder on all 2502 chunks (too slow),
    so we use it only on the top-20 candidates from the first retrieval stage.

Usage:
    from rag.hybrid_rag import HybridRAG
    rag = HybridRAG()
    result = rag.query("Which hospitals have the highest MRSA infection rates?")
"""

import json
import pickle
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = Path("data/cache/faiss_index/index.faiss")
FAISS_ID_MAP     = Path("data/cache/faiss_index/id_map.json")
BM25_PATH        = Path("data/cache/bm25_index.pkl")
CHUNKS_MAP_PATH  = Path("data/cache/chunks_map.json")

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "BAAI/bge-reranker-base"
LLM_MODEL        = "llama-3.3-70b-versatile"

TOP_K_RETRIEVE   = 20      # retrieve top-20 from each index
TOP_K_RERANK     = 5       # rerank down to top-5 for LLM context
RRF_K            = 60      # RRF constant (standard value from literature)


# ── HybridRAG Class ───────────────────────────────────────────────────────────

class HybridRAG:
    def __init__(self):
        print("[HybridRAG] Loading indexes and models...")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_ID_MAP) as f:
            self.id_map = json.load(f)             # position → chunk_id
        
        # Load BM25 index
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load chunks map for text lookup
        with open(CHUNKS_MAP_PATH) as f:
            self.chunks_map = json.load(f)
        self.chunk_ids = list(self.chunks_map.keys())  # ordered list for BM25

        # Load embedding model (for query encoding)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Load BGE cross-encoder reranker
        # Cross-encoder reads (query, chunk) pair together — much more accurate
        # than bi-encoder but too slow to run on all chunks
        print("[HybridRAG] Loading BGE reranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Groq client
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        print("[HybridRAG] Ready.")


    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _faiss_retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Dense retrieval: embeds query → finds top-k nearest vectors in FAISS.
        Returns list of (chunk_id, score) tuples.
        """
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.faiss_index.search(query_vec.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:                          # -1 = no result
                results.append((self.id_map[idx], float(score)))
        return results


    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Sparse retrieval: tokenizes query → BM25 scores all chunks.
        Returns list of (chunk_id, score) tuples.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]


    def _reciprocal_rank_fusion(
        self,
        faiss_results: list[tuple[str, float]],
        bm25_results:  list[tuple[str, float]],
        k: int = RRF_K,
    ) -> list[str]:
        """
        Reciprocal Rank Fusion merges two ranked lists into one.

        Formula: RRF(d) = Σ 1 / (k + rank(d))
        
        Why RRF?
            Simple, parameter-free, and empirically outperforms weighted score
            combination. A chunk ranked #1 in BM25 and #5 in FAISS scores higher
            than one ranked #3 in both — exactly what we want from hybrid search.
            k=60 is the standard value from the original RRF paper (Cormack 2009).
        """
        scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(faiss_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # Sort by RRF score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked]


    def _rerank(self, query: str, chunk_ids: list[str], top_k: int) -> list[dict]:
        """
        Cross-encoder reranking: scores each (query, chunk_text) pair directly.
        
        Unlike bi-encoders that embed query and chunk separately,
        cross-encoders read both together — like a human reading a query
        and a document side by side. Much more accurate but O(n) inference.
        That's why we only rerank the top-20 from RRF, not all 2502 chunks.
        """
        chunks = [self.chunks_map[cid] for cid in chunk_ids if cid in self.chunks_map]
        if not chunks:
            return []

        pairs  = [[query, c["text"]] for c in chunks]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]


    # ── Generation ────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        """Builds the prompt sent to Groq with retrieved context."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            if source == "pubmed":
                ref = f"[PubMed PMID:{chunk.get('pmid','')} | {chunk.get('condition','')} | {chunk.get('year','')}]"
            else:
                ref = f"[CMS Hospital: {chunk.get('facility_name','')} | {chunk.get('state','')} | Rating: {chunk.get('overall_rating','')}]"
            context_parts.append(f"Source {i+1} {ref}:\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        return f"""You are a healthcare data analyst assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite which source(s) you used in your answer.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


    def query(self, question: str) -> dict:
        """
        Full hybrid RAG pipeline:
        BM25 + FAISS → RRF → Reranker → Groq
        
        Returns dict with answer, retrieved chunks, and timing.
        """
        start = time.time()

        # Stage 1: Dual retrieval
        faiss_results = self._faiss_retrieve(question, TOP_K_RETRIEVE)
        bm25_results  = self._bm25_retrieve(question, TOP_K_RETRIEVE)

        # Stage 2: RRF fusion
        fused_ids = self._reciprocal_rank_fusion(faiss_results, bm25_results)[:TOP_K_RETRIEVE]

        # Stage 3: Cross-encoder reranking
        reranked_chunks = self._rerank(question, fused_ids, TOP_K_RERANK)

        # Stage 4: LLM generation
        prompt   = self._build_prompt(question, reranked_chunks)
        response = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        latency = time.time() - start

        return {
            "answer":          answer,
            "retrieved_chunks": reranked_chunks,
            "num_chunks":      len(reranked_chunks),
            "latency_sec":     round(latency, 2),
            "architecture":    "hybrid",
            "tokens_used":     response.usage.total_tokens,
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = HybridRAG()
    
    test_questions = [
        "What are the treatment options for type 2 diabetes?",
        "Which hospitals have high MRSA infection rates?",
        "What is the relationship between sepsis and ICU mortality?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer'][:300]}...")
        print(f"   Latency: {result['latency_sec']}s | Tokens: {result['tokens_used']} | Chunks: {result['num_chunks']}")
