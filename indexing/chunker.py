"""
chunker.py
----------
Splits PubMed abstracts and CMS hospital documents into chunks for indexing.

Why chunking matters:
    Embedding models have a token limit (~512 tokens for all-MiniLM-L6-v2).
    Long documents must be split into smaller overlapping chunks so no
    information is lost at boundaries. Overlap ensures a sentence split
    across two chunks is still retrievable.

Strategy:
    - PubMed abstracts  → sentence-aware chunking (avg ~200 words, fits in 1-2 chunks)
    - CMS documents     → single chunk per hospital (short templated text, ~100 words)

Output:
    data/cache/chunks.json   — all chunks with metadata, ready for indexing

Usage:
    python indexing/chunker.py
"""

import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PUBMED_CORPUS  = Path("data/raw/pubmed/corpus.json")
CMS_CORPUS     = Path("data/raw/cms/cms_documents.json")
OUTPUT_PATH    = Path("data/cache/chunks.json")

CHUNK_SIZE     = 400    # max words per chunk
CHUNK_OVERLAP  = 50     # words of overlap between consecutive chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter — splits on . ! ? followed by whitespace."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into overlapping word-based chunks.
    Tries to split on sentence boundaries to preserve meaning.
    """
    sentences = split_into_sentences(text)
    chunks    = []
    current   = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        # If adding this sentence exceeds chunk_size, save current chunk
        if word_count + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep last `overlap` words for next chunk
            overlap_words = " ".join(current).split()[-overlap:]
            current    = overlap_words + words
            word_count = len(current)
        else:
            current.extend(words)
            word_count += len(words)

    # Don't forget the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def build_pubmed_chunks(corpus: list[dict]) -> list[dict]:
    """
    Chunks PubMed abstracts.
    Each chunk carries metadata: pmid, title, condition, source, year.
    """
    chunks = []
    for doc in corpus:
        text = doc.get("abstract", "").strip()
        if not text:
            continue

        doc_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "chunk_id":  f"pubmed_{doc['pmid']}_{i}",
                "text":      chunk,
                "source":    "pubmed",
                "condition": doc.get("condition", ""),
                "pmid":      doc.get("pmid", ""),
                "title":     doc.get("title", ""),
                "year":      doc.get("year", ""),
                "journal":   doc.get("journal", ""),
                "chunk_idx": i,
                "total_chunks": len(doc_chunks),
            })

    return chunks


def build_cms_chunks(corpus: list[dict]) -> list[dict]:
    """
    Chunks CMS hospital documents.
    CMS docs are short templated text (~100 words) — one chunk per hospital.
    """
    chunks = []
    for i, doc in enumerate(corpus):
        text = doc.get("text", "").strip()
        if not text:
            continue

        # CMS docs are short enough to keep as single chunks
        # but we still run through chunker in case any are long
        doc_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for j, chunk in enumerate(doc_chunks):
            chunks.append({
                "chunk_id": f"cms_{doc.get('facility_id', i)}_{j}",
                "text":        chunk,
                "source":      "cms",
                "facility_id": doc.get("facility_id", ""),
                "facility_name": doc.get("facility_name", ""),
                "state":       doc.get("state", ""),
                "city":        doc.get("city", ""),
                "overall_rating": doc.get("overall_rating", ""),
                "chunk_idx":   j,
                "total_chunks": len(doc_chunks),
            })

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load corpora
    print("Loading PubMed corpus...")
    with open(PUBMED_CORPUS) as f:
        pubmed_corpus = json.load(f)
    print(f"  {len(pubmed_corpus)} PubMed abstracts loaded.")

    print("Loading CMS corpus...")
    with open(CMS_CORPUS) as f:
        cms_corpus = json.load(f)
    print(f"  {len(cms_corpus)} CMS documents loaded.")

    # Chunk
    print("Chunking PubMed abstracts...")
    pubmed_chunks = build_pubmed_chunks(pubmed_corpus)
    print(f"  {len(pubmed_chunks)} PubMed chunks created.")

    print("Chunking CMS documents...")
    cms_chunks = build_cms_chunks(cms_corpus)
    print(f"  {len(cms_chunks)} CMS chunks created.")

    # Merge and save
    all_chunks = pubmed_chunks + cms_chunks
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)} → {OUTPUT_PATH}")
    print(f"  PubMed: {len(pubmed_chunks)} chunks from {len(pubmed_corpus)} abstracts")
    print(f"  CMS:    {len(cms_chunks)} chunks from {len(cms_corpus)} documents")


if __name__ == "__main__":
    main()
