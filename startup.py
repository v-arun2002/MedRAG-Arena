"""
startup.py
----------
Builds indexes on HF Spaces if cache files don't exist.
Runs automatically before app.py starts.
"""
import os
from pathlib import Path

cache_dir = Path("data/cache")
faiss_path = cache_dir / "faiss_index/index.faiss"

if not faiss_path.exists():
    print("Cache not found — building indexes...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    from indexing.chunker import main as chunk_main
    from indexing.build_indexes import main as index_main
    chunk_main()
    index_main()
    print("Indexes built successfully.")
else:
    print("Cache found — skipping build.")