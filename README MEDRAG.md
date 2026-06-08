# 🏥 MedRAG Arena

**A benchmark comparing three RAG architectures on real healthcare data — PubMed abstracts + CMS hospital quality data + Snowflake**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/Arun-V2002/HealthRAG-Benchmark)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20llama--3.3--70b-orange)](https://groq.com)
[![Snowflake](https://img.shields.io/badge/Data-Snowflake-29B5E8)](https://snowflake.com)

---

## What is this?

MedRAG Arena benchmarks three production-grade RAG architectures on 50 healthcare questions across three difficulty tiers, using real clinical and hospital data. The goal is to answer a practical question: **when does each RAG architecture succeed, and where does it fail?**

The benchmark reveals a clear pattern — architecture choice matters more than tuning. Hybrid RAG wins on simple factual recall, Graph RAG trades quality for speed, and Agentic RAG — the only architecture with SQL access — is the only one that can answer multi-source questions requiring both clinical text and structured hospital statistics.

---

## Benchmark Results

### Overall performance (48 questions per architecture)

| Architecture | RAGAS Score | Avg Latency | Avg Cost/Query | Answer Relevancy |
|---|---|---|---|---|
| **Agentic RAG** | **0.377** | 28.4s | $0.00164 | **0.781** |
| Hybrid RAG | 0.294 | 28.2s | $0.00130 | 0.417 |
| Graph RAG | 0.200 | **6.6s** | **$0.00096** | 0.488 |

### RAGAS score by question tier

| Tier | Description | Hybrid | Graph | Agentic |
|---|---|---|---|---|
| Tier 1 | Factual — "What is sepsis?" | 0.426 | 0.235 | **0.486** |
| Tier 2 | Analytical — "Which states have highest MRSA rates?" | 0.275 | 0.197 | **0.369** |
| Tier 3 | Multi-hop — "How do outcomes in sepsis ICU patients relate to hospital quality?" | 0.146 | 0.159 | **0.246** |

### Key findings

- **Agentic RAG scores 28% higher** than Hybrid RAG overall (0.377 vs 0.294)
- **Graph RAG is 4.3× faster** than the other two (6.6s vs ~28s) — ideal for latency-sensitive applications
- **All architectures degrade on Tier 3** — multi-hop questions requiring SQL + clinical text remain the hardest challenge
- **Agentic RAG's answer relevancy (0.781) is nearly 2× Graph RAG's** — the ReAct + CRAG loop substantially improves answer quality
- **Faithfulness is low across all architectures** — a known weakness of LLM-as-judge RAGAS scoring on Groq free tier

---

## Architecture

### System overview

```
Data Sources          Indexing Layer           RAG Architectures
─────────────         ──────────────           ─────────────────
PubMed API  ──►  Chunker (2,502 chunks) ──►   Hybrid RAG
CMS API     ──►  FAISS (384-dim)        ──►   Graph RAG
                 BM25                   ──►   Agentic RAG
Snowflake ─────────────────────────────────►  (SQL tool)
                         │
                         ▼
                   Groq LLM (llama-3.3-70b)
                         │
                         ▼
              RAGAS Evaluation + Streamlit App
```

### Query routing by architecture

**Hybrid RAG** — BM25 keyword search + FAISS semantic search → RRF fusion → BGE cross-encoder reranker → top-k chunks → LLM

**Graph RAG** — Entity extraction (llama-3.1-8b) → NetworkX knowledge graph traversal (735 nodes, 660 edges) → FAISS fallback if no graph hits → LLM

**Agentic RAG** — ReAct agent decides: clinical question → vector search tool → FAISS; structured question → SQL tool → Snowflake views; multi-hop → both tools → CRAG self-reflection → LLM

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` (generation), `llama-3.1-8b-instant` (entity extraction) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Vector store | FAISS (2,502 vectors) |
| Keyword index | BM25 (rank-bm25) |
| Knowledge graph | NetworkX (735 nodes, 660 edges) |
| Data warehouse | Snowflake (HEALTHRAG_DB — 2 tables, 3 analytical views) |
| Evaluation | LLM-as-judge RAGAS (faithfulness, answer relevancy, context precision, context recall) |
| App | Streamlit (3 tabs: Chat, Benchmark Results, Data Explorer) |
| Deployment | Hugging Face Spaces (Docker) |
| Dashboard | Power BI (DirectQuery → Snowflake) |

---

## Data

### PubMed (clinical text)
- 482 abstracts across 5 conditions: sepsis, diabetes, cardiovascular disease, Alzheimer's, hospital-acquired infections
- Fetched via NCBI E-utilities API
- Chunked into 502 text segments with metadata (PMID, condition, source)

### CMS Hospital Quality (structured)
- 2,000 hospitals from the CMS Provider Data API
- HAI infection rates: CLABSI, CAUTI, MRSA, CDI, SSI (Standardised Infection Ratios)
- Loaded into Snowflake with 3 pre-aggregated analytical views for SQL querying

**Why two storage systems?** PubMed text is unstructured — it needs semantic search (FAISS) and keyword search (BM25). CMS data is tabular and aggregatable — it needs SQL (Snowflake). Agentic RAG routes queries to the right backend based on question type.

---

## Evaluation Methodology

50 questions across 3 difficulty tiers, evaluated with 4 RAGAS metrics using Groq `llama-3.3-70b` as judge:

| Tier | Count | Question type | Example |
|---|---|---|---|
| 1 | 18 | Factual definition | "What is CLABSI?" |
| 2 | 16 | Analytical / comparative | "Which hospital types have the highest quality ratings?" |
| 3 | 14 | Multi-hop cross-source | "For states with high Alzheimer's prevalence, what are the hospital quality metrics?" |

**RAGAS metrics:**
- **Faithfulness** — are all claims in the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question asked?
- **Context Precision** — are the retrieved chunks actually useful?
- **Context Recall** — did retrieval find all necessary information?

---

## Project Structure

```
medrag-arena/
├── app.py                          # Streamlit app (Chat / Benchmark / Data Explorer)
├── startup.py                      # Builds indexes on first startup (HF Spaces)
├── Dockerfile                      # HF Spaces deployment
├── requirements.txt
├── data/
│   ├── fetch_pubmed.py             # NCBI E-utilities → corpus.json
│   ├── fetch_cms.py                # CMS DKAN API → hospitals.csv + complications.csv
│   └── load_snowflake.py           # Loads CSVs → Snowflake + creates 3 views
├── indexing/
│   ├── chunker.py                  # 2,502 chunks with metadata
│   └── build_indexes.py            # FAISS + BM25 + chunks_map.json
├── rag/
│   ├── hybrid_rag.py               # BM25 + FAISS + RRF + BGE reranker
│   ├── graph_rag.py                # NetworkX entity graph + FAISS fallback
│   └── agentic_rag.py              # ReAct agent + CRAG + Snowflake SQL tool
└── evaluation/
    ├── test_questions.json         # 50 questions across 3 tiers
    ├── run_eval.py                 # RAGAS evaluation runner
    └── results/
        ├── benchmark.csv          # Full results (150 rows)
        └── summary.json           # Aggregated scores per architecture
```

---

## Local Setup

```bash
git clone https://github.com/v-arun2002/MedRAG-Arena.git
cd MedRAG-Arena
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
NCBI_API_KEY=your_ncbi_key
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=HEALTHRAG_DB
SNOWFLAKE_SCHEMA=PUBLIC
```

Fetch data and build indexes:
```bash
python data/fetch_pubmed.py
python data/fetch_cms.py
python data/load_snowflake.py
python indexing/chunker.py
python indexing/build_indexes.py
```

Run the app:
```bash
streamlit run app.py
```

Run the benchmark:
```bash
python evaluation/run_eval.py --arch hybrid --tier 1
python evaluation/run_eval.py --arch graph
python evaluation/run_eval.py --arch agentic
```

---

## Live Demo

**[huggingface.co/spaces/Arun-V2002/HealthRAG-Benchmark](https://huggingface.co/spaces/Arun-V2002/HealthRAG-Benchmark)**

The app has three tabs:
- **Chat** — ask any healthcare question, see all 3 architectures answer simultaneously with retrieved chunks and latency
- **Benchmark Results** — RAGAS scores, latency comparisons, tier-by-tier breakdown
- **Data Explorer** — CMS hospital quality map, HAI infection rates by state

---

## Limitations

- **RAGAS scoring uses LLM-as-judge** — scores are relative, not absolute. A different judge model would produce different numbers.
- **Groq free tier** limits evaluation throughput to ~25 questions/day, making large-scale benchmarking slow.
- **Graph RAG faithfulness = 0.0** — the NetworkX entity graph loses source attribution during traversal, so RAGAS cannot verify claims against retrieved chunks.
- **Tier 3 scores are low across all architectures** — genuine multi-hop reasoning over two data sources remains an open problem.

---

## Author

**Arun Valliappan** — MS Data Analytics Engineering, Northeastern University  
[LinkedIn](https://www.linkedin.com/in/arun-valliappan-990334263/) · [GitHub](https://github.com/v-arun2002)
