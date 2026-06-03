"""
run_eval.py
-----------
Runs all 50 test questions through all 3 RAG architectures and evaluates
with RAGAS metrics + latency + cost.

What is RAGAS?
    RAGAS (Retrieval Augmented Generation Assessment) is an evaluation
    framework specifically designed for RAG systems. It measures:

    - Faithfulness (0-1):
        Does the answer contain only information from the retrieved context?
        High faithfulness = no hallucinations.
        Low faithfulness = LLM is making things up beyond the context.

    - Answer Relevancy (0-1):
        Is the answer actually relevant to the question asked?
        Measures if the answer addresses the question, not just if it's factual.

    - Context Precision (0-1):
        Are the retrieved chunks actually useful for answering the question?
        High precision = retrieved the right chunks.
        Low precision = retrieved irrelevant chunks even if answer is ok.

    - Context Recall (0-1):
        Did we retrieve ALL the information needed to answer the question?
        High recall = nothing important was missed.
        Low recall = answer is incomplete because retrieval missed key chunks.

Why all 4 metrics?
    A system can score high on one and low on another:
    - High faithfulness + low relevancy = accurate but doesn't answer the question
    - High recall + low precision = retrieved too much irrelevant context
    - Low faithfulness = hallucinating regardless of retrieval quality

Output:
    evaluation/results/benchmark.csv   — full results for Power BI
    evaluation/results/summary.json    — aggregated scores per architecture

Usage:
    python evaluation/run_eval.py

    # Run only one architecture:
    python evaluation/run_eval.py --arch hybrid

    # Run only one tier:
    python evaluation/run_eval.py --tier 1
"""

import json
import csv
import time
import time as _time
import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TEST_QUESTIONS_PATH = Path("evaluation/test_questions.json")
RESULTS_DIR         = Path("evaluation/results")
BENCHMARK_CSV       = RESULTS_DIR / "benchmark.csv"
SUMMARY_JSON        = RESULTS_DIR / "summary.json"

LLM_MODEL           = "llama-3.3-70b-versatile"

# Groq pricing ($ per 1M tokens)
GROQ_INPUT_COST     = 0.59
GROQ_OUTPUT_COST    = 0.79


# ── Cost Calculator ───────────────────────────────────────────────────────────

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost  = (prompt_tokens    / 1_000_000) * GROQ_INPUT_COST
    output_cost = (completion_tokens / 1_000_000) * GROQ_OUTPUT_COST
    return round(input_cost + output_cost, 6)


# ── RAGAS-style Evaluation (LLM-based) ───────────────────────────────────────

class RAGASEvaluator:
    """
    LLM-based RAGAS evaluation.
    Uses Groq as the judge model — same approach used in production by many teams.
    Includes automatic retry with exponential backoff for rate limit handling.
    """
    def __init__(self, groq_client: Groq):
        self.groq = groq_client

    def _score(self, prompt: str) -> float:
        """Gets a 0.0-1.0 score from the LLM judge with retry on rate limit."""
        for attempt in range(3):
            try:
                response = self.groq.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                raw   = response.choices[0].message.content.strip()
                match = re.search(r"[0-9]+\.?[0-9]*", raw)
                score = float(match.group()) if match else 0.5
                return max(0.0, min(1.0, score))
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 60 * (attempt + 1)
                    print(f"   Rate limit hit — waiting {wait}s before retry...")
                    _time.sleep(wait)
                else:
                    return 0.5
        return 0.5

    def faithfulness(self, answer: str, context_chunks: list) -> float:
        """
        Faithfulness: is every claim in the answer supported by the context?
        Score 1.0 = fully grounded, 0.0 = complete hallucination.
        """
        context = "\n".join([c.get("text", "") for c in context_chunks[:3]])
        prompt = f"""Rate how faithful this answer is to the provided context.
Score 1.0 if every claim in the answer is directly supported by the context.
Score 0.0 if the answer contains information not present in the context.
Return ONLY a decimal number between 0.0 and 1.0.

Context: {context[:1000]}

Answer: {answer[:500]}

Faithfulness score (0.0-1.0):"""
        return self._score(prompt)

    def answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy: does the answer actually address the question?
        Score 1.0 = directly answers the question, 0.0 = completely off-topic.
        """
        prompt = f"""Rate how relevant this answer is to the question.
Score 1.0 if the answer directly and completely addresses the question.
Score 0.0 if the answer is off-topic or doesn't address the question.
Return ONLY a decimal number between 0.0 and 1.0.

Question: {question}

Answer: {answer[:500]}

Answer relevancy score (0.0-1.0):"""
        return self._score(prompt)

    def context_precision(self, question: str, context_chunks: list) -> float:
        """
        Context Precision: are the retrieved chunks relevant to the question?
        Score 1.0 = all retrieved chunks are useful, 0.0 = all irrelevant.
        """
        if not context_chunks:
            return 0.0
        context = "\n".join([c.get("text", "")[:200] for c in context_chunks[:5]])
        prompt = f"""Rate how relevant these retrieved text chunks are to answering the question.
Score 1.0 if all chunks are directly relevant to the question.
Score 0.0 if all chunks are irrelevant to the question.
Return ONLY a decimal number between 0.0 and 1.0.

Question: {question}

Retrieved chunks: {context[:1000]}

Context precision score (0.0-1.0):"""
        return self._score(prompt)

    def context_recall(self, question: str, answer: str, context_chunks: list) -> float:
        """
        Context Recall: did we retrieve all information needed to answer?
        Score 1.0 = context contains everything needed, 0.0 = key info missing.
        """
        if not context_chunks:
            return 0.0
        context = "\n".join([c.get("text", "")[:200] for c in context_chunks[:5]])
        prompt = f"""Rate whether the retrieved context contains all the information needed
to fully answer the question.
Score 1.0 if the context contains everything needed for a complete answer.
Score 0.0 if the context is missing key information needed to answer.
Return ONLY a decimal number between 0.0 and 1.0.

Question: {question}
Answer given: {answer[:300]}
Retrieved context: {context[:1000]}

Context recall score (0.0-1.0):"""
        return self._score(prompt)

    def evaluate(self, question: str, answer: str, context_chunks: list) -> dict:
        """Runs all 4 RAGAS metrics and returns scores."""
        return {
            "faithfulness":      self.faithfulness(answer, context_chunks),
            "answer_relevancy":  self.answer_relevancy(question, answer),
            "context_precision": self.context_precision(question, context_chunks),
            "context_recall":    self.context_recall(question, answer, context_chunks),
        }


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark(arch_filter: str = "all", tier_filter: int = None):
    """
    Runs all test questions through specified architectures and logs results.
    Skips already completed question/architecture combinations to allow safe re-runs.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test questions
    with open(TEST_QUESTIONS_PATH) as f:
        questions = json.load(f)

    # Filter by tier if specified
    if tier_filter:
        questions = [q for q in questions if q["tier"] == tier_filter]

    # ── Load already completed questions to skip duplicates ──────────────────
    completed = set()
    if BENCHMARK_CSV.exists():
        with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(f"{row['architecture']}_{row['question_id']}")
    if completed:
        print(f"Found {len(completed)} already completed results — will skip duplicates.")

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    evaluator   = RAGASEvaluator(groq_client)

    # Import RAG architectures
    architectures = {}
    if arch_filter in ["all", "hybrid"]:
        print("Loading HybridRAG...")
        from rag.hybrid_rag import HybridRAG
        architectures["hybrid"] = HybridRAG()

    if arch_filter in ["all", "graph"]:
        print("Loading GraphRAG...")
        from rag.graph_rag import GraphRAG
        architectures["graph"] = GraphRAG()

    if arch_filter in ["all", "agentic"]:
        print("Loading AgenticRAG...")
        from rag.agentic_rag import AgenticRAG
        architectures["agentic"] = AgenticRAG()

    # CSV setup — append mode so previous results are preserved
    file_exists = BENCHMARK_CSV.exists()
    csv_file    = open(BENCHMARK_CSV, "a", newline="", encoding="utf-8")
    writer      = csv.DictWriter(csv_file, fieldnames=[
        "timestamp", "architecture", "question_id", "tier", "category",
        "question", "answer", "latency_sec", "tokens_used", "cost_usd",
        "num_chunks", "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "ragas_avg",
    ])
    if not file_exists:
        writer.writeheader()

    total    = len(questions) * len(architectures)
    done     = 0
    all_rows = []

    for arch_name, rag in architectures.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {arch_name.upper()} ({len(questions)} questions)")
        print(f"{'='*60}")

        for q in questions:
            done += 1

            # ── Skip if already completed ────────────────────────────────────
            skip_key = f"{arch_name}_{q['id']}"
            if skip_key in completed:
                print(f"[{done}/{total}] Skipping {skip_key} — already completed")
                continue

            print(f"[{done}/{total}] {arch_name} | T{q['tier']} | {q['question'][:60]}...")

            try:
                # Run RAG query — retry once on rate limit
                result = None
                for attempt in range(2):
                    try:
                        result = rag.query(q["question"])
                        break
                    except Exception as e:
                        if "429" in str(e) and attempt == 0:
                            print(f"   Rate limit on query — waiting 60s...")
                            _time.sleep(60)
                        else:
                            raise

                if result is None:
                    raise Exception("Query failed after retry")

                # Calculate cost
                cost = calculate_cost(
                    result.get("tokens_used", 0) // 2,
                    result.get("tokens_used", 0) // 2,
                )

                # RAGAS evaluation
                ragas_scores = evaluator.evaluate(
                    q["question"],
                    result["answer"],
                    result.get("retrieved_chunks", []),
                )
                ragas_avg = round(sum(ragas_scores.values()) / len(ragas_scores), 4)

                row = {
                    "timestamp":         datetime.now().isoformat(),
                    "architecture":      arch_name,
                    "question_id":       q["id"],
                    "tier":              q["tier"],
                    "category":          q["category"],
                    "question":          q["question"],
                    "answer":            result["answer"][:500],
                    "latency_sec":       result.get("latency_sec", ""),
                    "tokens_used":       result.get("tokens_used", ""),
                    "cost_usd":          cost,
                    "num_chunks":        result.get("num_chunks", ""),
                    "faithfulness":      ragas_scores["faithfulness"],
                    "answer_relevancy":  ragas_scores["answer_relevancy"],
                    "context_precision": ragas_scores["context_precision"],
                    "context_recall":    ragas_scores["context_recall"],
                    "ragas_avg":         ragas_avg,
                }

                writer.writerow(row)
                csv_file.flush()
                all_rows.append(row)

                print(f"   ✓ Latency: {result.get('latency_sec')}s | "
                      f"RAGAS avg: {ragas_avg:.3f} | Cost: ${cost:.5f}")

            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue

            # Small delay to avoid rate limiting
            time.sleep(1)

    csv_file.close()

    # Regenerate summary from full CSV (includes previous runs)
    _generate_summary_from_csv()
    print(f"\nBenchmark complete → {BENCHMARK_CSV}")
    print(f"Summary → {SUMMARY_JSON}")


def _generate_summary_from_csv() -> None:
    """
    Reads the full benchmark.csv and generates aggregated summary.
    This way summary always reflects ALL runs including previous ones.
    """
    if not BENCHMARK_CSV.exists():
        return

    rows = []
    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    summary = defaultdict(lambda: {
        "count": 0,
        "avg_latency": 0,
        "avg_cost": 0,
        "avg_faithfulness": 0,
        "avg_answer_relevancy": 0,
        "avg_context_precision": 0,
        "avg_context_recall": 0,
        "avg_ragas": 0,
        "by_tier": defaultdict(lambda: {"count": 0, "avg_ragas": 0}),
    })

    for row in rows:
        arch = row.get("architecture", "")
        if not arch:
            continue
        summary[arch]["count"] += 1
        summary[arch]["avg_latency"]           += float(row.get("latency_sec") or 0)
        summary[arch]["avg_cost"]              += float(row.get("cost_usd") or 0)
        summary[arch]["avg_faithfulness"]      += float(row.get("faithfulness") or 0)
        summary[arch]["avg_answer_relevancy"]  += float(row.get("answer_relevancy") or 0)
        summary[arch]["avg_context_precision"] += float(row.get("context_precision") or 0)
        summary[arch]["avg_context_recall"]    += float(row.get("context_recall") or 0)
        summary[arch]["avg_ragas"]             += float(row.get("ragas_avg") or 0)

        tier = str(row.get("tier", ""))
        summary[arch]["by_tier"][tier]["count"]     += 1
        summary[arch]["by_tier"][tier]["avg_ragas"] += float(row.get("ragas_avg") or 0)

    final = {}
    for arch, data in summary.items():
        n = data["count"] or 1
        final[arch] = {
            "total_questions":       data["count"],
            "avg_latency_sec":       round(data["avg_latency"] / n, 2),
            "avg_cost_usd":          round(data["avg_cost"] / n, 6),
            "avg_faithfulness":      round(data["avg_faithfulness"] / n, 4),
            "avg_answer_relevancy":  round(data["avg_answer_relevancy"] / n, 4),
            "avg_context_precision": round(data["avg_context_precision"] / n, 4),
            "avg_context_recall":    round(data["avg_context_recall"] / n, 4),
            "avg_ragas_score":       round(data["avg_ragas"] / n, 4),
            "by_tier": {
                tier: {
                    "count":     td["count"],
                    "avg_ragas": round(td["avg_ragas"] / (td["count"] or 1), 4)
                }
                for tier, td in data["by_tier"].items()
            }
        }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(final, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthRAG Benchmark Evaluation")
    parser.add_argument("--arch", default="all",
                        choices=["all", "hybrid", "graph", "agentic"],
                        help="Which architecture to evaluate")
    parser.add_argument("--tier", type=int, default=None,
                        choices=[1, 2, 3],
                        help="Only run questions from this tier")
    args = parser.parse_args()

    run_benchmark(arch_filter=args.arch, tier_filter=args.tier)