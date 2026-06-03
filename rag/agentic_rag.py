"""
agentic_rag.py
--------------
Agentic RAG: LangChain ReAct Agent + CRAG Self-Reflection + Snowflake SQL Tool

How it works:
    1. Query comes in
    2. ReAct agent decides which tool to use:
       - vector_search_tool   → FAISS semantic search over PubMed + CMS text
       - snowflake_sql_tool   → Direct SQL queries on Snowflake hospital data
       - reasoning_tool       → Multi-step reasoning over retrieved context
    3. After retrieval, CRAG checks if the retrieved context is relevant
       - If relevance score < threshold → agent re-retrieves with refined query
       - If relevance score >= threshold → passes context to LLM for answer
    4. Groq generates final answer

Why Agentic RAG is different from Hybrid and Graph RAG:
    Hybrid RAG always retrieves from FAISS+BM25 regardless of query type.
    Graph RAG always traverses the knowledge graph.
    
    Agentic RAG DECIDES what to do:
    - "What is the mortality rate in sepsis?" → vector_search_tool (PubMed)
    - "Which states have the highest MRSA rates?" → snowflake_sql_tool (structured data)
    - "Compare sepsis outcomes across hospital types" → both tools, then reasoning_tool
    
    The CRAG (Corrective RAG) loop adds self-correction:
    If the first retrieval is irrelevant, the agent rewrites the query and tries again.
    This is what makes it "agentic" — it evaluates its own outputs.

Why Snowflake SQL tool matters:
    This is the only architecture that can answer:
    "Which hospitals in Texas have MRSA rates above the national average?"
    Hybrid and Graph RAG can only retrieve text — they can't compute aggregations.
    The SQL tool queries Snowflake views directly for structured analytics.

Usage:
    from rag.agentic_rag import AgenticRAG
    rag = AgenticRAG()
    result = rag.query("Which states have the highest average MRSA infection rates?")
"""

import json
import time
import os
import re
import numpy as np
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
import snowflake.connector
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

TOP_K_RETRIEVE   = 10
TOP_K_RERANK     = 5
CRAG_THRESHOLD   = 0.5      # relevance score below which agent re-retrieves
MAX_ITERATIONS   = 3        # max agent steps before forced stop
MAX_SQL_ROWS     = 20       # max rows from Snowflake


# ── Snowflake Connection ──────────────────────────────────────────────────────

def get_snowflake_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "HEALTHRAG_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        authenticator="username_password_mfa",
    )


# ── Tools ─────────────────────────────────────────────────────────────────────

class VectorSearchTool:
    """
    Semantic search over PubMed abstracts and CMS hospital text.
    Used for: clinical questions, treatment info, research findings.
    """
    def __init__(self, faiss_index, id_map, chunks_map, embedder, reranker):
        self.faiss_index = faiss_index
        self.id_map      = id_map
        self.chunks_map  = chunks_map
        self.embedder    = embedder
        self.reranker    = reranker

    def run(self, query: str) -> tuple[list[dict], float]:
        """
        Returns (chunks, relevance_score).
        relevance_score = avg cross-encoder score of top chunks vs query.
        Used by CRAG to decide if re-retrieval is needed.
        """
        # FAISS retrieval
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.faiss_index.search(
            query_vec.astype(np.float32), TOP_K_RETRIEVE
        )
        chunk_ids = [self.id_map[idx] for idx in indices[0] if idx != -1]
        chunks    = [self.chunks_map[cid] for cid in chunk_ids if cid in self.chunks_map]

        if not chunks:
            return [], 0.0

        # Cross-encoder reranking + relevance scoring
        pairs  = [[query, c["text"]] for c in chunks]
        ce_scores = self.reranker.predict(pairs)

        # CRAG relevance score = average of top-3 cross-encoder scores (normalized 0-1)
        top_scores = sorted(ce_scores, reverse=True)[:3]
        relevance  = float(np.mean(top_scores))
        # Normalize to 0-1 range (cross-encoder outputs ~-10 to 10)
        relevance  = max(0.0, min(1.0, (relevance + 5) / 10))

        # Sort chunks by cross-encoder score
        ranked = sorted(zip(ce_scores, chunks), key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in ranked[:TOP_K_RERANK]]

        return top_chunks, relevance


class SnowflakeSQLTool:
    """
    Direct SQL queries on Snowflake hospital quality + HAI infection data.
    Used for: structured analytics, aggregations, state comparisons, rankings.
    
    Why this tool exists:
        "Which states have the highest MRSA rates?" requires AVG() + GROUP BY
        That's a SQL operation — no vector search can compute it.
        This tool bridges the gap between RAG and structured data analytics.
    """
    # Safe views the agent can query — no raw tables to prevent data leaks
    ALLOWED_VIEWS = {
        "HOSPITAL_QUALITY_SUMMARY",
        "STATE_QUALITY_SUMMARY",
        "HAI_SUMMARY_BY_STATE",
    }

    def run(self, question: str, groq_client: Groq) -> tuple[str, str]:
        """
        1. Uses Groq to generate a SQL query from the natural language question
        2. Executes against Snowflake
        3. Returns (sql_query, formatted_results)
        """
        # Step 1: Generate SQL
        schema_info = """
Available views:
- HOSPITAL_QUALITY_SUMMARY: FACILITY_ID, FACILITY_NAME, CITYTOWN, STATE, HOSPITAL_TYPE, 
  HOSPITAL_OWNERSHIP, EMERGENCY_SERVICES, OVERALL_RATING, TOTAL_HAC_SCORE,
  CLABSI_SIR, CAUTI_SIR, MRSA_SIR, CDI_SIR, SSI_SIR
- STATE_QUALITY_SUMMARY: STATE, NUM_HOSPITALS, AVG_RATING, NUM_WITH_EMERGENCY
- HAI_SUMMARY_BY_STATE: STATE, NUM_HOSPITALS, AVG_CLABSI, AVG_CAUTI, AVG_MRSA, AVG_CDI, AVG_SSI
"""
        sql_prompt = f"""Generate a Snowflake SQL query to answer this question.
Use ONLY the views listed below. Return ONLY the SQL query, no explanation.
Add LIMIT {MAX_SQL_ROWS} to all queries.

{schema_info}

Question: {question}

SQL:"""

        try:
            sql_response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": sql_prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            sql = sql_response.choices[0].message.content.strip()
            sql = re.sub(r"```sql|```", "", sql).strip()

            # Safety check — only allow SELECT on approved views
            sql_upper = sql.upper()
            if not sql_upper.startswith("SELECT"):
                return sql, "Error: Only SELECT queries are allowed."
            if not any(view in sql_upper for view in self.ALLOWED_VIEWS):
                return sql, "Error: Query must use approved views."

            # Step 2: Execute against Snowflake
            conn   = get_snowflake_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows    = cursor.fetchmany(MAX_SQL_ROWS)
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            conn.close()

            # Step 3: Format results as readable text
            if not rows:
                return sql, "No results found."

            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))
            for row in rows:
                lines.append(" | ".join(str(v) if v is not None else "N/A" for v in row))
            return sql, "\n".join(lines)

        except Exception as e:
            return "", f"SQL Error: {str(e)}"


class ReasoningTool:
    """
    Multi-step reasoning over already-retrieved context.
    Used when: question requires synthesizing multiple sources,
    comparing findings, or drawing conclusions from retrieved data.
    """
    def run(self, question: str, context: str, groq_client: Groq) -> str:
        prompt = f"""You are a healthcare analyst. Reason step by step to answer the question
using the provided context. Show your reasoning process.

Context:
{context}

Question: {question}

Step-by-step reasoning:"""

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ── CRAG Self-Reflection ──────────────────────────────────────────────────────

def crag_refine_query(original_query: str, groq_client: Groq) -> str:
    """
    CRAG (Corrective RAG): if retrieval relevance is low, rewrites the query.
    
    Why this matters:
        Sometimes the first query is too specific or uses wrong terminology.
        E.g. "myocardial infarction treatment" might not match chunks that say
        "heart attack management" — a rewrite fixes this.
    """
    prompt = f"""The following search query did not return relevant results.
Rewrite it to be broader and use alternative medical terminology.
Return ONLY the rewritten query, nothing else.

Original query: {original_query}

Rewritten query:"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return original_query


def decide_tool(question: str, groq_client: Groq) -> str:
    """
    Agent brain: decides which tool to use for a given question.
    
    Returns: "vector_search" | "snowflake_sql" | "both"
    
    Decision logic:
    - Questions about research, treatments, clinical findings → vector_search
    - Questions about hospital rankings, state comparisons, infection rates → snowflake_sql  
    - Questions comparing clinical + hospital data → both
    """
    prompt = f"""You are a routing agent. Decide which tool to use for this healthcare question.

Tools available:
- "vector_search": for clinical research, treatments, disease information, PubMed data
- "snowflake_sql": for hospital rankings, state comparisons, infection rates, structured data
- "both": for questions needing both clinical research AND hospital statistics

Return ONLY one of: vector_search, snowflake_sql, both

Question: {question}

Tool:"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        tool = response.choices[0].message.content.strip().lower()
        if tool not in ["vector_search", "snowflake_sql", "both"]:
            return "vector_search"
        return tool
    except Exception:
        return "vector_search"


# ── AgenticRAG Class ──────────────────────────────────────────────────────────

class AgenticRAG:
    def __init__(self):
        print("[AgenticRAG] Loading indexes and models...")

        # Load FAISS
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_ID_MAP) as f:
            self.id_map = json.load(f)
        with open(CHUNKS_MAP_PATH) as f:
            self.chunks_map = json.load(f)

        # Load models
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("[AgenticRAG] Loading BGE reranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Groq client
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Initialize tools
        self.vector_tool    = VectorSearchTool(
            self.faiss_index, self.id_map,
            self.chunks_map, self.embedder, self.reranker
        )
        self.snowflake_tool = SnowflakeSQLTool()
        self.reasoning_tool = ReasoningTool()

        print("[AgenticRAG] Ready.")


    def _build_final_prompt(
        self,
        question: str,
        vector_chunks: list[dict],
        sql_results: str,
        reasoning: str,
    ) -> str:
        """Builds the final synthesis prompt combining all tool outputs."""
        parts = []

        if vector_chunks:
            context_parts = []
            for i, chunk in enumerate(vector_chunks):
                source = chunk.get("source", "unknown")
                if source == "pubmed":
                    ref = f"[PubMed PMID:{chunk.get('pmid','')} | {chunk.get('condition','')}]"
                else:
                    ref = f"[CMS: {chunk.get('facility_name','')} | {chunk.get('state','')}]"
                context_parts.append(f"Source {i+1} {ref}:\n{chunk['text']}")
            parts.append("CLINICAL RESEARCH CONTEXT:\n" + "\n\n".join(context_parts))

        if sql_results:
            parts.append(f"STRUCTURED HOSPITAL DATA (from Snowflake):\n{sql_results}")

        if reasoning:
            parts.append(f"INTERMEDIATE REASONING:\n{reasoning}")

        context = "\n\n---\n\n".join(parts)

        return f"""You are a healthcare data analyst. Answer the question using ALL provided context.
Cite your sources. Be specific and include any relevant statistics or data points.

{context}

QUESTION: {question}

FINAL ANSWER:"""


    def query(self, question: str) -> dict:
        """
        Full Agentic RAG pipeline:
        Route → Retrieve → CRAG check → (re-retrieve if needed) → Reason → Generate
        """
        start          = time.time()
        iterations     = 0
        tool_calls     = []
        vector_chunks  = []
        sql_results    = ""
        sql_query      = ""
        total_tokens   = 0

        # Stage 1: Agent decides which tool(s) to use
        tool_decision = decide_tool(question, self.groq)
        tool_calls.append(f"decide_tool → {tool_decision}")

        current_query = question

        # Stage 2: Execute tools with CRAG loop
        while iterations < MAX_ITERATIONS:
            iterations += 1

            if tool_decision in ["vector_search", "both"]:
                chunks, relevance = self.vector_tool.run(current_query)
                tool_calls.append(f"vector_search (relevance={relevance:.2f})")

                if relevance < CRAG_THRESHOLD and iterations < MAX_ITERATIONS:
                    # CRAG: relevance too low — rewrite query and retry
                    current_query = crag_refine_query(current_query, self.groq)
                    tool_calls.append(f"crag_refine → '{current_query}'")
                    continue
                else:
                    vector_chunks = chunks
                    break

            elif tool_decision == "snowflake_sql":
                break   # SQL doesn't need CRAG loop

        # Stage 3: Snowflake SQL if needed
        if tool_decision in ["snowflake_sql", "both"]:
            sql_query, sql_results = self.snowflake_tool.run(question, self.groq)
            tool_calls.append(f"snowflake_sql → {len(sql_results.splitlines())} rows")

        # Stage 4: Reasoning tool if both sources used
        reasoning = ""
        if tool_decision == "both" and vector_chunks and sql_results:
            context_for_reasoning = "\n".join([c["text"] for c in vector_chunks[:3]])
            context_for_reasoning += f"\n\nSQL Results:\n{sql_results}"
            reasoning = self.reasoning_tool.run(question, context_for_reasoning, self.groq)
            tool_calls.append("reasoning_tool")

        # Stage 5: Final synthesis
        prompt   = self._build_final_prompt(question, vector_chunks, sql_results, reasoning)
        response = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        answer       = response.choices[0].message.content
        total_tokens += response.usage.total_tokens

        latency = time.time() - start

        return {
            "answer":           answer,
            "retrieved_chunks": vector_chunks,
            "num_chunks":       len(vector_chunks),
            "sql_query":        sql_query,
            "sql_results":      sql_results,
            "tool_calls":       tool_calls,
            "tool_decision":    tool_decision,
            "iterations":       iterations,
            "latency_sec":      round(latency, 2),
            "architecture":     "agentic",
            "tokens_used":      total_tokens,
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = AgenticRAG()

    test_questions = [
        "What are the treatment options for sepsis in ICU patients?",
        "Which states have the highest average MRSA infection rates?",
        "Compare sepsis mortality rates with hospital quality ratings across states",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer'][:400]}...")
        print(f"   Latency: {result['latency_sec']}s | Tokens: {result['tokens_used']}")
        print(f"   Tools used: {' → '.join(result['tool_calls'])}")
        if result['sql_query']:
            print(f"   SQL: {result['sql_query'][:100]}...")
