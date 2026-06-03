"""
graph_rag.py
------------
GraphRAG: Entity extraction → Knowledge Graph → Graph-aware retrieval → Groq

How it works:
    1. During indexing: Groq extracts entities and relationships from each chunk
       and stores them in a NetworkX graph (nodes = entities, edges = relationships)
    2. At query time: entities are extracted from the query
    3. The graph is traversed to find connected entities (1-2 hops)
    4. Chunks containing those entities are retrieved
    5. Groq generates an answer using retrieved chunks + graph context

Why GraphRAG?
    Hybrid RAG is great for single-document retrieval but struggles with
    multi-hop questions like:
    "Which hospitals in states with high MRSA rates also have low overall ratings?"
    
    This requires connecting:
        MRSA rates (from HAI data) → state → hospitals → ratings
    
    A knowledge graph explicitly stores these relationships so multi-hop
    reasoning becomes a graph traversal instead of hoping the retriever
    finds all the right chunks.

Why NetworkX instead of Neo4j?
    NetworkX is a pure Python in-memory graph library — no external database,
    no Docker, deploys to HF Spaces with zero infrastructure. For ~2500 chunks
    the graph fits comfortably in memory. Neo4j would be needed at 100k+ nodes.

Usage:
    from rag.graph_rag import GraphRAG
    rag = GraphRAG()
    result = rag.query("Which conditions are associated with ICU mortality?")
"""

import json
import time
import os
import re
import networkx as nx
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_MAP_PATH  = Path("data/cache/chunks_map.json")
FAISS_INDEX_PATH = Path("data/cache/faiss_index/index.faiss")
FAISS_ID_MAP     = Path("data/cache/faiss_index/id_map.json")
GRAPH_PATH       = Path("data/cache/knowledge_graph.pkl")

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL        = "llama-3.3-70b-versatile"
ENTITY_MODEL     = "llama-3.1-8b-instant"       # cheaper model for entity extraction

TOP_K_GRAPH      = 5       # top chunks from graph traversal
MAX_HOPS         = 2       # how deep to traverse the graph
MAX_NODES        = 50      # max entities to extract per chunk (keep graph manageable)
GRAPH_BUILD_SAMPLE = 200   # build graph from first N chunks (cost control)


# ── Entity Extraction ─────────────────────────────────────────────────────────

def extract_entities_and_relations(text: str, groq_client: Groq) -> dict:
    """
    Uses Groq llama-3.1-8b to extract entities and relationships from a chunk.
    
    Why a smaller model (8b) for extraction?
        Entity extraction is a structured task — simpler than reasoning.
        8b is 10x cheaper than 70b and fast enough for batch processing.
        We save 70b for the final answer generation where quality matters most.
    
    Returns: {"entities": [...], "relations": [{"source": ..., "relation": ..., "target": ...}]}
    """
    prompt = f"""Extract medical entities and their relationships from this text.
Return ONLY valid JSON, no explanation, no markdown.

Text: {text[:800]}

Return this exact format:
{{
  "entities": ["entity1", "entity2"],
  "relations": [
    {{"source": "entity1", "relation": "treats", "target": "entity2"}}
  ]
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=ENTITY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"entities": [], "relations": []}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_knowledge_graph(chunks_map: dict, groq_client: Groq) -> nx.DiGraph:
    """
    Builds a directed knowledge graph from chunk entities and relationships.
    
    Node = entity (e.g. "MRSA", "sepsis", "ICU mortality")
    Edge = relationship (e.g. "MRSA" --causes--> "bloodstream infection")
    Node attribute: chunk_ids — which chunks mention this entity
    
    We sample GRAPH_BUILD_SAMPLE chunks to control API costs.
    PubMed chunks are prioritized since they have richer clinical entities.
    """
    G = nx.DiGraph()

    # Prioritize PubMed chunks (richer entities) over CMS chunks
    pubmed_chunks = [(cid, c) for cid, c in chunks_map.items() if c.get("source") == "pubmed"]
    cms_chunks    = [(cid, c) for cid, c in chunks_map.items() if c.get("source") == "cms"]
    
    sample = pubmed_chunks[:int(GRAPH_BUILD_SAMPLE * 0.7)] + \
             cms_chunks[:int(GRAPH_BUILD_SAMPLE * 0.3)]

    print(f"[GraphRAG] Building knowledge graph from {len(sample)} chunks...")

    for i, (chunk_id, chunk) in enumerate(sample):
        if i % 20 == 0:
            print(f"  Processing chunk {i+1}/{len(sample)}...")

        extracted = extract_entities_and_relations(chunk["text"], groq_client)
        entities  = extracted.get("entities", [])[:MAX_NODES]
        relations = extracted.get("relations", [])

        # Add entities as nodes, track which chunks they appear in
        for entity in entities:
            entity = entity.lower().strip()
            if entity:
                if G.has_node(entity):
                    G.nodes[entity]["chunk_ids"].add(chunk_id)
                else:
                    G.add_node(entity, chunk_ids={chunk_id})

        # Add relationships as directed edges
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            src = (rel.get("source") or "").lower().strip()
            tgt = (rel.get("target") or "").lower().strip()
            rel_type = (rel.get("relation") or "related_to").lower().strip()
            if src and tgt and G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt, relation=rel_type)

    print(f"[GraphRAG] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── GraphRAG Class ────────────────────────────────────────────────────────────

class GraphRAG:
    def __init__(self):
        print("[GraphRAG] Loading indexes and models...")

        # Load chunks map
        with open(CHUNKS_MAP_PATH) as f:
            self.chunks_map = json.load(f)

        # Load FAISS for fallback retrieval
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_ID_MAP) as f:
            self.id_map = json.load(f)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Groq client
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Load or build knowledge graph
        if GRAPH_PATH.exists():
            print("[GraphRAG] Loading cached knowledge graph...")
            with open(GRAPH_PATH, "rb") as f:
                self.graph = pickle.load(f)
            print(f"[GraphRAG] Graph loaded: {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")
        else:
            print("[GraphRAG] No cached graph found — building from scratch...")
            print("           This will take ~3-5 minutes (Groq API calls for entity extraction)")
            GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.graph = build_knowledge_graph(self.chunks_map, self.groq)
            with open(GRAPH_PATH, "wb") as f:
                pickle.dump(self.graph, f)
            print(f"[GraphRAG] Graph cached → {GRAPH_PATH}")

        print("[GraphRAG] Ready.")


    # ── Graph Retrieval ───────────────────────────────────────────────────────

    def _extract_query_entities(self, query: str) -> list[str]:
        """
        Extracts key entities from the query using the cheap 8b model.
        These become the starting nodes for graph traversal.
        """
        prompt = f"""Extract the key medical/healthcare entities from this query.
Return ONLY a JSON array of strings, no explanation.

Query: {query}

Example output: ["diabetes", "insulin", "treatment"]"""

        try:
            response = self.groq.chat.completions.create(
                model=ENTITY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            entities = json.loads(raw)
            return [e.lower().strip() for e in entities if e]
        except Exception:
            # Fallback: simple keyword extraction
            stopwords = {"what", "which", "how", "are", "is", "the", "a", "an", "in", "of", "for"}
            return [w.lower() for w in query.split() if w.lower() not in stopwords]


    def _graph_traverse(self, query_entities: list[str], max_hops: int) -> set[str]:
        """
        Traverses the knowledge graph from query entity nodes.
        
        For each query entity found in the graph:
          - Collect all nodes within max_hops distance (neighbors of neighbors)
          - Collect all chunk_ids associated with those nodes
        
        Multi-hop example:
          Query: "ICU mortality in sepsis patients"
          Entities: ["icu mortality", "sepsis"]
          Hop 1: neighbors of "sepsis" → ["septic shock", "organ failure", "antibiotics"]
          Hop 2: neighbors of "organ failure" → ["mechanical ventilation", "ICU stay"]
          All chunk_ids from these nodes get retrieved → comprehensive multi-hop answer
        """
        relevant_chunk_ids = set()

        for entity in query_entities:
            # Find matching nodes (exact or partial match)
            matched_nodes = [n for n in self.graph.nodes if entity in n or n in entity]

            for node in matched_nodes:
                # Add chunks from this node
                relevant_chunk_ids.update(self.graph.nodes[node].get("chunk_ids", set()))

                # Traverse neighbors up to max_hops
                try:
                    neighbors = nx.single_source_shortest_path_length(
                        self.graph, node, cutoff=max_hops
                    )
                    for neighbor in neighbors:
                        relevant_chunk_ids.update(
                            self.graph.nodes[neighbor].get("chunk_ids", set())
                        )
                except Exception:
                    continue

        return relevant_chunk_ids


    def _faiss_fallback(self, query: str, top_k: int) -> list[str]:
        """
        Falls back to FAISS semantic search if graph traversal finds no chunks.
        Ensures we always return something even for queries with no graph matches.
        """
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.faiss_index.search(query_vec.astype(np.float32), top_k)
        return [self.id_map[idx] for idx in indices[0] if idx != -1]


    # ── Generation ────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list[dict], graph_context: str) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            if source == "pubmed":
                ref = f"[PubMed PMID:{chunk.get('pmid','')} | {chunk.get('condition','')}]"
            else:
                ref = f"[CMS: {chunk.get('facility_name','')} | {chunk.get('state','')}]"
            context_parts.append(f"Source {i+1} {ref}:\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        return f"""You are a healthcare data analyst assistant with access to a medical knowledge graph.
Answer the question using the provided context and entity relationships.
Always cite which sources you used.

ENTITY RELATIONSHIPS FROM KNOWLEDGE GRAPH:
{graph_context}

RETRIEVED CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


    def query(self, question: str) -> dict:
        """
        Full GraphRAG pipeline:
        Query entities → Graph traversal → Chunk retrieval → Groq
        """
        start = time.time()

        # Stage 1: Extract entities from query
        query_entities = self._extract_query_entities(question)

        # Stage 2: Graph traversal
        relevant_ids = self._graph_traverse(query_entities, MAX_HOPS)

        # Stage 3: Get chunks — fallback to FAISS if graph finds nothing
        if len(relevant_ids) >= TOP_K_GRAPH:
            # Score by entity overlap and take top-k
            chunk_ids = list(relevant_ids)[:TOP_K_GRAPH * 3]
            # Re-rank by FAISS similarity within the graph-retrieved set
            query_vec = self.embedder.encode(
                [question], normalize_embeddings=True, convert_to_numpy=True
            )
            scored = []
            for cid in chunk_ids:
                if cid in self.chunks_map:
                    chunk_vec = self.embedder.encode(
                        [self.chunks_map[cid]["text"]],
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    score = float(np.dot(query_vec[0], chunk_vec[0]))
                    scored.append((score, cid))
            scored.sort(reverse=True)
            final_ids = [cid for _, cid in scored[:TOP_K_GRAPH]]
        else:
            print(f"[GraphRAG] Graph found {len(relevant_ids)} chunks — using FAISS fallback")
            final_ids = self._faiss_fallback(question, TOP_K_GRAPH)

        chunks = [self.chunks_map[cid] for cid in final_ids if cid in self.chunks_map]

        # Stage 4: Build graph context summary for prompt
        graph_context = f"Query entities identified: {', '.join(query_entities)}\n"
        graph_context += f"Graph traversal found {len(relevant_ids)} related chunks across {MAX_HOPS} hops."

        # Stage 5: LLM generation
        prompt   = self._build_prompt(question, chunks, graph_context)
        response = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        latency = time.time() - start

        return {
            "answer":           answer,
            "retrieved_chunks": chunks,
            "num_chunks":       len(chunks),
            "query_entities":   query_entities,
            "graph_hits":       len(relevant_ids),
            "latency_sec":      round(latency, 2),
            "architecture":     "graph",
            "tokens_used":      response.usage.total_tokens,
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = GraphRAG()

    test_questions = [
        "What conditions are associated with ICU mortality?",
        "Which treatments are used for cardiovascular disease?",
        "What is the relationship between diabetes and cardiovascular risk?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer'][:300]}...")
        print(f"   Latency: {result['latency_sec']}s | Tokens: {result['tokens_used']} | "
              f"Graph hits: {result['graph_hits']} | Entities: {result['query_entities']}")
