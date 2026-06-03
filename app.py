"""
app.py
------
HealthRAG Benchmark — Streamlit Application

Three tabs:
  1. Chat      — Side-by-side comparison of all 3 RAG architectures
  2. Benchmark — RAGAS evaluation results + Plotly charts
  3. Explorer  — CMS hospital data visualization (connected to Snowflake)

Deployment: Hugging Face Spaces (Streamlit SDK)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthRAG Benchmark",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #595959;
        margin-bottom: 2rem;
    }
    .arch-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2E75B6;
        margin-bottom: 1rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .metric-chip {
        background: #E6F1FB;
        color: #185FA5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .sql-block {
        background: #1E1E1E;
        color: #D4D4D4;
        padding: 0.75rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Load RAG models (cached so they only load once) ───────────────────────────

@st.cache_resource(show_spinner="Loading RAG models...")
def load_models():
    """
    Loads all 3 RAG architectures once and caches them.
    
    Why @st.cache_resource?
        Without caching, models reload on every user interaction.
        BGE reranker alone is 1.1GB — reloading it on every query
        would make the app unusable. cache_resource persists the
        model objects across all user sessions.
    """
    models = {}
    try:
        from rag.hybrid_rag import HybridRAG
        models["hybrid"] = HybridRAG()
    except Exception as e:
        models["hybrid"] = None
        st.warning(f"HybridRAG failed to load: {e}")

    try:
        from rag.graph_rag import GraphRAG
        models["graph"] = GraphRAG()
    except Exception as e:
        models["graph"] = None
        st.warning(f"GraphRAG failed to load: {e}")

    try:
        from rag.agentic_rag import AgenticRAG
        models["agentic"] = AgenticRAG()
    except Exception as e:
        models["agentic"] = None
        st.warning(f"AgenticRAG failed to load: {e}")

    return models


# ── Load benchmark results ────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_benchmark_data():
    """
    Loads benchmark.csv and summary.json.
    ttl=300 means it refreshes every 5 minutes so live eval results
    appear in the dashboard without restarting the app.
    """
    csv_path  = Path("evaluation/results/benchmark.csv")
    json_path = Path("evaluation/results/summary.json")

    df      = pd.read_csv(csv_path)      if csv_path.exists()  else pd.DataFrame()
    summary = json.load(open(json_path)) if json_path.exists() else {}
    return df, summary


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

def render_chat_tab(models: dict):
    st.markdown("### Ask a Healthcare Question")
    st.markdown("The same question runs through all 3 RAG architectures simultaneously.")

    # Example questions
    examples = [
        "What are the treatment options for sepsis in ICU patients?",
        "Which states have the highest average MRSA infection rates?",
        "What is the relationship between diabetes and cardiovascular risk?",
        "Which hospital types have the highest quality ratings?",
        "What interventions slow Alzheimer's disease progression?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the symptoms of sepsis?",
            label_visibility="collapsed",
        )
    with col2:
        ask = st.button("Ask All 3 Architectures", type="primary", use_container_width=True)

    # Example question buttons
    st.markdown("**Try an example:**")
    ex_cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(ex_cols, examples)):
        with col:
            if st.button(ex[:40] + "...", key=f"ex_{i}", use_container_width=True):
                question = ex
                ask = True

    if ask and question:
        st.markdown("---")
        st.markdown(f"**Q: {question}**")
        st.markdown("")

        # Run all 3 architectures in columns
        cols = st.columns(3)
        arch_configs = [
            ("hybrid",  "🔵 Hybrid RAG",  "#2E75B6"),
            ("graph",   "🟢 Graph RAG",   "#2D6A4F"),
            ("agentic", "🟠 Agentic RAG", "#C55A11"),
        ]

        for col, (arch_name, arch_label, color) in zip(cols, arch_configs):
            with col:
                st.markdown(f"#### {arch_label}")
                rag = models.get(arch_name)

                if rag is None:
                    st.error("Model not loaded")
                    continue

                with st.spinner(f"Running {arch_label}..."):
                    try:
                        start  = time.time()
                        result = rag.query(question)
                        elapsed = time.time() - start

                        # Answer
                        st.markdown(result["answer"])

                        # Metrics row
                        st.markdown("---")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Latency", f"{result.get('latency_sec', round(elapsed,1))}s")
                        m2.metric("Chunks", result.get("num_chunks", 0))
                        m3.metric("Tokens", result.get("tokens_used", 0))

                        # Architecture-specific details
                        if arch_name == "agentic":
                            if result.get("sql_query"):
                                with st.expander("SQL Query"):
                                    st.code(result["sql_query"], language="sql")
                            if result.get("tool_calls"):
                                st.caption(f"Tools: {' → '.join(result['tool_calls'])}")

                        if arch_name == "graph":
                            st.caption(f"Graph hits: {result.get('graph_hits', 0)} | "
                                      f"Entities: {', '.join(result.get('query_entities', [])[:3])}")

                        # Retrieved chunks expander
                        chunks = result.get("retrieved_chunks", [])
                        if chunks:
                            with st.expander(f"Retrieved chunks ({len(chunks)})"):
                                for i, chunk in enumerate(chunks[:3]):
                                    source = chunk.get("source", "")
                                    if source == "pubmed":
                                        label = f"PubMed | {chunk.get('condition','')} | PMID:{chunk.get('pmid','')}"
                                    else:
                                        label = f"CMS | {chunk.get('facility_name','')} | {chunk.get('state','')}"
                                    st.caption(f"**Chunk {i+1}:** {label}")
                                    st.text(chunk.get("text", "")[:300] + "...")
                                    st.divider()

                    except Exception as e:
                        st.error(f"Error: {str(e)[:200]}")


# ── Tab 2: Benchmark ──────────────────────────────────────────────────────────

def render_benchmark_tab():
    st.markdown("### Benchmark Results")
    st.markdown("RAGAS evaluation across 50 questions × 3 architectures × 4 metrics.")

    df, summary = load_benchmark_data()

    if df.empty:
        st.info("Benchmark evaluation is still running. Results will appear here automatically.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    arch_colors = {"hybrid": "#2E75B6", "graph": "#2D6A4F", "agentic": "#C55A11"}

    st.markdown("#### Overall Performance")
    cols = st.columns(len(summary))
    for col, (arch, data) in zip(cols, summary.items()):
        with col:
            color = arch_colors.get(arch, "#333")
            st.markdown(f"<div style='border-left:4px solid {color}; padding-left:12px'>", unsafe_allow_html=True)
            st.markdown(f"**{arch.upper()} RAG**")
            st.metric("RAGAS Score", f"{data.get('avg_ragas_score', 0):.3f}")
            st.metric("Avg Latency", f"{data.get('avg_latency_sec', 0):.1f}s")
            st.metric("Avg Cost", f"${data.get('avg_cost_usd', 0):.5f}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Radar chart — RAGAS metrics per architecture ──────────────────────────
    st.markdown("#### RAGAS Metrics Comparison")

    metrics = ["avg_faithfulness", "avg_answer_relevancy",
               "avg_context_precision", "avg_context_recall"]
    metric_labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]

    fig_radar = go.Figure()
    for arch, data in summary.items():
        values = [data.get(m, 0) for m in metrics]
        values += values[:1]   # close the radar loop
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + metric_labels[:1],
            fill="toself",
            name=arch.upper(),
            line_color=arch_colors.get(arch, "#333"),
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Latency vs Quality scatter ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Latency vs RAGAS Score")
        fig_scatter = px.scatter(
            df,
            x="latency_sec",
            y="ragas_avg",
            color="architecture",
            symbol="tier",
            color_discrete_map=arch_colors,
            labels={"latency_sec": "Latency (s)", "ragas_avg": "RAGAS Avg Score",
                    "architecture": "Architecture", "tier": "Tier"},
            hover_data=["question"],
            height=400,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.markdown("#### RAGAS Score by Tier")
        tier_data = df.groupby(["architecture", "tier"])["ragas_avg"].mean().reset_index()
        fig_tier = px.bar(
            tier_data,
            x="tier",
            y="ragas_avg",
            color="architecture",
            barmode="group",
            color_discrete_map=arch_colors,
            labels={"tier": "Question Tier", "ragas_avg": "Avg RAGAS Score",
                    "architecture": "Architecture"},
            height=400,
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    # ── Cost comparison ───────────────────────────────────────────────────────
    st.markdown("#### Cost per Query")
    cost_data = df.groupby("architecture")["cost_usd"].mean().reset_index()
    fig_cost  = px.bar(
        cost_data,
        x="architecture",
        y="cost_usd",
        color="architecture",
        color_discrete_map=arch_colors,
        labels={"architecture": "Architecture", "cost_usd": "Avg Cost (USD)"},
        height=350,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # ── Full results table ────────────────────────────────────────────────────
    st.markdown("#### Full Results Table")
    display_cols = ["architecture", "tier", "category", "question",
                    "latency_sec", "faithfulness", "answer_relevancy",
                    "context_precision", "context_recall", "ragas_avg", "cost_usd"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values(["architecture", "tier"]),
        use_container_width=True,
        height=400,
    )


# ── Tab 3: Data Explorer ──────────────────────────────────────────────────────

def render_explorer_tab():
    st.markdown("### Healthcare Data Explorer")
    st.markdown("Live data from Snowflake — CMS hospital quality and HAI infection rates.")

    # Load CMS data from local CSV (faster than Snowflake for dashboard)
    hospitals_path    = Path("data/raw/cms/hospitals.csv")
    complications_path = Path("data/raw/cms/complications.csv")

    if not hospitals_path.exists():
        st.error("CMS data not found. Run `python data/fetch_cms.py` first.")
        return

    df_h = pd.read_csv(hospitals_path)
    df_c = pd.read_csv(complications_path)

    # ── State-level hospital ratings map ─────────────────────────────────────
    st.markdown("#### Average Hospital Rating by State")

    if "state" in df_h.columns and "hospital_overall_rating" in df_h.columns:
        df_h["hospital_overall_rating"] = pd.to_numeric(
            df_h["hospital_overall_rating"], errors="coerce"
        )
        state_ratings = df_h.groupby("state")["hospital_overall_rating"].mean().reset_index()
        state_ratings.columns = ["state", "avg_rating"]

        fig_map = px.choropleth(
            state_ratings,
            locations="state",
            locationmode="USA-states",
            color="avg_rating",
            scope="usa",
            color_continuous_scale="Blues",
            labels={"avg_rating": "Avg Rating"},
            height=450,
        )
        fig_map.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_map, use_container_width=True)

    # ── HAI infection rates ───────────────────────────────────────────────────
    st.markdown("#### HAI Infection Rates by State")

    hai_cols = ["clabsi_sir", "cauti_sir", "mrsa_sir", "cdi_sir", "ssi_sir"]
    hai_cols = [c for c in hai_cols if c in df_c.columns]

    if hai_cols and "state" in df_c.columns:
        selected_metric = st.selectbox(
            "Select infection type",
            hai_cols,
            format_func=lambda x: x.upper().replace("_SIR", ""),
        )

        df_c[selected_metric] = pd.to_numeric(df_c[selected_metric], errors="coerce")
        state_hai = df_c.groupby("state")[selected_metric].mean().reset_index()
        state_hai.columns = ["state", "avg_sir"]
        state_hai = state_hai.dropna().sort_values("avg_sir", ascending=False).head(20)

        fig_hai = px.bar(
            state_hai,
            x="state",
            y="avg_sir",
            color="avg_sir",
            color_continuous_scale="Reds",
            labels={"state": "State", "avg_sir": f"Avg {selected_metric.upper()}"},
            height=400,
        )
        st.plotly_chart(fig_hai, use_container_width=True)

    # ── Hospital type breakdown ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Hospital Types")
        if "hospital_type" in df_h.columns:
            type_counts = df_h["hospital_type"].value_counts().reset_index()
            type_counts.columns = ["type", "count"]
            fig_type = px.pie(
                type_counts,
                values="count",
                names="type",
                height=350,
            )
            st.plotly_chart(fig_type, use_container_width=True)

    with col2:
        st.markdown("#### Emergency Services Coverage")
        if "emergency_services" in df_h.columns:
            em_counts = df_h["emergency_services"].value_counts().reset_index()
            em_counts.columns = ["has_emergency", "count"]
            fig_em = px.pie(
                em_counts,
                values="count",
                names="has_emergency",
                color_discrete_sequence=["#2E75B6", "#C55A11"],
                height=350,
            )
            st.plotly_chart(fig_em, use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────────────────
    st.markdown("#### Hospital Data Table")
    keep = ["facility_name", "citytown", "state", "hospital_type",
            "hospital_ownership", "emergency_services", "hospital_overall_rating"]
    keep = [c for c in keep if c in df_h.columns]
    st.dataframe(df_h[keep].head(100), use_container_width=True, height=350)


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-header">🏥 HealthRAG Benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Comparing Hybrid, Graph, and Agentic RAG architectures '
        'on real healthcare data — PubMed abstracts + CMS hospital quality data + Snowflake</div>',
        unsafe_allow_html=True
    )

    # Load models
    models = load_models()

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "💬 Chat — Compare Architectures",
        "📊 Benchmark Results",
        "🏥 Data Explorer",
    ])

    with tab1:
        render_chat_tab(models)

    with tab2:
        render_benchmark_tab()

    with tab3:
        render_explorer_tab()


if __name__ == "__main__":
    main()
