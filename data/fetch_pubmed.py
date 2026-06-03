"""
fetch_pubmed.py
---------------
Fetches PubMed abstracts across 5 clinical conditions using the NCBI E-utilities API.
No API key required for <= 3 requests/sec. With NCBI_API_KEY set, limit is 10/sec.

Output:
    data/raw/pubmed/          one .json file per condition
    data/raw/pubmed/corpus.json   merged corpus used by indexing pipeline

Usage:
    python data/fetch_pubmed.py
"""

import os
import time
import json
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NCBI_API_KEY   = os.getenv("NCBI_API_KEY", "")           # optional but recommended
BASE_URL       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUTPUT_DIR     = Path("data/raw/pubmed")
RESULTS_PER_CONDITION = 100                               # 500 total abstracts

CONDITIONS = {
    "cancer":         "cancer treatment clinical trial outcomes",
    "diabetes":       "type 2 diabetes management insulin resistance",
    "cardiovascular": "cardiovascular disease heart failure treatment",
    "alzheimers":     "alzheimers disease dementia cognitive decline",
    "sepsis":         "sepsis treatment ICU outcomes critical care",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int) -> list[str]:
    """Returns list of PubMed IDs for a search query."""
    params = {
        "db":       "pubmed",
        "term":     query,
        "retmax":   max_results,
        "retmode":  "json",
        "sort":     "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetches abstract text + metadata for a list of PubMed IDs (batch of 100)."""
    records = []
    batch_size = 100

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {
            "db":      "pubmed",
            "id":      ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=30)
        resp.raise_for_status()

        # Parse XML manually — avoids heavy dependency (lxml/BeautifulSoup)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID", "")
                title = article.findtext(".//ArticleTitle", "")
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(
                    (p.get("Label", "") + ": " if p.get("Label") else "") + (p.text or "")
                    for p in abstract_parts
                ).strip()

                pub_year = article.findtext(".//PubDate/Year", "")
                journal = article.findtext(".//Journal/Title", "")
                authors_nodes = article.findall(".//Author")
                authors = [
                    f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
                    for a in authors_nodes[:3]
                ]

                if abstract:                                # skip records with no abstract
                    records.append({
                        "pmid":     pmid,
                        "title":    title,
                        "abstract": abstract,
                        "year":     pub_year,
                        "journal":  journal,
                        "authors":  authors,
                    })
            except Exception:
                continue

        # Polite rate limiting
        sleep = 0.11 if NCBI_API_KEY else 0.35
        time.sleep(sleep)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = []

    for condition, query in CONDITIONS.items():
        print(f"[{condition}] Searching PubMed...")
        pmids = search_pubmed(query, RESULTS_PER_CONDITION)
        print(f"[{condition}] Found {len(pmids)} IDs — fetching abstracts...")

        records = fetch_abstracts(pmids)

        # Tag each record with its condition for downstream filtering
        for r in records:
            r["condition"] = condition

        # Save per-condition file
        out_path = OUTPUT_DIR / f"{condition}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"[{condition}] Saved {len(records)} abstracts → {out_path}")
        corpus.extend(records)

    # Save merged corpus
    corpus_path = OUTPUT_DIR / "corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\nTotal abstracts: {len(corpus)} → {corpus_path}")


if __name__ == "__main__":
    main()
