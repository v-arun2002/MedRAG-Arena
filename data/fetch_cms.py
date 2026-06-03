"""
fetch_cms.py
------------
Downloads CMS (Centers for Medicare & Medicaid Services) hospital quality
and cost data directly from the CMS public data API.

Two datasets:
  1. Hospital General Information   — hospital names, locations, overall ratings
  2. Complications and Deaths       — complication rates, mortality rates by condition

Output:
    data/raw/cms/hospitals.csv
    data/raw/cms/complications.csv
    data/raw/cms/cms_documents.json   — converted to text docs for RAG indexing

Usage:
    python data/fetch_cms.py
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/raw/cms")
MAX_RECORDS = 2000          # cap per dataset to stay manageable

# CMS Socrata Open Data API endpoints (no auth required)
CMS_ENDPOINTS = {
    "hospitals": "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0",
    "complications": "https://data.cms.gov/provider-data/api/1/datastore/query/yq43-i98g/0",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_cms_dataset(url: str, max_records: int) -> list[dict]:
    """Fetches records from CMS DKAN API with pagination."""
    records = []
    limit  = 500
    offset = 0

    while len(records) < max_records:
        params = {
            "limit":  limit,
            "offset": offset,
            "count":  "false",
            "schema": "false",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        records.extend(batch)
        offset += limit
        time.sleep(0.3)

    return records[:max_records]

def records_to_documents(hospitals: list[dict], complications: list[dict]) -> list[dict]:
    """
    Converts raw CMS records into natural-language text documents
    suitable for embedding and RAG retrieval.
    Each document = one hospital's summary (info + complication rates merged).
    """
    # Build a lookup: provider_id → complication rows
    comp_lookup: dict[str, list] = {}
    for row in complications:
        pid = row.get("provider_id", "")
        comp_lookup.setdefault(pid, []).append(row)

    documents = []
    for h in hospitals:
        pid      = h.get("provider_id", "")
        name     = h.get("hospital_name", "Unknown Hospital")
        city     = h.get("city", "")
        state    = h.get("state", "")
        rating   = h.get("hospital_overall_rating", "N/A")
        h_type   = h.get("hospital_type", "")
        ownership= h.get("hospital_ownership", "")
        emergency= h.get("emergency_services", "")

        # Build base description
        text = (
            f"{name} is a {h_type} hospital located in {city}, {state}. "
            f"It is {ownership}-owned and "
            f"{'provides' if emergency == 'Yes' else 'does not provide'} emergency services. "
            f"The hospital received an overall CMS quality rating of {rating} out of 5."
        )

        # Append complication/mortality data if available
        comps = comp_lookup.get(pid, [])
        if comps:
            text += " Clinical performance measures: "
            measure_parts = []
            for c in comps[:5]:             # cap at 5 measures per hospital
                measure = c.get("measure_name", "")
                score   = c.get("score", "N/A")
                compare = c.get("compared_to_national", "")
                if measure:
                    measure_parts.append(
                        f"{measure} score {score} ({compare} national average)"
                    )
            text += "; ".join(measure_parts) + "."

        documents.append({
            "provider_id": pid,
            "hospital_name": name,
            "state": state,
            "city": city,
            "overall_rating": rating,
            "text": text,
            "source": "CMS Hospital Quality Data",
        })

    return documents


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch hospital general info
    print("Fetching CMS hospital general information...")
    hospitals = fetch_cms_dataset(CMS_ENDPOINTS["hospitals"], MAX_RECORDS)
    df_hospitals = pd.DataFrame(hospitals)
    df_hospitals.to_csv(OUTPUT_DIR / "hospitals.csv", index=False)
    print(f"  Saved {len(hospitals)} hospitals → data/raw/cms/hospitals.csv")

    # 2. Fetch complications and deaths
    print("Fetching CMS complications and deaths data...")
    complications = fetch_cms_dataset(CMS_ENDPOINTS["complications"], MAX_RECORDS)
    df_complications = pd.DataFrame(complications)
    df_complications.to_csv(OUTPUT_DIR / "complications.csv", index=False)
    print(f"  Saved {len(complications)} rows → data/raw/cms/complications.csv")

    # 3. Convert to RAG documents
    print("Converting to RAG documents...")
    documents = records_to_documents(hospitals, complications)
    out_path = OUTPUT_DIR / "cms_documents.json"
    with open(out_path, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"  Saved {len(documents)} documents → {out_path}")

    print("\nCMS data fetch complete.")


if __name__ == "__main__":
    main()
