"""
load_snowflake.py
-----------------
Loads CMS hospital data into Snowflake for:
  1. Agentic RAG SQL tool — live querying during RAG sessions
  2. Power BI dashboard — connect Snowflake as a data source in Power BI Desktop

Prerequisites:
    pip install snowflake-connector-python pandas

Environment variables (add to .env):
    SNOWFLAKE_ACCOUNT    e.g. LCSLWOH-HCC93385
    SNOWFLAKE_USER
    SNOWFLAKE_PASSWORD
    SNOWFLAKE_WAREHOUSE  e.g. COMPUTE_WH
    SNOWFLAKE_DATABASE   e.g. HEALTHRAG_DB
    SNOWFLAKE_SCHEMA     e.g. PUBLIC

Usage:
    python data/load_snowflake.py
"""

import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SNOWFLAKE_CONFIG = {
    "account":       os.getenv("SNOWFLAKE_ACCOUNT"),
    "user":          os.getenv("SNOWFLAKE_USER"),
    "password":      os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse":     os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database":      os.getenv("SNOWFLAKE_DATABASE", "HEALTHRAG_DB"),
    "schema":        os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
    "authenticator": "username_password_mfa",
}

CMS_DIR = Path("data/raw/cms")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_connection():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


def setup_database(cursor):
    """Create database and schema if they don't exist."""
    db     = SNOWFLAKE_CONFIG["database"]
    schema = SNOWFLAKE_CONFIG["schema"]
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cursor.execute(f"USE DATABASE {db}")
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cursor.execute(f"USE SCHEMA {schema}")
    print(f"Database: {db}.{schema} ready.")


def load_hospitals(cursor, conn):
    """Load hospital general info into HOSPITALS table."""
    df = pd.read_csv(CMS_DIR / "hospitals.csv")
    df.columns = [c.upper().replace(" ", "_").replace("-", "_") for c in df.columns]

    keep = [
        "FACILITY_ID", "FACILITY_NAME", "ADDRESS", "CITYTOWN", "STATE",
        "ZIP_CODE", "COUNTYPARISH", "TELEPHONE_NUMBER", "HOSPITAL_TYPE",
        "HOSPITAL_OWNERSHIP", "EMERGENCY_SERVICES", "HOSPITAL_OVERALL_RATING",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    cursor.execute("DROP TABLE IF EXISTS HOSPITALS")
    success, nchunks, nrows, _ = write_pandas(
        conn, df, "HOSPITALS",
        auto_create_table=True,
        quote_identifiers=False
    )
    print(f"  HOSPITALS: {nrows} rows loaded ({nchunks} chunks)")


def load_complications(cursor, conn):
    """Load HAI (Hospital Acquired Infections) data into COMPLICATIONS table."""
    df = pd.read_csv(CMS_DIR / "complications.csv")
    df.columns = [c.upper().replace(" ", "_").replace("-", "_") for c in df.columns]

    keep = [
        "FACILITY_ID", "FACILITY_NAME", "STATE", "FISCAL_YEAR",
        "CLABSI_SIR", "CAUTI_SIR", "SSI_SIR", "CDI_SIR", "MRSA_SIR",
        "TOTAL_HAC_SCORE", "PAYMENT_REDUCTION",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    cursor.execute("DROP TABLE IF EXISTS COMPLICATIONS")
    success, nchunks, nrows, _ = write_pandas(
        conn, df, "COMPLICATIONS",
        auto_create_table=True,
        quote_identifiers=False
    )
    print(f"  COMPLICATIONS: {nrows} rows loaded ({nchunks} chunks)")


def create_views(cursor):

    cursor.execute("""
        CREATE OR REPLACE VIEW HOSPITAL_QUALITY_SUMMARY AS
        SELECT
            h.FACILITY_ID,
            h.FACILITY_NAME,
            h.CITYTOWN,
            h.STATE,
            h.HOSPITAL_TYPE,
            h.HOSPITAL_OWNERSHIP,
            h.EMERGENCY_SERVICES,
            TRY_TO_NUMBER(h.HOSPITAL_OVERALL_RATING) AS OVERALL_RATING,
            c.TOTAL_HAC_SCORE,
            c.CLABSI_SIR,
            c.CAUTI_SIR,
            c.MRSA_SIR,
            c.CDI_SIR,
            c.SSI_SIR
        FROM HOSPITALS h
        LEFT JOIN COMPLICATIONS c ON h.FACILITY_ID = c.FACILITY_ID
    """)
    print("  View HOSPITAL_QUALITY_SUMMARY created.")

    cursor.execute("""
        CREATE OR REPLACE VIEW STATE_QUALITY_SUMMARY AS
        SELECT
            STATE,
            COUNT(DISTINCT FACILITY_ID)                       AS NUM_HOSPITALS,
            AVG(TRY_TO_NUMBER(HOSPITAL_OVERALL_RATING))       AS AVG_RATING,
            SUM(CASE WHEN EMERGENCY_SERVICES = 'Yes' THEN 1 ELSE 0 END) AS NUM_WITH_EMERGENCY
        FROM HOSPITALS
        GROUP BY STATE
        ORDER BY AVG_RATING DESC NULLS LAST
    """)
    print("  View STATE_QUALITY_SUMMARY created.")

    cursor.execute("""
        CREATE OR REPLACE VIEW HAI_SUMMARY_BY_STATE AS
        SELECT
            STATE,
            COUNT(*)                        AS NUM_HOSPITALS,
            AVG(TRY_TO_DOUBLE(CLABSI_SIR))  AS AVG_CLABSI,
            AVG(TRY_TO_DOUBLE(CAUTI_SIR))   AS AVG_CAUTI,
            AVG(TRY_TO_DOUBLE(MRSA_SIR))    AS AVG_MRSA,
            AVG(TRY_TO_DOUBLE(CDI_SIR))     AS AVG_CDI,
            AVG(TRY_TO_DOUBLE(SSI_SIR))     AS AVG_SSI
        FROM COMPLICATIONS
        GROUP BY STATE
        ORDER BY AVG_CLABSI DESC NULLS LAST
    """)
    print("  View HAI_SUMMARY_BY_STATE created.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to Snowflake...")
    conn   = get_connection()
    cursor = conn.cursor()

    try:
        setup_database(cursor)
        print("Loading tables...")
        load_hospitals(cursor, conn)
        load_complications(cursor, conn)
        print("Creating analytical views...")
        create_views(cursor)
        print("\nSnowflake load complete.")
        print(f"Connect Power BI to: {SNOWFLAKE_CONFIG['account']} → "
              f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()