"""
test_schema_consistency.py
Pytest tests to check required tables/columns exist in production DB.
"""
import pytest
from src.utils.db_utils import get_db_connection

REQUIRED_TABLES = {
    "assets": ["asset_id", "ticker"],
    "price_data": ["asset_id", "date", "close"],
    "fundamentals": ["asset_id", "date", "key", "value"]
}

def table_columns(conn, table):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s;
        """, (table,))
        return [r[0] for r in cur.fetchall()]

def test_required_tables_exist():
    with get_db_connection() as conn:
        cur = conn.cursor()
        for t in REQUIRED_TABLES.keys():
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema='public' AND table_name=%s
                );
            """, (t,))
            assert cur.fetchone()[0], f"Table {t} is missing"

def test_columns_for_tables():
    with get_db_connection() as conn:
        for t, cols in REQUIRED_TABLES.items():
            existing = table_columns(conn, t)
            for c in cols:
                assert c in existing, f"Column {c} missing from {t}"
