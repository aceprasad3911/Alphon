"""
test_data_quality.py
Basic data quality tests: no duplicate (asset_id,date), no NaNs in required cols.
"""
import pytest
from src.utils.db_utils import get_db_connection

def has_nulls(conn, table, column):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL;")
        return cur.fetchone()[0] > 0

def has_duplicate_asset_date(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT asset_id, date, COUNT(*) as c
                FROM {table}
                GROUP BY asset_id, date
                HAVING COUNT(*) > 1
            ) s;
        """)
        return cur.fetchone()[0] > 0

def test_no_null_prices_close():
    with get_db_connection() as conn:
        assert not has_nulls(conn, "price_data", "close"), "Nulls in price_data.close"

def test_price_asset_date_uniqueness():
    with get_db_connection() as conn:
        assert not has_duplicate_asset_date(conn, "price_data"), "Duplicate (asset,date) in price_data"
