"""
test_feature_integrity.py
Check that derived features fall into expected ranges (sanity).
"""
import pytest
from src.utils.db_utils import get_db_connection

def test_volatility_non_negative():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM time_series_features WHERE value < 0 AND feature_name = 'volatility';")
        assert cur.fetchone()[0] == 0, "Negative vol values found"

def test_rsi_bounds():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM technical_indicators WHERE indicator_name='rsi' AND (value < 0 OR value > 100);")
        assert cur.fetchone()[0] == 0, "RSI out of bounds"
