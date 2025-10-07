"""
helpers.py
Small helpers used across the pipeline.
"""
from datetime import datetime

def today_iso():
    return datetime.utcnow().date().isoformat()

def safe_commit(conn):
    try:
        conn.commit()
    except Exception:
        conn.rollback()
        raise
