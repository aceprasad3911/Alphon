"""
db_utils.py
Database connection helpers. Use psycopg2 for simple transactional work.
"""
import psycopg2
from contextlib import contextmanager
from src.utils.config_utils import load_db_config

@contextmanager
def get_db_connection():
    cfg = load_db_config()
    conn = None
    try:
        conn = psycopg2.connect(
            host=cfg['host'],
            port=cfg['port'],
            dbname=cfg['dbname'],
            user=cfg['user'],
            password=cfg['password']
        )
        yield conn
    finally:
        if conn:
            conn.close()

def test_connect():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT version();")
        v = cur.fetchone()
        cur.close()
        return v
