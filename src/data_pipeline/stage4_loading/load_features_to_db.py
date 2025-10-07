"""
load_features_to_db.py
Load processed feature parquet files into production tables (features, factors).
Implements idempotent upsert pattern via staging tables or ON CONFLICT.
"""
import pandas as pd
from pathlib import Path
from src.utils.db_utils import get_db_connection
from src.utils.logging_utils import setup_logger
from src.utils.helpers import safe_commit

logger = setup_logger("load.features", "reports/pipeline.log")
DATA_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

def upsert_feature_table(df, table_name):
    if df.empty:
        return
    with get_db_connection() as conn:
        cur = conn.cursor()
        for _, row in df.iterrows():
            # Expect asset_id present; customize to your schema
            cur.execute(f"""
            INSERT INTO {table_name} (asset_id, date, feature_name, value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (asset_id, date, feature_name) DO UPDATE SET value = EXCLUDED.value;
            """, (row.get("asset_id"), row.get("date"), "example_feature", row.get("z_ret", 0.0)))
        safe_commit(conn)
        cur.close()

def run():
    files = list(DATA_PROCESSED.glob("*_features.parquet"))
    if not files:
        logger.warning("No processed feature files to load.")
        return
    for f in files:
        df = pd.read_parquet(f)
        upsert_feature_table(df, "time_series_features")
    logger.info("Features loaded to DB.")
