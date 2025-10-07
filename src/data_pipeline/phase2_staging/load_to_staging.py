"""
load_to_staging.py
Reads raw data files (parquet/json) and loads them into staging tables (stg_ prefix)
Perform minimal validation and type normalization.
"""
from pathlib import Path
import pandas as pd
from src.utils.db_utils import get_db_connection
from src.utils.logging_utils import setup_logger
import sqlalchemy

logger = setup_logger("staging", "reports/pipeline.log")

DATA_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
STAGED = Path(__file__).resolve().parents[3] / "data" / "staged"

def _parquet_to_stg_table(parquet_path, table_name):
    df = pd.read_parquet(parquet_path)
    # basic normalization - lower cols, replace spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    with get_db_connection() as conn:
        # Use psycopg2 copy_from or sqlalchemy for convenience; using sqlalchemy here for upsert ease
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{conn.get_dsn_parameters()['user']}@{conn.get_dsn_parameters()['host']}:{conn.get_dsn_parameters()['port']}/{conn.get_dsn_parameters()['dbname']}",
            pool_pre_ping=True
        )
        df.to_sql(table_name, engine, if_exists="append", index=False)
        logger.info(f"Wrote {len(df)} rows to staging table {table_name}")

def run():
    STAGED.mkdir(parents=True, exist_ok=True)
    # Example: scan raw/prices and stage into stg_prices
    prices_dir = DATA_RAW / "prices"
    if prices_dir.exists():
        for fn in prices_dir.glob("*.parquet"):
            logger.info(f"Staging {fn.name} -> stg_prices")
            _parquet_to_stg_table(fn, "stg_prices")
    else:
        logger.warning("No raw prices found to stage.")
    logger.info("Staging complete.")
