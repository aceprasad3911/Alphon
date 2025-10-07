"""
fetch_macro.py
Fetch macroeconomic time series (FRED or local CSV). Stores to data/raw/macro.
"""
from pathlib import Path
from src.utils.logging_utils import setup_logger
import pandas as pd
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "macro"
logger = setup_logger("ingest.macro", "reports/pipeline.log")

def run():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # Example synthetic CPI series
    dates = pd.date_range("2018-01-01", "2025-01-01", freq="M")
    series = pd.Series(100 + (np.random.rand(len(dates)) - 0.5).cumsum(), index=dates)
    df = series.reset_index().rename(columns={"index": "date", 0: "cpi"})
    df.to_parquet(DATA_ROOT / "cpi.parquet")
    logger.info("Macro synthetic data written to raw/macro.")
