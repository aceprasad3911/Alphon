"""
feature_engineering.py
Generate core cross-sectional & time-series features:
- returns
- moving averages
- volatility
- z-scores / ranks
Outputs processed parquet files in data/processed and staging DB tables for validation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.logging_utils import setup_logger

logger = setup_logger("transform.features", "reports/pipeline.log")
DATA_STAGED = Path(__file__).resolve().parents[3] / "data" / "staged"
DATA_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

def compute_features(df):
    df = df.sort_values("date")
    df["return_1d"] = df["close"].pct_change()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["vol_20"] = df["return_1d"].rolling(20).std()
    df["z_ret"] = (df["return_1d"] - df["return_1d"].mean()) / (df["return_1d"].std() + 1e-9)
    return df

def run():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    prices_parquets = list((DATA_STAGED).glob("*.parquet"))
    if not prices_parquets:
        logger.warning("No staged data found for feature engineering.")
        return
    for p in prices_parquets:
        logger.info(f"Processing {p.name}")
        df = pd.read_parquet(p)
        # Expect df has date, close, etc.
        df_feat = compute_features(df)
        out = DATA_PROCESSED / p.name.replace(".parquet", "_features.parquet")
        df_feat.to_parquet(out)
        logger.info(f"Wrote features to {out}")
    logger.info("Feature engineering complete.")
