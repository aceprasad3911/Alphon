"""
timeseries_features.py
Additional time-series transformations: seasonality, lags, autocorrelation, etc.
"""
from pathlib import Path
import pandas as pd
from statsmodels.tsa.stattools import acf
from src.utils.logging_utils import setup_logger

logger = setup_logger("transform.ts", "reports/pipeline.log")
DATA_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

def compute_autocorr(series, nlags=10):
    return acf(series.dropna(), nlags=nlags, fft=False)

def run():
    files = list(DATA_PROCESSED.glob("*features.parquet"))
    if not files:
        logger.warning("No features to augment with time-series features.")
        return
    for f in files:
        df = pd.read_parquet(f)
        # example: compute autocorr of returns
        if "return_1d" in df.columns:
            ac = compute_autocorr(df["return_1d"])
            logger.info(f"{f.name} autocorr[1:3] = {ac[1:3]}")
    logger.info("Time-series feature augmentation complete.")
