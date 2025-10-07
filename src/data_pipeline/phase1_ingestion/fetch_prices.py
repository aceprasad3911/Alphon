"""
fetch_prices.py
Fetch historical OHLCV data (example using yfinance).
Saves raw CSVs to data/raw/prices/<date> and/or writes to raw_data_cache table.
"""
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from src.utils.config_utils import load_data_sources
from src.utils.logging_utils import setup_logger

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "prices"

logger = setup_logger("ingest.prices", "reports/pipeline.log")

def fetch_ticker_history(ticker, start=None, end=None):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
    return df

def save_raw(df, ticker):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_ROOT / f"{ticker}.parquet"
    df.to_parquet(out_path)
    logger.info(f"Saved raw prices for {ticker} to {out_path}")

def run(tickers=None, start=None, end=None):
    cfg = load_data_sources().get("prices", {})
    start = start or cfg.get("default_start")
    end = end or cfg.get("default_end")
    if not tickers:
        # Simple default list - replace with config CSV read if required
        tickers = ["AAPL", "MSFT", "GOOG"]
    logger.info(f"Starting price ingestion for {len(tickers)} tickers from {start} to {end}")
    for t in tickers:
        try:
            df = fetch_ticker_history(t, start, end)
            if not df.empty:
                save_raw(df, t)
        except Exception as e:
            logger.exception(f"Failed fetching {t}: {e}")
    logger.info("Price ingestion completed.")
