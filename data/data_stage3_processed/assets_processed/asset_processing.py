#!/usr/bin/env python3
# asset_processing_full.py

import os
import time
import pandas as pd
import yfinance as yf
import requests
from src.utils.env_utils import get_env_var
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage2_staged" / "assets" / "all_assets.csv"
OUTPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage3_processed" / "assets_processed" / "all_assets_processed.csv"
FAILED_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage3_processed" / "assets_processed" / "failed_tickers.csv"

BATCH_SIZE = 100  # process 100 tickers at a time
SLEEP_YFINANCE = 0.5  # seconds between Yahoo fetches
SLEEP_ALPHA_VANTAGE = 12  # AV free tier rate limit (~5 requests per minute)
USE_ALPHA_VANTAGE = True

# Fetch Alpha Vantage API key from environment
ALPHA_VANTAGE_KEY = get_env_var("API_KEY_ALPHA_VANTAGE")

# Columns to fill
TEXT_COLUMNS = ['name','exchange','sector','industry','country','currency','isin']

# -----------------------------
# Alpha Vantage helpers
# -----------------------------
def fetch_av_overview(ticker: str):
    """
    Fetch company overview from Alpha Vantage.
    Returns a dict with keys matching TEXT_COLUMNS.
    """
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "Symbol" in data:
            return {
                "name": data.get("Name",""),
                "exchange": data.get("Exchange",""),
                "sector": data.get("Sector",""),
                "industry": data.get("Industry",""),
                "country": data.get("Country",""),
                "currency": data.get("Currency",""),
                "isin": data.get("ISIN","")
            }
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Alpha Vantage failed for {ticker}: {e}")
        return None

# -----------------------------
# Main processing
# -----------------------------
def enrich_assets():
    df = pd.read_csv(INPUT_CSV)
    df[TEXT_COLUMNS] = df[TEXT_COLUMNS].astype(object)
    df.fillna('', inplace=True)

    failed_tickers = []

    total = len(df)
    print(f"Tickers needing update: {total}")

    for idx, row in df.iterrows():
        ticker = str(row['ticker']).strip().replace('^','-')  # normalize preferred shares
        updated_data = {}

        # Skip tickers that already have full data
        if all(row[col] != '' for col in TEXT_COLUMNS):
            continue

        # -----------------------------
        # 1. Yahoo Finance fetch
        # -----------------------------
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            if 'longName' in info and info['longName']:
                updated_data = {
                    'name': info.get('longName',''),
                    'exchange': info.get('exchange',''),
                    'sector': info.get('sector',''),
                    'industry': info.get('industry',''),
                    'country': info.get('country',''),
                    'currency': info.get('currency',''),
                    'isin': info.get('isin','')
                }
                print(f"[INFO] Updated {ticker} via Yahoo: {updated_data['name']}")
            else:
                raise ValueError("No Yahoo data")
        except Exception as e:
            print(f"[WARN] Yahoo failed for {ticker}: {e}")

        # -----------------------------
        # 2. Alpha Vantage fallback
        # -----------------------------
        if USE_ALPHA_VANTAGE and (not updated_data or updated_data['name']==''):
            av_data = fetch_av_overview(ticker)
            if av_data:
                updated_data = av_data
                print(f"[INFO] Updated {ticker} via Alpha Vantage: {updated_data['name']}")
            else:
                print(f"[WARN] No data for {ticker} from Alpha Vantage")
                failed_tickers.append(ticker)

            time.sleep(SLEEP_ALPHA_VANTAGE)  # rate limit

        # -----------------------------
        # 3. Update dataframe
        # -----------------------------
        if updated_data:
            for col in TEXT_COLUMNS:
                if updated_data.get(col):
                    df.at[idx, col] = updated_data[col]

        # Sleep between Yahoo requests
        time.sleep(SLEEP_YFINANCE)

        # -----------------------------
        # 4. Checkpointing
        # -----------------------------
        if (idx+1) % 100 == 0 or idx == total-1:
            df.to_csv(OUTPUT_CSV, index=False)
            pd.DataFrame(failed_tickers, columns=['ticker']).to_csv(FAILED_CSV, index=False)
            print(f"[CHECKPOINT] Saved {idx+1}/{total} tickers.")

    print(f"[DONE] Asset enrichment finished. Failed tickers: {len(failed_tickers)}")
    if failed_tickers:
        print(f"[FAILED TICKERS] {failed_tickers}")

if __name__ == "__main__":
    enrich_assets()
