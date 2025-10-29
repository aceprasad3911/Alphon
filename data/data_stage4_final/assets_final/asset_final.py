#!/usr/bin/env python3
# asset_processing_final.py
# Enrich all assets for DB injection with multi-source fallback and checkpointing

import os
import time
import pandas as pd
import yfinance as yf
import requests
import cloudscraper
from bs4 import BeautifulSoup
from pathlib import Path
from src.utils.env_utils import get_env_var

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage3_processed" / "assets_processed" / "all_assets_processed.csv"
OUTPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "assets_final" / "all_assets_final.csv"
FAILED_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "assets_final" / "failed_tickers_final.csv"

SLEEP_YFINANCE = 0.8
SLEEP_ALPHA_VANTAGE = 12
SLEEP_BETWEEN = 2
CHECKPOINT_EVERY = 25

ALPHA_VANTAGE_KEY = get_env_var("API_KEY_ALPHA_VANTAGE")

# Columns to ensure exist
COLUMNS = [
    "ticker", "name", "exchange", "sector", "industry", "country",
    "currency", "isin", "inception_date", "active_status", "source"
]

# -----------------------------
# Helpers
# -----------------------------
def create_scraper():
    return cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})

# -----------------------------
# Yahoo Finance
# -----------------------------
def fetch_yfinance_data(ticker: str):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info:
            return None
        return {
            "name": info.get("longName", ""),
            "exchange": info.get("exchange", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "currency": info.get("currency", ""),
            "isin": info.get("isin", ""),
            "inception_date": info.get("fundInceptionDate", ""),
            "active_status": "Active" if info.get("regularMarketPrice") else "Inactive",
            "source": "YahooFinance"
        }
    except Exception as e:
        print(f"[WARN] Yahoo Finance failed for {ticker}: {e}")
        return None

# -----------------------------
# Alpha Vantage
# -----------------------------
def fetch_alpha_vantage_data(ticker: str):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "Symbol" not in data:
            return None
        return {
            "name": data.get("Name", ""),
            "exchange": data.get("Exchange", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "country": data.get("Country", ""),
            "currency": data.get("Currency", ""),
            "isin": data.get("ISIN", ""),
            "inception_date": data.get("IPODate", ""),
            "active_status": "Active",
            "source": "AlphaVantage"
        }
    except Exception as e:
        print(f"[WARN] Alpha Vantage failed for {ticker}: {e}")
        return None

# -----------------------------
# Finviz
# -----------------------------
def fetch_finviz_data(ticker: str):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    scraper = create_scraper()
    try:
        res = scraper.get(url, timeout=15)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        snapshot_data = {}
        all_tds = [td.text.strip() for td in soup.find_all("td", class_="snapshot-td2")]
        if all_tds:
            snapshot_data = {all_tds[i]: all_tds[i + 1] for i in range(0, len(all_tds)-1, 2)}
        name = ""
        title_table = soup.find("table", class_="fullview-title")
        if title_table and title_table.find("a"):
            name = title_table.find("a").text.strip()
        return {
            "name": name,
            "exchange": snapshot_data.get("Exchange", ""),
            "sector": snapshot_data.get("Sector", ""),
            "industry": snapshot_data.get("Industry", ""),
            "country": snapshot_data.get("Country", ""),
            "currency": snapshot_data.get("Currency", ""),
            "isin": "",
            "inception_date": snapshot_data.get("IPO Date", ""),
            "active_status": "Active" if snapshot_data.get("Price") not in ["", "-", None] else "Inactive",
            "source": "Finviz"
        }
    except Exception as e:
        print(f"[WARN] Finviz failed for {ticker}: {e}")
        return None

# -----------------------------
# Cloudscraper fallback
# -----------------------------
def fetch_cloudscraper_backup(ticker: str):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    scraper = create_scraper()
    try:
        r = scraper.get(url, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        name_tag = soup.find("table", class_="fullview-title")
        name = name_tag.text.strip() if name_tag else ""
        return {
            "name": name,
            "exchange": "",
            "sector": "",
            "industry": "",
            "country": "",
            "currency": "",
            "isin": "",
            "inception_date": "",
            "active_status": "Unknown",
            "source": "Cloudscraper"
        }
    except Exception as e:
        print(f"[WARN] Cloudscraper failed for {ticker}: {e}")
        return None

# -----------------------------
# Enrichment Loop
# -----------------------------
def enrich_all_assets():
    print(f"[START] Loading assets from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Ensure all required columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df[COLUMNS] = df[COLUMNS].astype(object)
    df.fillna("", inplace=True)

    failed_tickers = []
    total = len(df)
    print(f"[INFO] Total tickers in dataset: {total}")

    for idx, row in df.iterrows():
        ticker = str(row['ticker']).strip().upper()
        print(f"\n[PROCESSING] {ticker} ({idx+1}/{total})")
        combined_data = {}

        # -----------------------------
        # 1. Yahoo Finance
        # -----------------------------
        yf_data = fetch_yfinance_data(ticker)
        if yf_data:
            combined_data.update(yf_data)
            print(f"[SUCCESS] {ticker} via Yahoo: {yf_data.get('name','')}")
        else:
            print(f"[WARN] Yahoo unavailable for {ticker}")
        time.sleep(SLEEP_YFINANCE)

        # -----------------------------
        # 2. Alpha Vantage
        # -----------------------------
        if any(combined_data.get(c,"")=="" for c in ["name","sector","industry"]):
            av_data = fetch_alpha_vantage_data(ticker)
            if av_data:
                for k,v in av_data.items():
                    if not combined_data.get(k):
                        combined_data[k] = v
                print(f"[SUCCESS] {ticker} via Alpha Vantage")
            else:
                print(f"[WARN] Alpha Vantage failed for {ticker}")
                failed_tickers.append(ticker)
            time.sleep(SLEEP_ALPHA_VANTAGE)

        # -----------------------------
        # 3. Finviz
        # -----------------------------
        if any(combined_data.get(c,"")=="" for c in ["name","sector","industry"]):
            finviz_data = fetch_finviz_data(ticker)
            if finviz_data:
                for k,v in finviz_data.items():
                    if not combined_data.get(k):
                        combined_data[k] = v
                print(f"[SUCCESS] {ticker} via Finviz")
            else:
                print(f"[WARN] Finviz failed for {ticker}")
            time.sleep(SLEEP_BETWEEN)

        # -----------------------------
        # 4. Cloudscraper fallback
        # -----------------------------
        if not combined_data.get("name"):
            cloud_data = fetch_cloudscraper_backup(ticker)
            if cloud_data:
                combined_data.update(cloud_data)
                print(f"[SUCCESS] {ticker} via Cloudscraper fallback")
            else:
                print(f"[FAILED] No data sources succeeded for {ticker}")
                if ticker not in failed_tickers:
                    failed_tickers.append(ticker)

        # -----------------------------
        # Update dataframe
        # -----------------------------
        for k,v in combined_data.items():
            if v:
                df.at[idx,k] = v

        # -----------------------------
        # Checkpointing
        # -----------------------------
        if (idx+1) % CHECKPOINT_EVERY == 0 or idx == total-1:
            df.to_csv(OUTPUT_CSV,index=False)
            pd.DataFrame(failed_tickers,columns=['ticker']).to_csv(FAILED_CSV,index=False)
            print(f"[CHECKPOINT] Saved {idx+1}/{total} tickers. Failed tickers: {len(failed_tickers)}")

    print(f"\n[DONE] Enrichment completed. Total failed tickers: {len(failed_tickers)}")
    if failed_tickers:
        print(f"[FAILED TICKERS] {failed_tickers}")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    enrich_all_assets()
