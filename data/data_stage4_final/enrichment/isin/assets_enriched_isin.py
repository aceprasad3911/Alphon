#!/usr/bin/env python3
"""
assets_enriched_isin_v3.py

Enrich 'isin' for every row in the final assets CSV.
Strategy (fast -> slow):
 1) Alpha Vantage SYMBOL_SEARCH (primary) — sequential, rate-limited
 2) OpenFIGI mapping API (fallback)
 3) Optional Finviz scraping (disabled by default; fragile)

Writes checkpoints to OUTPUT_CSV and a failed list to FAILED_CSV.
"""

import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.env_utils import get_env_var

# -----------------------
# CONFIG
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "enrichment" / "isin" / "all_assets_final.csv"
OUTPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "enrichment" / "isin" / "all_assets_final_enriched.csv"
FAILED_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "enrichment" / "isin" / "failed_inception_dates.csv"

# Alpha Vantage config
ALPHA_KEY = get_env_var("API_KEY_ALPHA_VANTAGE")
ALPHA_SLEEP = 12       # recommended free-tier delay
USE_ALPHA_VANTAGE = True

# OpenFIGI fallback
USE_OPENFIGI = True

# Finviz config
USE_FINVIZ = False     # set True only if you want to try scraping (may be slow/fragile)
FINVIZ_TIMEOUT = 6

# Parallelization
AV_WORKERS = 1  # sequential due to AlphaVantage rate limit

# Checkpointing
CHECKPOINT_EVERY = 200  # save CSV + failed list every N tickers

# -----------------------
# HELPERS
# -----------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV robustly; retry with python engine if default fails."""
    print(f"[START] Loading dataset: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] CSV parse issue: {e}. Retrying with python engine and skipping bad lines.")
        df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    print(f"[INFO] Loaded {len(df)} rows.")
    return df


def save_checkpoint(df: pd.DataFrame, failed: list, idx_processed: int, total: int):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame({"ticker": failed}).to_csv(FAILED_CSV, index=False)
    print(f"[CHECKPOINT] Saved progress {idx_processed}/{total}. Failed so far: {len(failed)}")

# -----------------------
# DATA FETCHERS
# -----------------------
def fetch_isin_alpha_vantage(ticker: str):
    """
    Query Alpha Vantage SYMBOL_SEARCH for ISIN.
    Returns ISIN string or None.
    """
    if not ALPHA_KEY:
        return None
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={ticker}&apikey={ALPHA_KEY}"
    try:
        r = requests.get(url, timeout=15)
        j = r.json()
        matches = j.get("bestMatches", [])
        if matches:
            for m in matches:
                isin = m.get("8. isin") or m.get("ISIN")
                if isin and isinstance(isin, str):
                    return isin.strip()
    except Exception as e:
        print(f"[WARN] Alpha Vantage ISIN error for {ticker}: {e}")
    return None


def fetch_isin_openfigi(ticker: str):
    """
    Fallback: query OpenFIGI mapping API.
    Returns ISIN or FIGI if available.
    """
    if not USE_OPENFIGI:
        return None
    try:
        url = "https://api.openfigi.com/v3/mapping"
        payload = [{"idType": "TICKER", "idValue": ticker}]
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and "data" in data[0]:
                rec = data[0]["data"][0]
                return rec.get("isin") or rec.get("figi")
    except Exception as e:
        print(f"[WARN] OpenFIGI error for {ticker}: {e}")
    return None


def fetch_isin_finviz(ticker: str):
    """
    Optional: attempt to scrape ISIN-like info from Finviz (not reliable).
    """
    if not USE_FINVIZ:
        return None
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
        scraper = cloudscraper.create_scraper()
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        res = scraper.get(url, timeout=FINVIZ_TIMEOUT)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        cells = [td.text.strip() for td in soup.find_all("td", class_="snapshot-td2")]
        if len(cells) >= 2:
            pairs = {cells[i]: cells[i+1] for i in range(0, len(cells)-1, 2)}
            for key in ("ISIN", "FIGI", "Cusip"):
                if key in pairs and pairs[key].strip():
                    return pairs[key].strip()
        return None
    except Exception as e:
        print(f"[WARN] Finviz ISIN fetch error for {ticker}: {e}")
        return None


# -----------------------
# MAIN ENRICHMENT
# -----------------------
def enrich_all_isin():
    df = safe_read_csv(INPUT_CSV)

    # Ensure all columns exist for DB format
    required_cols = [
        "ticker", "name", "exchange", "sp500_status", "sector",
        "country", "currency", "isin", "source", "inception_date",
        "active_status", "notes"
    ]
    for col in required_cols:
        if col not in df.columns:
            if col == "sp500_status":
                df[col] = False
            elif col == "active_status":
                df[col] = True
            else:
                df[col] = ""

    total = len(df)
    print(f"[INFO] Beginning ISIN enrichment for {total} assets.")
    tickers = df["ticker"].astype(str).fillna("").str.strip().str.upper().tolist()

    failed = []
    updated = 0

    # Sequential loop (AV + fallback + optional Finviz)
    for count, ticker in enumerate(tickers, start=1):
        if not ticker:
            continue

        current_isin = str(df.at[count - 1, "isin"]).strip()
        if current_isin:
            continue  # skip already enriched rows

        isin = fetch_isin_alpha_vantage(ticker)
        if not isin:
            isin = fetch_isin_openfigi(ticker)
        if not isin:
            isin = fetch_isin_finviz(ticker) if USE_FINVIZ else None

        if isin:
            df.at[count - 1, "isin"] = isin
            updated += 1
            print(f"[OK] {ticker}: ISIN -> {isin}")
        else:
            df.at[count - 1, "notes"] = "ISIN not found"
            failed.append(ticker)
            print(f"[WARN] {ticker}: ISIN not found")

        # Rate-limit for Alpha Vantage
        time.sleep(ALPHA_SLEEP)

        if count % CHECKPOINT_EVERY == 0:
            save_checkpoint(df, failed, count, total)

    # Final save
    save_checkpoint(df, failed, total, total)
    print(f"[DONE] ISIN enrichment complete. {updated} updated, {len(failed)} missing.")


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    start = time.time()
    enrich_all_isin()
    print(f"[COMPLETE] Time elapsed: {(time.time() - start) / 60:.2f} minutes")
