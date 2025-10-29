#!/usr/bin/env python3
"""
assets_enriched_ipo_date_v3.py

Enrich 'inception_date' (IPO / first trade date) for every row in the final assets CSV.
Strategy (fast -> slow):
 1) Yahoo Finance history (first trade date) — parallelized
 2) Alpha Vantage OVERVIEW (IPODate) — sequential, rate-limited (only for tickers Yahoo couldn't)
 3) Optional Finviz (disabled by default) — fragile; only try if you explicitly enable it.

Writes checkpoints to OUTPUT_CSV and a failed list to FAILED_CSV.
"""

import time
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.env_utils import get_env_var

# -----------------------
# CONFIG
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "assets_final" / "all_assets_final.csv"
OUTPUT_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "assets_final" / "all_assets_final_enriched.csv"
FAILED_CSV = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage4_final" / "assets_final" / "failed_inception_dates.csv"

# Alpha Vantage config
ALPHA_KEY = get_env_var("API_KEY_ALPHA_VANTAGE")
ALPHA_SLEEP = 12       # recommended free-tier delay; adjust if you have premium
USE_ALPHA_VANTAGE = True

# Finviz config
USE_FINVIZ = False     # set True only if you want to try scraping (may be slow/fragile)
FINVIZ_TIMEOUT = 6

# Parallelization
YF_WORKERS = 10        # number of threads for Yahoo concurrent history fetches

# Checkpointing
CHECKPOINT_EVERY = 200  # save CSV + failed list every N tickers

# -----------------------
# HELPERS
# -----------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV robustly; if default engine fails retry with python engine and skip bad lines."""
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
# Data fetchers
# -----------------------
def fetch_first_trade_date_yahoo(ticker: str):
    """
    Use yfinance history to estimate the inception date: first available trade date (index min).
    Returns ISO date string 'YYYY-MM-DD' or None.
    """
    try:
        t = yf.Ticker(ticker)
        # fetch only index info: period max but we request minimal columns by using 'actions=False'
        hist = t.history(period="max", auto_adjust=False, actions=False)
        if hist is None or hist.empty:
            return None
        # index may be timezone-aware; convert to date
        first_ts = hist.index.min()
        # Pandas Timestamp -> date string
        return pd.to_datetime(first_ts).strftime("%Y-%m-%d")
    except Exception:
        return None

def fetch_ipo_date_alpha(ticker: str):
    """
    Query Alpha Vantage OVERVIEW for IPODate field.
    Returns ISO date string 'YYYY-MM-DD' or None.
    """
    if not ALPHA_KEY:
        return None
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_KEY}"
    try:
        r = requests.get(url, timeout=15)
        j = r.json()
        # 'IPODate' sometimes present. Try common keys.
        ipo = j.get("IPODate") or j.get("IPO Date") or j.get("ipoDate")
        if ipo and isinstance(ipo, str) and ipo.strip():
            # AV usually returns 'YYYY-MM-DD' or 'YYYY-MM-DD' - leave as-is
            return ipo.strip()
    except Exception as e:
        # network or JSON decode problem
        print(f"[WARN] Alpha Vantage call error for {ticker}: {e}")
    return None

def fetch_ipo_date_finviz(ticker: str):
    """
    Optional: attempt to scrape Finviz for IPO Date.
    This is fragile, may fail, and may be blocked; used only if explicitly enabled.
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
        # Finviz snapshot table: labels & values often in pairs under class snapshot-td2
        cells = [td.text.strip() for td in soup.find_all("td", class_="snapshot-td2")]
        # build dictionary from pairs if possible
        if len(cells) >= 2:
            pairs = {cells[i]: cells[i+1] for i in range(0, len(cells)-1, 2)}
            # try several possible keys
            for key in ("IPO Date", "IPO", "IPOdate", "IPODate"):
                if key in pairs and pairs[key].strip():
                    return pairs[key].strip()
        # fallback: try finding 'IPO Date' text elsewhere
        return None
    except Exception as e:
        # avoid spamming errors — just return None
        print(f"[WARN] Finviz fetch error for {ticker}: {e}")
        return None

# -----------------------
# MAIN ENRICHMENT
# -----------------------
def enrich_all_inception_dates():
    df = safe_read_csv(INPUT_CSV)

    # Ensure target columns exist
    if "inception_date" not in df.columns:
        df["inception_date"] = ""
    if "active_status" not in df.columns:
        df["active_status"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""

    total = len(df)
    print(f"[INFO] Beginning enrichment for {total} assets.")

    # Prepare tickers list (preserve order to write back safely)
    tickers = df["ticker"].astype(str).fillna("").str.strip().str.upper().tolist()

    # Stage 1: parallel Yahoo first-trade fetch (fast for many tickers)
    print("[STAGE 1] Running parallel Yahoo history to get first-trade dates (fast).")
    yahoo_results = {}  # ticker -> date or None
    with ThreadPoolExecutor(max_workers=YF_WORKERS) as ex:
        futures = {ex.submit(fetch_first_trade_date_yahoo, t): idx for idx, t in enumerate(tickers)}
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            ticker = tickers[idx]
            try:
                val = fut.result()
                yahoo_results[ticker] = val
                status = f"OK -> {val}" if val else "no data"
                print(f"[YF] {ticker}: {status}")
            except Exception as e:
                yahoo_results[ticker] = None
                print(f"[YF] {ticker}: ERROR {e}")
            completed += 1
            if completed % 100 == 0:
                print(f"[YF] Completed {completed}/{total}")

    # Apply Yahoo results to dataframe
    for i, ticker in enumerate(tickers):
        ydate = yahoo_results.get(ticker)
        if ydate:
            df.at[i, "inception_date"] = ydate
            df.at[i, "active_status"] = True
        else:
            # leave blank for now; will try AV/Finviz
            pass

    # Stage 2: Alpha Vantage for remaining (rate-limited)
    if USE_ALPHA_VANTAGE and ALPHA_KEY:
        print("[STAGE 2] Alpha Vantage fallback for tickers Yahoo couldn't resolve (rate-limited).")
        failed = []  # collect tickers that still don't have dates after AV+Finviz
        remaining_indices = [i for i, t in enumerate(tickers) if not df.at[i, "inception_date"]]
        print(f"[AV] Need to query Alpha Vantage for {len(remaining_indices)} tickers.")
        for count, i in enumerate(remaining_indices, start=1):
            ticker = tickers[i]
            # AV call
            ipo = fetch_ipo_date_alpha(ticker)
            if ipo:
                df.at[i, "inception_date"] = ipo
                df.at[i, "active_status"] = True
                print(f"[AV] {ticker}: IPO -> {ipo}")
            else:
                # Stage 3 optional Finviz (only if enabled)
                ipo_fv = fetch_ipo_date_finviz(ticker) if USE_FINVIZ else None
                if ipo_fv:
                    df.at[i, "inception_date"] = ipo_fv
                    df.at[i, "active_status"] = True
                    print(f"[FINVIZ] {ticker}: IPO -> {ipo_fv}")
                else:
                    # mark as missing for now — will be written to failed list
                    df.at[i, "active_status"] = False
                    df.at[i, "notes"] = df.at[i, "notes"] if pd.notna(df.at[i, "notes"]) else ""
                    # append note only if empty
                    if not df.at[i, "notes"]:
                        df.at[i, "notes"] = "inception_date missing after YF+AV"
                    failed.append(ticker)
                    print(f"[AV] {ticker}: not found")
            # respect AV rate limit
            time.sleep(ALPHA_SLEEP)
            # checkpoint periodically
            if count % CHECKPOINT_EVERY == 0:
                save_checkpoint(df, failed, count, len(remaining_indices))

        # final write of failed list after AV stage
        pd.DataFrame({"ticker": failed}).to_csv(FAILED_CSV, index=False)
        print(f"[AV] Alpha Vantage stage finished. {len(failed)} tickers still missing.")
    else:
        print("[INFO] Skipping Alpha Vantage (no key or disabled). Marking unresolved as inactive.")
        unresolved = []
        for i, ticker in enumerate(tickers):
            if not df.at[i, "inception_date"]:
                df.at[i, "active_status"] = False
                df.at[i, "notes"] = "inception_date missing; AV skipped"
                unresolved.append(ticker)
        pd.DataFrame({"ticker": unresolved}).to_csv(FAILED_CSV, index=False)
        print(f"[INFO] Marked {len(unresolved)} tickers as unresolved.")

    # Final checkpoint / write
    save_checkpoint(df, pd.read_csv(FAILED_CSV)["ticker"].tolist() if Path(FAILED_CSV).exists() else [], total, total)
    print("[DONE] Enrichment finished. Final CSV written to:", OUTPUT_CSV)

if __name__ == "__main__":
    start = time.time()
    enrich_all_inception_dates()
    print(f"[COMPLETE] Time elapsed: {(time.time()-start)/60:.2f} minutes")
