import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# ===============================
# CONFIG
# ===============================
SCRIPT_DIR = Path(__file__).resolve().parent
FAILED_TICKERS_FILE = SCRIPT_DIR / ".." / ".." / "data_stage3_processed" / "assets_processed" / "failed_tickers.csv"
OUTPUT_FILE = SCRIPT_DIR / ".." / ".." / "data_stage3_processed" / "assets_processed" / "finviz_enriched.csv"

FAILED_TICKERS_FILE = FAILED_TICKERS_FILE.resolve()
OUTPUT_FILE = OUTPUT_FILE.resolve()

# ===============================
# FINVIZ SCRAPER
# ===============================
def finviz_scrape_metadata(ticker: str):
    """
    Scrape company metadata from Finviz.
    Returns dict with keys matching your schema or None if not found.
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        name_tag = soup.select_one(".fullview-title b")
        name = name_tag.text.strip() if name_tag else ""

        table = soup.select_one(".snapshot-table2")
        if not table:
            return None
        tds = [td.get_text(strip=True) for td in table.find_all("td")]
        data = dict(zip(tds[::2], tds[1::2]))

        return {
            "ticker": ticker,
            "name": name,
            "exchange": data.get("Exchange", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "country": data.get("Country", ""),
            "currency": "USD" if data.get("Country", "") == "USA" else "",
            "isin": "",
            "source": "FINVIZ"
        }

    except Exception:
        return None


# ===============================
# MAIN FALLBACK ROUTINE
# ===============================
def finviz_enrich_failed():
    if not FAILED_TICKERS_FILE.exists():
        raise FileNotFoundError(f"❌ Missing {FAILED_TICKERS_FILE}")

    failed_df = pd.read_csv(FAILED_TICKERS_FILE)
    if "ticker" not in failed_df.columns:
        raise ValueError("❌ failed_tickers.csv must contain a 'ticker' column")

    tickers = failed_df["ticker"].astype(str).str.strip().str.upper().unique().tolist()
    print(f"🔍 Found {len(tickers)} failed tickers to enrich from Finviz...")

    enriched_records = []
    for i, tkr in enumerate(tickers, 1):
        meta = finviz_scrape_metadata(tkr)
        if meta:
            enriched_records.append(meta)
            print(f"[{i}/{len(tickers)}] ✅ Enriched {tkr} via Finviz")
        else:
            print(f"[{i}/{len(tickers)}] ⚠️ No Finviz data for {tkr}")
        time.sleep(0.5)  # polite delay to avoid rate-limit

    if not enriched_records:
        print("⚠️ No tickers were enriched from Finviz.")
        return

    enriched_df = pd.DataFrame(enriched_records)
    enriched_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(enriched_df)} enriched records → {OUTPUT_FILE}")


if __name__ == "__main__":
    finviz_enrich_failed()
