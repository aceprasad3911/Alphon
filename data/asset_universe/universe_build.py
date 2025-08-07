import pandas as pd
import yfinance as yf
import os

# ========= CONFIG ==========
sector_filter = None   # <-- Set to e.g. 'Financials' or 'Information Technology'
# Set to None to include all sectors
max_assets = 1000               # Optional cap for computational manageability
# ===========================

# Ensure output directory exists
os.makedirs("../data", exist_ok=True)

# STEP 1: Scrape the S&P 500 tickers from Wikipedia using pandas
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url, header=0)
    df = tables[0]
    return df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

# STEP 2: Filter by sector if specified
def select_sector_universe(df, sector=None, max_assets=500):
    if sector:
        filtered_df = df[df['GICS Sector'] == sector]
    else:
        filtered_df = df
    return filtered_df.head(max_assets).reset_index(drop=True)

# STEP 3: Save to CSV
def save_universe(df, path='assets.csv'):
    df.to_csv(path, index=False)
    print(f"[✓] Asset universe saved to {path} ({len(df)} companies)")

# Execute steps
if __name__ == "__main__":
    try:
        raw_df = get_sp500_tickers()
        universe_df = select_sector_universe(raw_df, sector=sector_filter, max_assets=max_assets)
        save_universe(universe_df)
    except Exception as e:
        print(f"[ERROR] {e}")
