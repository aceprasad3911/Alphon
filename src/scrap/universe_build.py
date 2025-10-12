import time
import pandas as pd
import yfinance as yf
import os
from numpy.random import uniform

# ========= CONFIG ==========
sector_filter = None   # e.g., 'Financials' or 'Information Technology'; use None to include all
max_assets = 1000      # Limit universe size to manage compute
volume_threshold = 500_000
missing_data_threshold = 0.1
lookback_days = 252    # ~1 year of trading days
retry_limit = 2
delay_range = (0.5, 1.5)  # Random sleep between API calls
# ===========================

# Ensure output directory exists
os.makedirs("../../data/asset_universe", exist_ok=True)

# STEP 1: Scrape the S&P 500 tickers from Wikipedia using pandas
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url, header=0)
    df = tables[0]
    # Clean tickers: Yahoo uses '-' instead of '.'
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
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

# STEP 4: Filter illiquid or incomplete tickers
def filter_liquid_stocks(df, save_path):
    valid_tickers = []
    failed_tickers = []

    for i, symbol in enumerate(df['Symbol']):
        success = False
        for attempt in range(retry_limit):
            try:
                time.sleep(uniform(*delay_range))
                data = yf.download(symbol, period=f"{lookback_days}d", interval="1d", progress=False)

                if data.empty:
                    print(f"[!] {symbol} - No data returned from Yahoo Finance")
                    raise ValueError("Insufficient data")

                if len(data) < lookback_days * (1 - missing_data_threshold):
                    raise ValueError("Too few trading days")

                missing_ratio = data.isna().sum().sum() / (data.shape[0] * data.shape[1])
                avg_volume = data['Volume'].mean()

                if missing_ratio <= missing_data_threshold and avg_volume >= volume_threshold:
                    valid_tickers.append(symbol)

                success = True
                break

            except Exception as e:
                if attempt == retry_limit - 1:
                    print(f"[!] Failed to get ticker '{symbol}' after {retry_limit} attempts: {e}")
                    failed_tickers.append(symbol)
                else:
                    time.sleep(1)

        if i % 25 == 0 and i > 0:
            print(f"[•] Checked {i}/{len(df)} tickers... {len(valid_tickers)} passed")

    # Save filtered tickers
    filtered_df = df[df['Symbol'].isin(valid_tickers)]
    filtered_df.to_csv(save_path, index=False)
    print(f"[✓] Filtered asset universe saved to {save_path} ({len(filtered_df)} tickers)")

    # Save failed tickers separately
    if failed_tickers:
        failed_df = df[df['Symbol'].isin(failed_tickers)]
        failed_df.to_csv("assets_failed.csv", index=False)
        print(f"[!] {len(failed_tickers)} tickers failed and saved to assets_failed.csv")

# Execute full pipeline
if __name__ == "__main__":
    try:
        raw_df = get_sp500_tickers()
        universe_df = select_sector_universe(raw_df, sector=sector_filter, max_assets=max_assets)
        save_universe(universe_df, "assets.csv")
        filter_liquid_stocks(universe_df, "assets_cleaned.csv")

    except KeyboardInterrupt:
        print("\n[✘] Script interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] {e}")
