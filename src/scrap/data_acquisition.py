import yfinance as yf
import pandas as pd
from fredapi import Fred
import os
from datetime import datetime

# --- Configuration ---
# Define your asset universe (e.g., S&P 500 components)
# For a real project, you'd fetch this dynamically or from a curated list.
# For demonstration, let's use a small sample.
SP500_SAMPLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'PG', 'UNH', 'NVDA']

# Define date range
START_DATE = '2018-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d') # Up to today

# FRED API Key (Get yours from https://fred.stlouisfed.org/docs/api/api_key.html)
# It's best practice to store API keys securely (e.g., environment variables)
FRED_API_KEY = 'YOUR_FRED_API_KEY' #TODO REPLACE WITH YOUR ACTUAL FRED API KEY

# Output directories
RAW_DATA_DIR = 'data/raw'
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# --- Data Acquisition Functions ---

def download_stock_data(tickers, start_date, end_date, output_dir):
    """Downloads historical stock OHLCV and fundamental data using yfinance."""
    print(f"Downloading stock data for {len(tickers)} tickers from {start_date} to {end_date}...")
    all_stock_data = {}
    for ticker in tickers:
        try:
            # Download daily OHLCV and adjusted close
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                # Add ticker as a column
                data['Ticker'] = ticker
                all_stock_data[ticker] = data
                print(f"  Successfully downloaded {ticker}")
            else:
                print(f"  No data found for {ticker}")

            # You can also try to get some fundamental data, though yfinance's direct fundamental
            # access can be limited and often requires parsing info() or actions()
            # For more robust fundamental data, you'd integrate Alpha Vantage or EDGAR API here.
            # Example: info = yf.Ticker(ticker).info
            # print(f"  {ticker} Info: {info.get('marketCap')}")

        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")

    # Concatenate all stock data into a single DataFrame
    if all_stock_data:
        df_stocks = pd.concat(all_stock_data.values())
        df_stocks.index.name = 'Date'
        # Save to Parquet for efficiency
        output_path = os.path.join(output_dir, 'raw_stock_data.parquet')
        df_stocks.to_parquet(output_path)
        print(f"All raw stock data saved to {output_path}")
        return df_stocks
    else:
        print("No stock data downloaded.")
        return pd.DataFrame()

def download_macro_data(api_key, start_date, end_date, output_dir):
    """Downloads macroeconomic data from FRED."""
    print(f"Downloading macroeconomic data from FRED from {start_date} to {end_date}...")
    fred = Fred(api_key=api_key)

    # Common FRED series IDs (you can add more)
    fred_series = {
        'GDP': 'GDP',
        'Inflation_CPI': 'CPIAUCSL', # Consumer Price Index for All Urban Consumers
        'Federal_Funds_Rate': 'FEDFUNDS',
        'Unemployment_Rate': 'UNRATE',
        '10_Year_Treasury_Yield': 'DGS10'
    }

    all_macro_data = {}
    for name, series_id in fred_series.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            if data is not None and not data.empty:
                all_macro_data[name] = data.rename(name)
                print(f"  Successfully downloaded {name} ({series_id})")
            else:
                print(f"  No data found for {name} ({series_id})")
        except Exception as e:
            print(f"  Error downloading {name} ({series_id}): {e}")

    if all_macro_data:
        # Combine into a single DataFrame, forward-fill to align dates later
        df_macro = pd.DataFrame(all_macro_data)
        df_macro.index.name = 'Date'
        output_path = os.path.join(output_dir, 'raw_macro_data.parquet')
        df_macro.to_parquet(output_path)
        print(f"All raw macro data saved to {output_path}")
        return df_macro
    else:
        print("No macro data downloaded.")
        return pd.DataFrame()

if __name__ == "__main__":
    print("--- Starting Data Acquisition Phase ---")
    # Download stock data
    stock_df = download_stock_data(SP500_SAMPLE_TICKERS, START_DATE, END_DATE, RAW_DATA_DIR)

    # Download macroeconomic data
    macro_df = download_macro_data(FRED_API_KEY, START_DATE, END_DATE, RAW_DATA_DIR)

    print("\nData Acquisition Phase Complete.")
    print(f"Raw stock data shape: {stock_df.shape}")
    print(f"Raw macro data shape: {macro_df.shape}")
