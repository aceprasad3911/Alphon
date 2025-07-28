import pandas as pd
import numpy as np
import os

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Data Cleaning and Alignment Functions ---

def load_raw_data(data_dir):
    """Loads raw stock and macro data from parquet files."""
    stock_path = os.path.join(data_dir, 'raw_stock_data.parquet')
    macro_path = os.path.join(data_dir, 'raw_macro_data.parquet')

    stock_df = pd.read_parquet(stock_path) if os.path.exists(stock_path) else pd.DataFrame()
    macro_df = pd.read_parquet(macro_path) if os.path.exists(macro_path) else pd.DataFrame()

    return stock_df, macro_df

def clean_stock_data(df_stock):
    """Performs basic cleaning on stock data."""
    print("Cleaning stock data...")
    if df_stock.empty:
        print("  Stock DataFrame is empty, skipping cleaning.")
        return df_stock

    # Ensure Date is datetime index
    df_stock.index = pd.to_datetime(df_stock.index)

    # Handle missing values (e.g., forward fill then back fill for OHLCV)
    # Note: Adjusted Close ('Adj Close') is usually already adjusted for splits/dividends by yfinance
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df_stock.columns:
            df_stock[col] = df_stock.groupby('Ticker')[col].transform(lambda x: x.ffill().bfill())

    # Drop rows where 'Adj Close' is still NaN (e.g., if a ticker had no data at all)
    df_stock.dropna(subset=['Adj Close'], inplace=True)

    # Basic outlier detection (e.g., capping extreme volumes, though often not needed for daily data)
    # For simplicity, we'll skip complex outlier handling here, but it's a point for refinement.

    print(f"  Cleaned stock data shape: {df_stock.shape}")
    return df_stock

def align_dataframes(df_stock, df_macro, target_frequency='D'):
    """Aligns stock and macro data to a common frequency."""
    print(f"Aligning dataframes to {target_frequency} frequency...")
    if df_stock.empty:
        print("  Stock DataFrame is empty, cannot align.")
        return pd.DataFrame()
    if df_macro.empty:
        print("  Macro DataFrame is empty, cannot align.")
        return df_stock # Return stock data as is if no macro to merge

    # Ensure Date is datetime index for both
    df_stock.index = pd.to_datetime(df_stock.index)
    df_macro.index = pd.to_datetime(df_macro.index)

    # Create a full date range for alignment
    min_date = max(df_stock.index.min(), df_macro.index.min())
    max_date = min(df_stock.index.max(), df_macro.index.max())
    full_date_range = pd.date_range(start=min_date, end=max_date, freq=target_frequency)

    # Reindex macro data to the full date range and forward-fill
    # Macro data is typically lower frequency and should be forward-filled to daily stock dates
    df_macro_aligned = df_macro.reindex(full_date_range).ffill()

    # Merge stock data with aligned macro data
    # We need to merge for each ticker. This requires unstacking/stacking or a clever merge.
    # A more robust way is to iterate or use a multi-index merge.
    # Let's pivot stock data to make merging easier, then merge macro, then unpivot.
    # Or, simpler: merge macro data onto each stock's daily data.

    # Reset index to merge on Date and Ticker
    df_stock_reset = df_stock.reset_index()

    # Merge macro data onto stock data based on Date
    # Use a left merge to keep all stock data and add macro columns
    df_combined = pd.merge(df_stock_reset, df_macro_aligned, on='Date', how='left')

    # Forward fill macro data within each ticker group after merge, in case of gaps
    # This handles cases where a stock might have data on a day macro data isn't available
    # (though df_macro_aligned should already be daily)
    for col in df_macro_aligned.columns:
        df_combined[col] = df_combined.groupby('Ticker')[col].transform(lambda x: x.ffill())

    # Set multi-index back
    df_combined = df_combined.set_index(['Date', 'Ticker']).sort_index()

    print(f"  Aligned combined data shape: {df_combined.shape}")
    return df_combined

if __name__ == "__main__":
    print("--- Starting Data Cleaning & Alignment Phase ---")
    stock_df_raw, macro_df_raw = load_raw_data(RAW_DATA_DIR)

    stock_df_cleaned = clean_stock_data(stock_df_raw)

    # Align and combine
    combined_df = align_dataframes(stock_df_cleaned, macro_df_raw, target_frequency='D')

    # Save processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, 'processed_combined_data.parquet')
    combined_df.to_parquet(output_path)
    print(f"Processed combined data saved to {output_path}")

    print("\nData Cleaning & Alignment Phase Complete.")
    print(f"Final processed data head:\n{combined_df.head()}")
    print(f"Final processed data info:")
    combined_df.info()
