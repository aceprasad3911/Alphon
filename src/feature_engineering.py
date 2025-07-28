import pandas as pd
import numpy as np
import os
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
FEATURES_DATA_DIR = 'data/features'
os.makedirs(FEATURES_DATA_DIR, exist_ok=True)

# --- Feature Engineering Functions ---

def load_processed_data(data_dir):
    """Loads processed combined data."""
    path = os.path.join(data_dir, 'processed_combined_data.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        print(f"Error: Processed data not found at {path}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculates basic technical indicators."""
    print("Calculating technical indicators...")
    if df.empty:
        print("  DataFrame is empty, skipping technical indicator calculation.")
        return df

    # Ensure multi-index is (Date, Ticker)
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['Date', 'Ticker']:
        df = df.set_index(['Date', 'Ticker']).sort_index()

    # Calculate daily returns
    df['Daily_Return'] = df.groupby('Ticker')['Adj Close'].pct_change()

    # Simple Moving Averages (SMA)
    df['SMA_10'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=10).mean())
    df['SMA_50'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50).mean())

    # Relative Strength Index (RSI) - Simplified for demonstration
    # For a full RSI, you'd need more complex calculations or a dedicated library like ta.
    # Here, just a basic momentum proxy.
    delta = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(level='Ticker').transform(lambda x: x.ewm(com=13, adjust=False).mean())
    avg_loss = loss.groupby(level='Ticker').transform(lambda x: x.ewm(com=13, adjust=False).mean())
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero

    # Volatility (e.g., 20-day rolling standard deviation of returns)
    df['Volatility_20D'] = df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(window=20).std())

    print("  Technical indicators calculated.")
    return df

def extract_tsfresh_features(df):
    """Extracts a comprehensive set of time-series features using tsfresh."""
    print("Extracting tsfresh features...")
    if df.empty:
        print("  DataFrame is empty, skipping tsfresh feature extraction.")
        return df

    # tsfresh requires a DataFrame with an 'id' column (Ticker), 'time' column (Date),
    # and the value columns.
    # Ensure 'Date' and 'Ticker' are columns, not index levels, for tsfresh.
    df_tsfresh_input = df.reset_index()

    # Select columns for feature extraction. 'Adj Close' is a good start.
    # You can add 'Volume', 'Daily_Return', etc., if desired.
    # For comprehensive features, use ComprehensiveFCParameters().
    # For faster testing, use MinimalFCParameters().
    settings = MinimalFCParameters() # Use Minimal for quicker testing, change to ComprehensiveFCParameters() for full set

    # Impute missing values before tsfresh extraction
    impute(df_tsfresh_input)

    # Extract features
    # The 'column_id' is 'Ticker', 'column_sort' is 'Date'
    # 'column_value' specifies which columns to extract features from.
    # 'column_kind' is optional, used if you have different types of time series.
    extracted_features = extract_features(
        df_tsfresh_input,
        column_id='Ticker',
        column_sort='Date',
        default_fc_parameters=settings,
        # You can specify column_value if you only want features from certain columns
        # column_value='Adj Close',
        # n_jobs=0 for single core, -1 for all cores (can be slow for large datasets)
        n_jobs=0
    )

    print(f"  tsfresh features extracted. Shape: {extracted_features.shape}")
    return extracted_features

if __name__ == "__main__":
    print("--- Starting Feature Engineering Phase ---")
    combined_df = load_processed_data(PROCESSED_DATA_DIR)

    if not combined_df.empty:
        # Calculate traditional technical indicators
        df_with_tech_indicators = calculate_technical_indicators(combined_df.copy()) # Use .copy() to avoid SettingWithCopyWarning

        # Extract tsfresh features (these will be per-ticker, not per-day)
        # tsfresh features are typically used as cross-sectional features for a given day
        # or as input to models that expect a fixed-size feature vector per time series.
        # For now, we'll save them separately.
        tsfresh_features_df = extract_tsfresh_features(combined_df.copy())

        # Save the DataFrame with technical indicators (daily, per-ticker)
        output_path_daily_features = os.path.join(FEATURES_DATA_DIR, 'daily_features.parquet')
        df_with_tech_indicators.to_parquet(output_path_daily_features)
        print(f"Daily features saved to {output_path_daily_features}")

        # Save the tsfresh features (per-ticker, aggregated over time)
        output_path_tsfresh_features = os.path.join(FEATURES_DATA_DIR, 'tsfresh_features.parquet')
        tsfresh_features_df.to_parquet(output_path_tsfresh_features)
        print(f"tsfresh features saved to {output_path_tsfresh_features}")

        print("\nFeature Engineering Phase Complete.")
        print(f"Daily features head:\n{df_with_tech_indicators.head()}")
        print(f"tsfresh features head:\n{tsfresh_features_df.head()}")
    else:
        print("Skipping feature engineering as no processed data was loaded.")

