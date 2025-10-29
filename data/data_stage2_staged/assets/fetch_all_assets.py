import pandas as pd
from pathlib import Path

# ===============================
# CONFIG
# ===============================
SCRIPT_DIR = Path(__file__).resolve().parent
DHSP_asset_file = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage1_raw" / "local" / "DHSP_1970_to_2018" / "historical_stocks.csv"
MYFD_asset_file = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage1_raw" / "local" / "MYFD" / "stock_details_5_years.csv"
SP500_asset_file = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage1_raw" / "local" / "SP_500_Stock_Data" / "all_stocks_5yr.csv"
USHSP_asset_file = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage1_raw" / "local" / "US_HSP_wt_earnings_data" / "dataset_summary.csv"
OUTPUT_FILE = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage2_staged" / "assets" / "all_assets.csv"

# Resolve paths to absolute
DHSP_RAW_FILE = DHSP_asset_file.resolve()
MYFD_RAW_FILE = MYFD_asset_file.resolve()
SP500_RAW_FILE = SP500_asset_file.resolve()
USHSP_RAW_FILE = USHSP_asset_file.resolve()
OUTPUT_FILE = OUTPUT_FILE.resolve()

# ===============================
# INDIVIDUAL DATASET LOADERS
# (unchanged from your version, keep structure)
# ===============================
def DHSP_assets():
    df = pd.read_csv(DHSP_RAW_FILE)
    df = df.rename(columns={
        "ticker": "ticker",
        "exchange": "exchange",
        "name": "name",
        "sector": "sector",
        "industry": "industry"
    })
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["exchange"] = df["exchange"].astype(str).str.strip().str.upper().replace("NAN", "")
    df["source"] = "DHSP"
    # Add missing fields
    for col in ["country", "currency", "isin"]:
        df[col] = ""
    # Keep only expected columns (some may not exist; fill if missing)
    cols = ["ticker", "name", "exchange", "sector", "industry", "country", "currency", "isin", "source"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    return df.drop_duplicates(subset=["ticker"])


def MYFD_assets():
    df = pd.read_csv(MYFD_RAW_FILE)
    df = df.rename(columns={"Company": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["source"] = "MYFD"
    # Ensure expected columns exist
    for c in ["name", "exchange", "sector", "industry", "country", "currency", "isin"]:
        if c not in df.columns:
            df[c] = ""
    cols = ["ticker", "name", "exchange", "sector", "industry", "country", "currency", "isin", "source"]
    df = df[cols]
    return df.drop_duplicates(subset=["ticker"])


def SP500_assets():
    df = pd.read_csv(SP500_RAW_FILE)
    # ticker stored in "Name"
    if "Name" not in df.columns:
        raise KeyError("Expected 'Name' column in SP500 input")
    df = df.rename(columns={"Name": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["source"] = "SP500"
    df["exchange"] = "NYSE/NASDAQ"
    # Ensure expected columns exist
    for c in ["name", "sector", "industry", "country", "currency", "isin"]:
        if c not in df.columns:
            df[c] = ""
    cols = ["ticker", "name", "exchange", "sector", "industry", "country", "currency", "isin", "source"]
    df = df[cols]
    return df.drop_duplicates(subset=["ticker"])


def USHSP_assets():
    df = pd.read_csv(USHSP_RAW_FILE)
    df = df.rename(columns={"symbol": "ticker"} if "symbol" in df.columns else {})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["source"] = "USHSP"
    for c in ["name", "exchange", "sector", "industry", "country", "currency", "isin"]:
        if c not in df.columns:
            df[c] = ""
    cols = ["ticker", "name", "exchange", "sector", "industry", "country", "currency", "isin", "source"]
    df = df[cols]
    return df.drop_duplicates(subset=["ticker"])


# ===============================
# MERGE / DEDUP LOGIC
# ===============================
def choose_first_nonempty(values):
    """
    From a list/Series of values, return the first that is not null/empty-string.
    If none, return empty string.
    """
    for v in values:
        if pd.isna(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return ""


def merge_group(df_group):
    """
    Merge rows for a single ticker (DataFrame slice) into one record.
    """
    # columns we want to merge (order matters for choice)
    out = {}
    out["ticker"] = df_group.name  # groupby key

    # name, sector, industry: choose first non-empty
    for col in ["name", "sector", "industry", "country", "currency", "isin"]:
        out[col] = choose_first_nonempty(df_group[col].tolist())

    # exchange: collect distinct non-empty exchanges and join with '|'
    exchanges = [e for e in df_group["exchange"].astype(str).str.strip().unique() if e and e.upper() != "NAN"]
    # normalize and keep order
    exchanges = [e for e in exchanges if e]
    out["exchange"] = "|".join(exchanges) if exchanges else ""

    # Combine sources: unique and sorted
    sources = [s for s in df_group["source"].astype(str).unique() if s]
    out["source"] = "|".join(sorted(sources))

    return pd.Series(out)


def combine_all_assets():
    print("🔹 Loading DHSP assets...")
    dhsp = DHSP_assets()

    print("🔹 Loading MYFD assets...")
    myfd = MYFD_assets()

    print("🔹 Loading SP500 assets...")
    sp500 = SP500_assets()

    print("🔹 Loading USHSP assets...")
    ushsp = USHSP_assets()

    print("✅ Concatenating sources...")
    combined = pd.concat([dhsp, myfd, sp500, ushsp], ignore_index=True, sort=False)

    # sanitize columns to ensure presence
    required_cols = ["ticker", "name", "exchange", "sector", "industry", "country", "currency", "isin", "source"]
    for c in required_cols:
        if c not in combined.columns:
            combined[c] = ""

    print("🔹 Grouping by ticker and merging rows (first-nonempty strategy)...")
    merged = combined.groupby("ticker", sort=True).apply(merge_group).reset_index(drop=True)

    # Final column order
    final_cols = ["ticker", "name", "exchange", "sector", "industry", "country", "currency", "isin", "source"]
    merged = merged[final_cols]

    # Save final CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(merged)} unique tickers to {OUTPUT_FILE}")


if __name__ == "__main__":
    combine_all_assets()
