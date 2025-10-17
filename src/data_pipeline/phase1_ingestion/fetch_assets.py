import pandas as pd
from pathlib import Path

# ===============================
# CONFIG
# ===============================
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_FILE_1 = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage0_sources" / "local" / "DHSP_1970_to_2018" / "historical_stocks.csv"
OUTPUT_FILE = SCRIPT_DIR / ".." / ".." / ".." / "data" / "data_stage2_staged" / "assets.csv"

# Resolve paths to absolute
RAW_FILE = RAW_FILE_1.resolve()
OUTPUT_FILE = OUTPUT_FILE.resolve()

# ===============================
# STEP 1: LOAD RAW CSV
# ===============================
if not RAW_FILE_1.exists():
    raise FileNotFoundError(f"CSV not found at: {RAW_FILE_1}")

df = pd.read_csv(RAW_FILE_1)
print(df.head())

# ===============================
# STEP 2: NORMALIZE COLUMNS
# ===============================
# Rename columns to match the database schema if needed
df = df.rename(columns={
    "ticker": "ticker",
    "exchange": "exchange",
    "name": "name",
    "sector": "sector"
})

# ===============================
# STEP 3: ADD MISSING SCHEMA FIELDS
# ===============================
# Fill in columns not in CSV
missing_columns = {
    "country": "",
    "currency": "",
    "isin": "",
    "active": True,
    "inception_date": pd.NaT
}

for col, default_value in missing_columns.items():
    df[col] = df.get(col, default_value)

# ===============================
# STEP 4: CLEAN AND VALIDATE
# ===============================
# Deduplicate
df = df.drop_duplicates(subset=["ticker", "exchange"])

# Standardize strings
df["ticker"] = df["ticker"].str.strip().str.upper()
df["exchange"] = df["exchange"].str.strip().str.upper()
df["name"] = df["name"].str.strip()
df["sector"] = df["sector"].str.strip().str.title()

# ===============================
# STEP 5: REORDER COLUMNS TO MATCH DB SCHEMA
# ===============================
ordered_cols = [
    "ticker", "name", "exchange", "sector",
    "country", "currency", "isin", "active", "inception_date"
]
df = df[ordered_cols]

# ===============================
# STEP 6: SAVE TO PARQUET (CREATE OR APPEND)
# ===============================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    existing_df = pd.read_csv(OUTPUT_FILE)
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ticker", "exchange"])
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Appended {len(df)} new rows. Total assets now: {len(combined_df)}")
else:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Created new assets CSV file with {len(df)} rows")