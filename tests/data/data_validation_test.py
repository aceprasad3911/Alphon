import pandas as pd
from pathlib import Path

# ===============================
# CONFIG: File Paths
# ===============================
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_FILE = SCRIPT_DIR / ".." / ".." / ".." / "Alphon_final" / "data" / "data_stage2_staged" / "assets.csv"
ALL_STOCKS_FILE = SCRIPT_DIR / ".." / ".." / ".." / "Alphon_final" / "data" / "data_stage0_sources" / "local" / "S&P_500_Stock_Data" / "all_stocks_5yr.csv"

MATCHED_OUTPUT = SCRIPT_DIR / ".." / ".." / ".." / "Alphon_final" / "data" / "data_stage2_staged" / "processed" / "assets_matched.csv"
UNMATCHED_OUTPUT = SCRIPT_DIR / ".." / ".." / ".." / "Alphon_final" / "data" / "data_stage2_staged" / "processed" / "assets_unmatched.csv"

# ===============================
# STEP 1: LOAD CSVs
# ===============================
assets_df = pd.read_csv(ASSETS_FILE)
all_stocks_df = pd.read_csv(ALL_STOCKS_FILE)

# ===============================
# STEP 2: NORMALIZE FOR MATCHING
# ===============================
assets_df['ticker_norm'] = assets_df['ticker'].str.strip().str.upper()
all_stocks_df['Name_norm'] = all_stocks_df['Name'].str.strip().str.upper()

# ===============================
# STEP 3: MERGE ON TICKER
# ===============================
merged = pd.merge(
    all_stocks_df,
    assets_df[['ticker', 'ticker_norm']],
    left_on='Name_norm',
    right_on='ticker_norm',
    how='left'
)

# Drop duplicates so only one row per ticker
merged = merged.drop_duplicates(subset=['Name_norm'])

# Matched if ticker exists, unmatched if ticker is NaN
matched = merged[merged['ticker'].notna()]
unmatched = merged[merged['ticker'].isna()]

# ===============================
# STEP 4: SAVE RESULTS
# ===============================
MATCHED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
matched.to_csv(MATCHED_OUTPUT, index=False)
print(f"✅ Saved {len(matched)} matched assets to {MATCHED_OUTPUT}")

UNMATCHED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
unmatched.to_csv(UNMATCHED_OUTPUT, index=False)
print(f"⚠️ Saved {len(unmatched)} unmatched assets to {UNMATCHED_OUTPUT}")
