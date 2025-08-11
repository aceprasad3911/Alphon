"""
seed_data.py
Populates Alpha Signal Discovery DB with sample assets and OHLCV price data.
"""

import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# --------------------
# CONFIGURE DB CONNECTION
# --------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password_here")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "alpha_signals")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# --------------------
# 1. Insert Example Assets
# --------------------
assets_data = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "country": "USA", "currency": "USD"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "country": "USA", "currency": "USD"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "country": "USA", "currency": "USD"},
]

with engine.begin() as conn:
    for asset in assets_data:
        conn.execute("""
            INSERT INTO assets (ticker, name, sector, country, currency)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO NOTHING;
        """, (asset["ticker"], asset["name"], asset["sector"], asset["country"], asset["currency"]))

print("✅ Inserted example assets.")

# --------------------
# 2. Download and Insert OHLCV Data
# --------------------
tickers = [a["ticker"] for a in assets_data]
start_date = "2024-01-01"
end_date = "2024-06-30"

for ticker in tickers:
    print(f"📥 Downloading {ticker} data...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print(f"⚠️ No data returned for {ticker}")
        continue

    # Reset index to get Date column
    df.reset_index(inplace=True)

    # Get asset_id for ticker
    with engine.begin() as conn:
        asset_id = conn.execute("SELECT asset_id FROM assets WHERE ticker = %s", (ticker,)).scalar()

    # Prepare dataframe for insertion
    df["asset_id"] = asset_id
    df.rename(columns={
        "Date": "date",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Adj Close": "adj_close_price",
        "Volume": "volume"
    }, inplace=True)

    # Insert into DB
    df[["asset_id", "date", "open_price", "high_price", "low_price",
        "close_price", "adj_close_price", "volume"]].to_sql(
        "price_data", engine, if_exists="append", index=False
    )

print("✅ OHLCV price data inserted.")

# --------------------
# 3. Quick Check
# --------------------
with engine.begin() as conn:
    result = conn.execute("""
        SELECT a.ticker, COUNT(p.price_id) AS rows
        FROM assets a
        JOIN price_data p ON a.asset_id = p.asset_id
        GROUP BY a.ticker;
    """)
    for row in result:
        print(f"{row.ticker}: {row.rows} rows in price_data")

print("🎯 Database seeding complete!")
