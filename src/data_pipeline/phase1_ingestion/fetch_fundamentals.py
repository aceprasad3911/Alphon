"""
fetch_fundamentals.py
Placeholder for fundamentals ingestion (FMP, AlphaVantage, EDGAR parsing).
Writes raw JSON/CSV to data/raw/fundamentals.
"""
import json
from pathlib import Path
from src.utils.logging_utils import setup_logger

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "fundamentals"
logger = setup_logger("ingest.fund", "reports/pipeline.log")

def run(symbols=None):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # TODO: Put real API calls here. For now we write a dummy file.
    dummy = {"symbol": "AAPL", "fiscalDate": "2024-12-31", "peRatio": 25.3}
    out_file = DATA_ROOT / "sample_fundamentals.json"
    with open(out_file, "w") as f:
        json.dump(dummy, f, indent=2)
    logger.info(f"Wrote dummy fundamentals to {out_file}")
