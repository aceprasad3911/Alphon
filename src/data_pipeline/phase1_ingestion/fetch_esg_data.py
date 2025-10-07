"""
fetch_esg_data.py
Ingest ESG dataset(s) into data/raw/esg.
"""
import json
from pathlib import Path
from src.utils.logging_utils import setup_logger

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "esg"
logger = setup_logger("ingest.esg", "reports/pipeline.log")

def run(symbols=None):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # Placeholder ESG sample
    sample = {"symbol": "AAPL", "esg_score": 72.1}
    with open(DATA_ROOT / "sample_esg.json", "w") as f:
        json.dump(sample, f)
    logger.info("ESG ingestion placeholder complete.")
