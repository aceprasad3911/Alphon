"""
drift_monitor.py
Runs basic data stability & drift checks and writes simple HTML report.
For production use, integrate Evidently or WhyLabs.
"""
from pathlib import Path
import pandas as pd
from src.utils.logging_utils import setup_logger

logger = setup_logger("monitor.drift", "reports/pipeline.log")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run():
    # Placeholder checks: compare distributions of a sample feature between two files
    try:
        current = pd.read_parquet("data/processed/AAPL_features.parquet") if Path("data/processed/AAPL_features.parquet").exists() else None
        baseline = pd.read_parquet("data/processed/AAPL_features.parquet") if Path("data/processed/AAPL_features.parquet").exists() else None
        report = REPORTS_DIR / "data_drift_report.html"
        with open(report, "w") as f:
            f.write("<html><body><h1>Data Drift Report</h1><p>Placeholder - integrate Evidently for real checks.</p></body></html>")
        logger.info(f"Generated drift report at {report}")
    except Exception as e:
        logger.exception(f"Failed drift monitoring: {e}")
