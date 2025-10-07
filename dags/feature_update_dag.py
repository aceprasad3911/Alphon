"""
Simple DAG to run incremental feature updates (e.g., hourly/minute pipelines).
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.ingestion.fetch_prices import run as fetch_prices
from src.transformations.feature_engineering import run as feature_engineering

default_args = {
    "owner": "alphon",
    "retries": 0,
}

with DAG(
    "alphon_feature_update",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:
    t_fetch = PythonOperator(task_id="fetch_latest_prices", python_callable=lambda: fetch_prices(tickers=None))
    t_feat = PythonOperator(task_id="update_features", python_callable=feature_engineering)
    t_fetch >> t_feat
