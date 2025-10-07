"""
Airflow DAG that orchestrates the pipeline phases.
Requires Airflow environment to run.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.ingestion.fetch_prices import run as fetch_prices
from src.ingestion.fetch_fundamentals import run as fetch_fundamentals
from src.staging.load_to_staging import run as load_to_staging
from src.transformations.feature_engineering import run as feature_engineering
from src.loading.load_features_to_db import run as load_features
from src.monitoring.drift_monitor import run as drift_monitor

default_args = {
    "owner": "alphon",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    "alphon_data_pipeline",
    default_args=default_args,
    description="Alphon pipeline DAG",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    t1 = PythonOperator(task_id="fetch_prices", python_callable=fetch_prices)
    t1b = PythonOperator(task_id="fetch_fundamentals", python_callable=fetch_fundamentals)
    t2 = PythonOperator(task_id="load_to_staging", python_callable=load_to_staging)
    t3 = PythonOperator(task_id="feature_engineering", python_callable=feature_engineering)
    t4 = PythonOperator(task_id="load_features", python_callable=load_features)
    t5 = PythonOperator(task_id="drift_monitor", python_callable=drift_monitor)

    [t1, t1b] >> t2 >> t3 >> t4 >> t5
