"""
db_client.py
Initializes the Alpha Signal Discovery database schema.
"""

import os
from sqlalchemy import create_engine, text

# --------------------
# DB CONFIG
# --------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password_here")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "alpha_signals")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# --------------------
# CREATE TABLES
# --------------------
CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS assets (
        asset_id SERIAL PRIMARY KEY,
        ticker TEXT UNIQUE NOT NULL,
        name TEXT,
        sector TEXT,
        country TEXT,
        currency TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS asset_universe_versions (
        version_id SERIAL PRIMARY KEY,
        version_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS asset_universe_members (
        member_id SERIAL PRIMARY KEY,
        version_id INT REFERENCES asset_universe_versions(version_id),
        asset_id INT REFERENCES assets(asset_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS price_data (
        price_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC,
        adj_close_price NUMERIC,
        volume BIGINT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamentals (
        fundamental_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        key TEXT,
        value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS macro_indicators (
        macro_id SERIAL PRIMARY KEY,
        indicator_name TEXT,
        date DATE NOT NULL,
        value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS regime_indicators (
        regime_id SERIAL PRIMARY KEY,
        date DATE NOT NULL,
        indicator_name TEXT,
        value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS technical_indicators (
        tech_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        indicator_name TEXT,
        value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_features (
        graph_feature_id SERIAL PRIMARY KEY,
        date DATE NOT NULL,
        feature_name TEXT,
        value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS time_series_features (
        ts_feature_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        feature_name TEXT,
        value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id SERIAL PRIMARY KEY,
        model_name TEXT,
        run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_tags (
        tag_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        tag TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_experiment_link (
        link_id SERIAL PRIMARY KEY,
        model_id INT REFERENCES model_runs(run_id),
        experiment_id INT REFERENCES experiment_tags(tag_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_folds (
        fold_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        fold_number INT,
        metric_name TEXT,
        metric_value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_explanations (
        explanation_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        explanation JSONB
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS alpha_signals (
        alpha_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        signal_value NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        start_date DATE,
        end_date DATE,
        performance_metric NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        holding_id SERIAL PRIMARY KEY,
        portfolio_id INT,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        weight NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trade_log (
        trade_id SERIAL PRIMARY KEY,
        portfolio_id INT,
        asset_id INT REFERENCES assets(asset_id),
        trade_date DATE NOT NULL,
        action TEXT,
        quantity NUMERIC,
        price NUMERIC
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS data_source_log (
        log_id SERIAL PRIMARY KEY,
        source_name TEXT,
        date_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS preprocessing_steps (
        step_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        step_description TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_data_cache (
        cache_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        data_type TEXT,
        data JSONB,
        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
]


def create_all_tables():
    """Creates all tables in the database."""
    with engine.begin() as conn:
        for sql in CREATE_TABLES_SQL:
            conn.execute(text(sql))
    print("✅ All tables created successfully.")


if __name__ == "__main__":
    print("Creating all tables in the database...")
    create_all_tables()

