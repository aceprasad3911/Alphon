# db_test.py

import pytest
import psycopg2
import os

# Fixture for database connection

@pytest.fixture(scope="module")
def conn():
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'dbname': os.getenv('DB_NAME', 'alphondb'),
        'user': os.getenv('DB_USER', 'ayushmaanprasad'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    connection = psycopg2.connect(**config)
    yield connection
    connection.close()

# ---------------------------
# Test 1: Connection
# ---------------------------
def test_connection(conn):
    cur = conn.cursor()
    cur.execute("SELECT current_database();")
    db_name = cur.fetchone()[0]
    assert db_name == 'alphondb'
    cur.close()

# ---------------------------
# Test 2: Tables exist
# ---------------------------
def test_tables_exist(conn):
    tables = ['assets', 'asset_universe_versions', 'asset_universe_members',
              'price_data', 'fundamentals', 'macro_indicators', 'regime_indicators',
              'technical_indicators', 'graph_features', 'time_series_features',
              'model_runs', 'experiment_tags', 'model_experiment_link',
              'validation_folds', 'model_explanations', 'alpha_signals',
              'backtest_results', 'portfolio_holdings', 'trade_log',
              'data_source_log', 'preprocessing_steps', 'raw_data_cache']

    cur = conn.cursor()
    for table in tables:
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table}'
            );
        """)
        exists = cur.fetchone()[0]
        assert exists, f"Table {table} does not exist."
    cur.close()

# ---------------------------
# Test 3: Assets table has data
# ---------------------------
def test_assets(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM assets;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 4: Asset universe versions
# ---------------------------
def test_asset_universe_versions(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_universe_versions;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 5: Asset universe members
# ---------------------------
def test_asset_universe_members(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM asset_universe_members;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 6: Price data
# ---------------------------
def test_price_data(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM price_data;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 7: Fundamentals
# ---------------------------
def test_fundamentals(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fundamentals;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 8: Macro indicators
# ---------------------------
def test_macro_indicators(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM macro_indicators;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 9: Regime indicators
# ---------------------------
def test_regime_indicators(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM regime_indicators;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 10: Technical indicators
# ---------------------------
def test_technical_indicators(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM technical_indicators;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 11: Graph features
# ---------------------------
def test_graph_features(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM graph_features;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 12: Time series features
# ---------------------------
def test_time_series_features(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM time_series_features;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 13: Model runs
# ---------------------------
def test_model_runs(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM model_runs;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 14: Experiment tags
# ---------------------------
def test_experiment_tags(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM experiment_tags;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 15: Model experiment link
# ---------------------------
def test_model_experiment_link(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM model_experiment_link;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 16: Validation folds
# ---------------------------
def test_validation_folds(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM validation_folds;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 17: Model explanations
# ---------------------------
def test_model_explanations(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM model_explanations;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 18: Alpha signals
# ---------------------------
def test_alpha_signals(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM alpha_signals;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 19: Backtest results
# ---------------------------
def test_backtest_results(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM backtest_results;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 20: Portfolio holdings
# ---------------------------
def test_portfolio_holdings(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM portfolio_holdings;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 21: Trade log
# ---------------------------
def test_trade_log(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM trade_log;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 22: Data source log
# ---------------------------
def test_data_source_log(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM data_source_log;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 23: Preprocessing steps
# ---------------------------
def test_preprocessing_steps(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM preprocessing_steps;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()

# ---------------------------
# Test 24: Raw data cache
# ---------------------------
def test_raw_data_cache(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_data_cache;")
    count = cur.fetchone()[0]
    assert count > 0
    cur.close()
