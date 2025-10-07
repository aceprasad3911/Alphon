# db_init.py

"""
Starts the local PostgreSQL server, initializes the 'alphondb' database structure,
and tests the connection.

Assumptions:
- PostgreSQL installed via Homebrew (macOS).
- Superuser: 'postgres' or from config (may need password prompt).
- Database: 'alphondb'.
- Config: Loads from config/database_config.yaml.

Usage: python db_init.py
"""

import os
import subprocess
import sys
import time  # Added for time.sleep
import yaml
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


# Load database configuration (adapted from previous edit)
def load_db_config():
    try:
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_PORT = int(os.getenv('DB_PORT', 5432))
        DB_NAME = os.getenv('DB_NAME', 'alphondb')
        DB_USER = os.getenv('DB_USER', 'ayushmaanprasad')
        DB_PASSWORD = os.getenv('DB_PASSWORD', '')  # No placeholder needed—direct from .env
        return {
            'host': DB_HOST,
            'port': DB_PORT,
            'dbname': 'postgres',  # For initial connection
            'user': DB_USER,
            'password': DB_PASSWORD
        }
    except ValueError as e:  # e.g., invalid port
        print(f"Config error: {e}")
        sys.exit(1)


# Start PostgreSQL server (enhanced with better error handling and fallbacks)
def start_postgres_server():
    print("Checking if PostgreSQL is already running...")

    # Helper: Check if server is responsive using pg_isready (if available)
    def is_server_running():
        try:
            result = subprocess.run(['pg_isready', '-h', 'localhost', '-p', '5432'],
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0  # 0 = accepting connections
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    if is_server_running():
        print("PostgreSQL is already running (via pg_isready). Skipping start.")
        return True

    # Brew check and start
    brew_installed = True
    try:
        # Check status via brew
        result = subprocess.run(['brew', 'services', 'list', 'postgresql@15'],
                                capture_output=True, text=True)
        if 'started' in result.stdout.lower() or 'running' in result.stdout.lower():
            print("PostgreSQL is already running (via brew). Skipping start.")
            return True
    except FileNotFoundError:
        print("Homebrew not found. Falling back to pg_ctl.")
        brew_installed = False

    if brew_installed:
        # Start via brew with output capture
        try:
            result = subprocess.run(['brew', 'services', 'start', 'postgresql@15'],
                                    capture_output=True, text=True, check=True)
            print("Started PostgreSQL server via Homebrew ✅")
            print(f"Brew output: {result.stdout.strip() if result.stdout else 'No output'}\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to start via brew: Exit code {e.returncode}")
            if e.stderr:
                print(f"Brew error details: {e.stderr.strip()}")
            print("Try manual brew start: brew services start postgresql")
            print("Or if versioned: brew services start postgresql@15")

        # Fallback: pg_ctl start (adjust paths for your setup)
    print("Attempting start via pg_ctl...")
    # Detect platform: Apple Silicon vs Intel
    if os.uname().machine == 'arm64':  # Apple Silicon
        pg_ctl_path = '/opt/homebrew/bin/pg_ctl'
        data_dir = '/opt/homebrew/var/postgres'
    else:  # Intel
        pg_ctl_path = '/usr/local/bin/pg_ctl'
        data_dir = '/usr/local/var/postgres'

    try:
        # Check if data dir exists; init if not
        if not os.path.exists(data_dir):
            init_cmd = [pg_ctl_path.replace('pg_ctl', 'initdb'), data_dir]
            subprocess.run(init_cmd, check=True, capture_output=True)
            print(f"Initialized Postgres data dir: {data_dir}")

        result = subprocess.run([pg_ctl_path, 'start', '-D', data_dir],
                                capture_output=True, text=True, check=True)
        print("Started PostgreSQL via pg_ctl.")
        if result.stdout:
            print(f"pg_ctl output: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print(f"pg_ctl not found at {pg_ctl_path}. Install PostgreSQL or adjust path.")
        print("Manual install: brew install postgresql")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start via pg_ctl: Exit code {e.returncode}")
        if e.stderr:
            print(f"pg_ctl error: {e.stderr.strip()}")
        print("Manual pg_ctl start: /opt/homebrew/bin/pg_ctl -D /opt/homebrew/var/postgres start")

        # If all fails, warn but don't exit—proceed to connection test
    print("WARNING: Could not auto-start PostgreSQL. Assuming manual start or already running.")
    print("Please start manually (see above) and re-run the script.")
    return True  # But don't sys.exit here


# Create database if not exists
def create_database(config):
    try:
        # Connect to default 'postgres' DB as superuser
        conn = psycopg2.connect(**config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        db_name = os.getenv('DB_NAME', 'alphondb')
        cur.execute(sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s"), (db_name,))

        if not cur.fetchone():
            # Fixed: Added space in SQL template
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            print(f"Created database '{db_name}'.")
        else:
            print(f"Database '{db_name}' already exists ✅")

        cur.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
        return False


# Initialize database schema (basic example: create a 'trades' table for backtesting)
def init_schema(config):
    db_name = os.getenv('DB_NAME', 'alphondb')
    app_config = load_db_config()
    app_config['dbname'] = db_name  # Switch to app DB

    try:
        conn = psycopg2.connect(**app_config)
        cur = conn.cursor()

        # Example schema: Trades table (adjust for your needs)
        create_table_sql = """CREATE TABLE IF NOT EXISTS assets (
        asset_id SERIAL PRIMARY KEY,
        ticker TEXT UNIQUE NOT NULL,
        name TEXT,
        sector TEXT,
        country TEXT,
        currency TEXT
    );
    CREATE TABLE IF NOT EXISTS asset_universe_versions (
        version_id SERIAL PRIMARY KEY,
        version_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS asset_universe_members (
        member_id SERIAL PRIMARY KEY,
        version_id INT REFERENCES asset_universe_versions(version_id),
        asset_id INT REFERENCES assets(asset_id)
    );
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
    CREATE TABLE IF NOT EXISTS fundamentals (
        fundamental_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        key TEXT,
        value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS macro_indicators (
        macro_id SERIAL PRIMARY KEY,
        indicator_name TEXT,
        date DATE NOT NULL,
        value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS regime_indicators (
        regime_id SERIAL PRIMARY KEY,
        date DATE NOT NULL,
        indicator_name TEXT,
        value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS technical_indicators (
        tech_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        indicator_name TEXT,
        value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS graph_features (
        graph_feature_id SERIAL PRIMARY KEY,
        date DATE NOT NULL,
        feature_name TEXT,
        value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS time_series_features (
        ts_feature_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        feature_name TEXT,
        value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id SERIAL PRIMARY KEY,
        model_name TEXT,
        run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS experiment_tags (
        tag_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        tag TEXT
    );
    CREATE TABLE IF NOT EXISTS model_experiment_link (
        link_id SERIAL PRIMARY KEY,
        model_id INT REFERENCES model_runs(run_id),
        experiment_id INT REFERENCES experiment_tags(tag_id)
    );
    CREATE TABLE IF NOT EXISTS validation_folds (
        fold_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        fold_number INT,
        metric_name TEXT,
        metric_value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS model_explanations (
        explanation_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        explanation JSONB
    );
    CREATE TABLE IF NOT EXISTS alpha_signals (
        alpha_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        signal_value NUMERIC
    );
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        start_date DATE,
        end_date DATE,
        performance_metric NUMERIC
    );
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        holding_id SERIAL PRIMARY KEY,
        portfolio_id INT,
        asset_id INT REFERENCES assets(asset_id),
        date DATE NOT NULL,
        weight NUMERIC
    );
    CREATE TABLE IF NOT EXISTS trade_log (
        trade_id SERIAL PRIMARY KEY,
        portfolio_id INT,
        asset_id INT REFERENCES assets(asset_id),
        trade_date DATE NOT NULL,
        action TEXT,
        quantity NUMERIC,
        price NUMERIC
    );
    CREATE TABLE IF NOT EXISTS data_source_log (
        log_id SERIAL PRIMARY KEY,
        source_name TEXT,
        date_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT
    );
 
    CREATE TABLE IF NOT EXISTS preprocessing_steps (
        step_id SERIAL PRIMARY KEY,
        run_id INT REFERENCES model_runs(run_id),
        step_description TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS raw_data_cache (
        cache_id SERIAL PRIMARY KEY,
        asset_id INT REFERENCES assets(asset_id),
        data_type TEXT,
        data JSONB,
        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
        """
        cur.execute(create_table_sql)

        # Insert a sample record for testing
        # Insert a sample record for testing
        insert_sample_sql = """
        -- Ensure an asset exists first
        INSERT INTO assets (ticker, name, sector, country, currency)
        VALUES ('AAPL', 'Apple Inc.', 'Technology', 'USA', 'USD')
        ON CONFLICT (ticker) DO NOTHING;

        -- Then insert a sample trade record
        INSERT INTO trade_log (portfolio_id, asset_id, trade_date, action, quantity, price)
        VALUES (
            1,
            (SELECT asset_id FROM assets WHERE ticker = 'AAPL'),
            CURRENT_DATE,
            'BUY',
            100,
            150.00
        )
        ON CONFLICT DO NOTHING;
        """

        cur.execute(insert_sample_sql)

        conn.commit()
        cur.close()
        conn.close()
        print("Initialized database schema with sample data ✅")
        return True
    except psycopg2.Error as e:
        print(f"Error initializing schema: {e}")
        return False


# Verify connection (renamed from 'test_connection' to avoid pytest collection)
def verify_connection(config):
    """Verify PostgreSQL connection and basic queries using psycopg2."""
    db_name = os.getenv('DB_NAME', 'alphondb')
    app_config = load_db_config()
    app_config['dbname'] = db_name  # Switch to app DB

    try:
        conn = psycopg2.connect(**app_config)
        cur = conn.cursor()

        # Test version query
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"PostgreSQL version: {version} Connection successful ✅\n")

        # List tables in public schema
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables in 'public' schema: {tables}")

        # Test current DB/user
        cur.execute("SELECT current_database(), current_user;")
        db_info = cur.fetchone()
        current_db, current_user = db_info
        print(f"\nConnected to database: {current_db} as user: {current_user} ✅")

        # Test query on sample table
        cur.execute("SELECT COUNT(*) FROM assets;")
        count = cur.fetchone()[0]
        print(f"Test query: Found {count} trades in the database ✅")

        cur.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"Connection failed: {e}")
        return False


# Clear all data from tables after verification (resets sequences and handles FKs)
def clear_data(config):
    """Clear all data from the database tables after verification, resetting to empty state."""
    db_name = os.getenv('DB_NAME', 'alphondb')
    app_config = load_db_config()
    app_config['dbname'] = db_name  # Switch to app DB
    try:
        conn = psycopg2.connect(**app_config)
        cur = conn.cursor()
        clear_sql = """
        TRUNCATE TABLE assets, asset_universe_versions, asset_universe_members, price_data, 
        fundamentals, macro_indicators, regime_indicators, technical_indicators, graph_features, 
        time_series_features, model_runs, experiment_tags, model_experiment_link, validation_folds, 
        model_explanations, alpha_signals, backtest_results, portfolio_holdings, trade_log, 
        data_source_log, preprocessing_steps, raw_data_cache
        RESTART IDENTITY CASCADE;
        """
        cur.execute(clear_sql)
        conn.commit()
        cur.close()
        conn.close()
        print("Cleared all data from tables (sequences reset) ✅")
        return True
    except psycopg2.Error as e:
        print(f"Error clearing data: {e}")
        return False


def db_init():
    print("Starting PostgreSQL setup...\n")

    # Load config
    config = load_db_config()
    print(f"Using config: Host={config['host']}, Port={config['port']}, User={config['user']}")

    # Step 1: Start server
    if not start_postgres_server():
        sys.exit(1)

    # Wait a bit for server to fully start
    time.sleep(3)

    # Step 2: Create database
    if not create_database(config):
        sys.exit(1)

    # Step 3: Initialize schema
    if not init_schema(config):
        sys.exit(1)

    # Step 4: Verify connection (updated function name)
    if not verify_connection(config):
        sys.exit(1)

    # Step 5: Clear data after verification
    if not clear_data(config):
        sys.exit(1)

    print(" Database is ready. PostgreSQL startup and initialization complete ✅")
    print("\nBegin application now. \nUse 'python shutdown_postgres.py' to stop the server.")


if __name__ == "__main__":
    db_init()
