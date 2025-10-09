# db_init.py

import os
import subprocess
import sys
import time
from pathlib import Path
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from src.utils.env_utils import init_env

# --- Load environment variables ---
env = init_env(validate=True)


# --- Load database configuration ---
def load_db_config():
    try:
        config = {
            "host": env["DB_HOST"],
            "port": int(env["DB_PORT"]),
            "dbname": env["DB_TYPE"],  # Connect to default first for CREATE DATABASE
            "user": env["DB_USER"],
            "password": env["DB_PASSWORD"],
        }
        return config
    except Exception as e:
        print(f"[ERROR] Invalid DB config: {e}")
        sys.exit(1)


# --- Start PostgreSQL server ---
def start_postgres_server():
    print("Checking if PostgreSQL is already running...")

    def is_server_running():
        try:
            result = subprocess.run(['pg_isready', '-h', 'localhost', '-p', '5432'],
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    if is_server_running():
        print("PostgreSQL is already running (via pg_isready). ✅")
        return True

    brew_installed = True
    try:
        result = subprocess.run(['brew', 'services', 'list', 'postgresql@15'],
                                capture_output=True, text=True)
        if 'started' in result.stdout.lower() or 'running' in result.stdout.lower():
            print("PostgreSQL is already running (via brew). ✅")
            return True
    except FileNotFoundError:
        print("Homebrew not found. Falling back to pg_ctl.")
        brew_installed = False

    if brew_installed:
        try:
            result = subprocess.run(['brew', 'services', 'start', 'postgresql@15'],
                                    capture_output=True, text=True, check=True)
            print("Started PostgreSQL via Homebrew ✅")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to start PostgreSQL via Homebrew: {e.stderr}")

    print("Attempting to start via pg_ctl...")
    if os.uname().machine == 'arm64':
        pg_ctl_path = '/opt/homebrew/bin/pg_ctl'
        data_dir = '/opt/homebrew/var/postgres'
    else:
        pg_ctl_path = '/usr/local/bin/pg_ctl'
        data_dir = '/usr/local/var/postgres'

    try:
        if not os.path.exists(data_dir):
            subprocess.run([pg_ctl_path.replace('pg_ctl', 'initdb'), data_dir],
                           check=True, capture_output=True)
            print(f"Initialized Postgres data dir: {data_dir}")

        subprocess.run([pg_ctl_path, 'start', '-D', data_dir],
                       capture_output=True, text=True, check=True)
        print("Started PostgreSQL via pg_ctl ✅")
        return True
    except Exception as e:
        print(f"WARNING: Could not auto-start PostgreSQL: {e}")
        print("Please start manually: brew services start postgresql@15")
        return True  # Continue even if manual start needed


# --- Create database if not exists ---
def create_database(config):
    try:
        conn = psycopg2.connect(**config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        db_name = env["DB_NAME"]
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            print(f"Created database '{db_name}' ✅")
        else:
            print(f"Database '{db_name}' already exists ✅")

        cur.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"[ERROR] Database creation failed: {e}")
        return False


# --- Initialize database schema ---
def init_schema():
    app_config = load_db_config()
    app_config["dbname"] = env["DB_NAME"]

    schema_sql = """
    CREATE TABLE IF NOT EXISTS assets (
        asset_id SERIAL PRIMARY KEY,
        ticker TEXT UNIQUE NOT NULL,
        name TEXT,
        sector TEXT,
        country TEXT,
        currency TEXT
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
    """

    try:
        conn = psycopg2.connect(**app_config)
        cur = conn.cursor()
        cur.execute(schema_sql)
        conn.commit()
        cur.close()
        conn.close()
        print("Initialized schema ✅")
        return True
    except psycopg2.Error as e:
        print(f"[ERROR] Schema initialization failed: {e}")
        return False


# --- Verify database connection ---
def verify_connection():
    app_config = load_db_config()
    app_config["dbname"] = env["DB_NAME"]
    try:
        conn = psycopg2.connect(**app_config)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"Connected to PostgreSQL version: {version} ✅")
        cur.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"[ERROR] Verification failed: {e}")
        return False


# --- Clear all data ---
def clear_data():
    app_config = load_db_config()
    app_config["dbname"] = env["DB_NAME"]
    try:
        conn = psycopg2.connect(**app_config)
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE assets, trade_log RESTART IDENTITY CASCADE;")
        conn.commit()
        cur.close()
        conn.close()
        print("Cleared all data ✅")
        return True
    except psycopg2.Error as e:
        print(f"[ERROR] Failed to clear data: {e}")
        return False


# --- Full DB Initialization Pipeline ---
def db_init():
    print("Starting PostgreSQL setup...\n")
    config = load_db_config()

    print(f"Using config: {config['host']}:{config['port']} (User: {config['user']})")
    if not start_postgres_server():
        sys.exit(1)
    time.sleep(3)

    if not create_database(config):
        sys.exit(1)

    if not init_schema():
        sys.exit(1)

    if not verify_connection():
        sys.exit(1)

    if not clear_data():
        sys.exit(1)

    print("\n✅ Database is ready for use.\nUse 'python shutdown_postgres.py' to stop the server.")


if __name__ == "__main__":
    db_init()
