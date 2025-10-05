"""
main1.py
Full database initialization + optional reset + seed data.
"""

import os
import subprocess
import time
import sys
import yaml
from pathlib import Path

# ----------------------------
# CONFIGURATION FROM YAML
# ----------------------------
base_dir = Path(__file__).resolve().parent
config_path = base_dir / 'config' / 'database_config.yaml'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

db_host = config['postgres']['host']
db_port = str(config['postgres']['port'])
db_name = config['postgres']['db_name']
db_user = config['postgres']['user']
db_password = config['postgres']['password']  # Use env vars for production

print(f"DB Host: {db_host}")
print(f"DB Port: {db_port}")
print(f"DB Name: {db_name}")
print(f"DB User: {db_user}")
print(f"DB Password: {db_password}")

# Paths to scripts
PROJECT_ROOT = Path(__file__).resolve().parent
DB_CLIENT = PROJECT_ROOT / "data" / "db" / "db_client.py"
SEED_DATA = PROJECT_ROOT / "data" / "db" / "seed_data.py"

# ----------------------------
# HELPER FUNCTION
# ----------------------------
def run_command(cmd, check=True):
    """Run a shell command and print it."""
    print(f"➡️  {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

# ----------------------------
# MAIN LOGIC
# ----------------------------
if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv

    # 1. Start PostgreSQL service (Homebrew)
    print("🚀 Starting PostgreSQL service...")
    try:
        run_command(["brew", "services", "start", "postgresql"], check=False)
    except Exception as e:
        print(f"⚠️ Warning: Could not start PostgreSQL via brew. {e}")

    time.sleep(2)

    # 2. Full reset if flag is passed
    if reset_flag:
        print(f"🗑 Dropping database '{db_name}' (if exists)...")
        run_command([
            "dropdb", "-h", db_host, "-p", db_port, "-U", db_user, db_name
        ], check=False)

    # 3. Create fresh database
    print(f"📦 Creating database '{db_name}'...")
    run_command([
        "createdb", "-h", db_host, "-p", db_port, "-U", db_user, db_name
    ], check=False)

    # 4. Export env vars for Python scripts
    os.environ["DB_USER"] = db_user
    os.environ["DB_PASSWORD"] = db_password
    os.environ["DB_HOST"] = db_host
    os.environ["DB_PORT"] = db_port
    os.environ["DB_NAME"] = db_name

    # 5. Create tables from db_client.py
    print("🛠 Creating tables...")
    run_command(["python", str(DB_CLIENT)])

    # 6. Seed data
    print("🌱 Inserting seed data...")
    run_command(["python", str(SEED_DATA)])

    # 7. Verify data
    print("🔍 Verifying database contents...")
    run_command([
        "psql",
        "-h", db_host,
        "-p", db_port,
        "-U", db_user,
        "-d", db_name,
        "-c", "SELECT ticker, COUNT(*) FROM assets a JOIN price_data p ON a.asset_id = p.asset_id GROUP BY ticker;"
    ], check=False)

    print("🎯 Database initialization complete!")
