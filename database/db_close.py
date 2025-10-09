"""
db_close.py
------------------------------------
Gracefully stops the PostgreSQL service (Homebrew-managed)
and verifies shutdown using environment-based configuration.
"""

import subprocess
import time
import psycopg2
from src.utils.env_utils import init_env, get_database_config


def stop_postgres_service():
    """Stop PostgreSQL service managed by Homebrew."""
    print("Attempting to stop PostgreSQL (Homebrew)...\n")
    try:
        result = subprocess.run(
            ["brew", "services", "stop", "postgresql@15"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        output = result.stdout.strip() or result.stderr.strip()
        if any(keyword in output for keyword in ["Stopping", "stopped", "Service `postgresql@15`"]):
            print("✅ PostgreSQL stop command executed successfully.")
        else:
            print(f"⚠️ Unexpected brew output:\n{output}")
    except Exception as e:
        print(f"❌ Error stopping PostgreSQL: {e}")


def verify_shutdown(db_config: dict) -> bool:
    """Verify that PostgreSQL is no longer accepting connections."""
    print("Verifying PostgreSQL shutdown...")
    try:
        conn = psycopg2.connect(
            host=db_config["db_host"],
            port=db_config["db_port"],
            user=db_config["db_user"],
            password=db_config["db_password"],
            dbname=db_config["db_name"],
            connect_timeout=3
        )
        conn.close()
        print("❌ PostgreSQL is still running.")
        return False
    except psycopg2.OperationalError:
        print("✅ PostgreSQL stoppage verification successful.")
        return True
    except Exception as e:
        print(f"⚠️ Unexpected error during verification: {e}")
        return False


def db_close():
    """Main shutdown sequence."""
    print("🔻 Shutting down PostgreSQL server...")

    # 1️⃣ Load environment variables safely
    init_env()
    db_config = get_database_config()

    # 2️⃣ Stop PostgreSQL service
    stop_postgres_service()

    # 3️⃣ Allow time for service to stop completely
    time.sleep(3)

    # 4️⃣ Verify stoppage
    if verify_shutdown(db_config):
        print("🟢 PostgreSQL Server has been cleanly stopped.")
    else:
        print("🔴 PostgreSQL may still be running or failed to stop properly.")


if __name__ == "__main__":
    db_close()
