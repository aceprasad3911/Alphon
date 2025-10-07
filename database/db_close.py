"""
db_close.py
Gracefully stops the PostgreSQL service (Homebrew-managed) and verifies shutdown.
"""

import subprocess
import time
import psycopg2

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "ayushmaanprasad"
DB_PASSWORD = "Bday3911*"
DB_NAME = "alphondb"


# ------------------------------------------------------------
# STOP POSTGRESQL SERVICE
# ------------------------------------------------------------
def stop_postgres_service():
    print("Attempting to stop PostgreSQL (Homebrew)...\n")
    try:
        result = subprocess.run(
            ["brew", "services", "stop", "postgresql@15"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output = result.stdout.strip() or result.stderr.strip()
        if "Stopping" in output or "stopped" in output or "Service `postgresql@15`" in output:
            print("✅ PostgreSQL stop command executed successfully.")
        else:
            print(f"⚠️ Unusual brew output:\n{output}")
    except Exception as e:
        print(f"❌ Error stopping PostgreSQL: {e}")


# ------------------------------------------------------------
# VERIFY SHUTDOWN STATUS
# ------------------------------------------------------------
def verify_shutdown():
    print("Verifying PostgreSQL shutdown...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            connect_timeout=3
        )
        conn.close()
        print("❌ PostgreSQL is still running.")
        return False
    except psycopg2.OperationalError:
        print("✅ PostgreSQL stoppage verification succesful.")
        return True
    except Exception as e:
        print(f"⚠️ Unexpected error during verification: {e}")
        return False


# ------------------------------------------------------------
# MAIN SHUTDOWN SEQUENCE
# ------------------------------------------------------------
def db_close():
    print("🔻 Shutting down PostgreSQL server...")
    stop_postgres_service()

    # Allow time for service to stop completely
    time.sleep(3)

    if verify_shutdown():
        print("🟢 PostgreSQL Server has been cleanly stopped.")
    else:
        print("🔴 PostgreSQL may still be running or failed to stop properly.")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    db_close()
