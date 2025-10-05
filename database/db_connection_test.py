import pytest
from sqlalchemy import create_engine, text
import os

# Load database configuration from environment variables (with defaults)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'alphondb')
DB_USER = os.getenv('DB_USER', 'ayushmaanprasad')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Bday3911*')

# Construct the connection URL
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@pytest.fixture(scope="session")
def engine():
    return create_engine(DATABASE_URL)


def test_db_connection(engine):
    """Test PostgreSQL connection and basic queries."""
    try:
        with engine.connect() as conn:
            # Test version query
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            assert "PostgreSQL" in version
            print("Connection successful! PostgreSQL version:", version)

            # List tables
            tables_result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """))
            tables = [row[0] for row in tables_result.fetchall()]
            print("Tables in 'public' schema:", tables)
            assert isinstance(tables, list)

            # Test current DB/user
            db_info = conn.execute(text("SELECT current_database(), current_user;"))
            db_name, user = db_info.fetchone()
            assert db_name == DB_NAME
            assert user == DB_USER
            print(f"Connected to database: {db_name} as user: {user}")

    except Exception as e:
        pytest.fail(f"Connection failed: {e}")


# If run directly (non-pytest), execute the test
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
