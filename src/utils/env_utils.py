"""
env_utils.py
------------------------------------
Unified environment utility for loading, validating, and accessing
configuration values across the Alphon_Final project.

Features:
- Automatically finds and loads the root .env file (from anywhere in repo)
- Provides safe access to API and DB credentials
- Supports type casting and validation
- Industry-standard logging and structure for data pipelines
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# 1. LOAD .ENV FILE (project-root aware)
# ============================================================

def load_env():
    """
    Loads the .env file located at the project root, regardless of where called from.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"  # project_root/.env
    if not env_path.exists():
        raise FileNotFoundError(f"❌ .env file not found at expected location: {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)
    return env_path


# Automatically load on import
try:
    ENV_PATH = load_env()
except FileNotFoundError:
    print("⚠️ Warning: .env file not found. Ensure it exists at project root before running scripts.")


# ============================================================
# 2. CORE ENV VAR RETRIEVAL FUNCTION
# ============================================================

def get_env_var(var_name: str, required: bool = True, cast_type=str, default=None):
    """
    Retrieve and optionally type-cast an environment variable.

    Args:
        var_name (str): Name of the variable in the .env file.
        required (bool): Raise an error if missing (default True).
        cast_type: Optional type (e.g., int, bool, float).
        default: Default value if variable is not found.

    Returns:
        The value of the environment variable, properly casted.
    """
    value = os.getenv(var_name, default)

    if required and (value is None or value == ""):
        raise EnvironmentError(f"❌ Missing required environment variable: {var_name}")

    if value is not None and cast_type is not str:
        try:
            if cast_type is bool:
                value = str(value).lower() in ("1", "true", "yes", "on")
            else:
                value = cast_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"❌ Failed to cast environment variable {var_name} to {cast_type.__name__}")

    return value


# ============================================================
# 3. API CONFIGURATION HANDLERS
# ============================================================

def get_api_credentials(source: str) -> dict:
    """
    Retrieve API key and base URL for a given data source.

    Args:
        source (str): API source name (e.g., 'ALPHA_VANTAGE', 'FRED', 'QUANDL').

    Returns:
        dict: { 'api_key': str, 'base_url': str }
    """
    source = source.upper()
    api_key_var = f"API_KEY_{source}"
    base_url_var = f"BASE_URL_{source}"

    api_key = get_env_var(api_key_var)
    base_url = get_env_var(base_url_var, required=False)

    if not base_url:
        print(f"⚠️ Warning: No BASE_URL found for {source}. Please complete it in your .env file.")

    return {
        "api_key": api_key,
        "base_url": base_url or ""
    }


# ============================================================
# 4. DATABASE CONFIGURATION HANDLER
# ============================================================

def get_database_config() -> dict:
    """
    Retrieve all database connection settings from the environment.

    Returns:
        dict: Dictionary ready for psycopg2.connect() or SQLAlchemy engine.
    """
    return {
        "db_host": get_env_var("DB_HOST"),
        "db_port": get_env_var("DB_PORT", cast_type=int),
        "db_name": get_env_var("DB_NAME"),
        "db_user": get_env_var("DB_USER"),
        "db_password": get_env_var("DB_PASSWORD"),
        "db_type": get_env_var("DB_TYPE", required=False, default="postgres")
    }


# ============================================================
# 5. VALIDATION UTILITY
# ============================================================

def validate_env():
    """
    Run validation to confirm critical environment variables are loaded.
    """
    required_vars = [
        "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
        "API_KEY_ALPHA_VANTAGE", "API_KEY_FRED", "API_KEY_QUANDL"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"❌ Missing environment variables: {', '.join(missing)}")

    print(f"✅ Environment validated successfully from: {ENV_PATH}")


# ============================================================
# 6. STANDALONE EXECUTION TEST
# ============================================================

# ============================================================
# 7. UNIVERSAL ENV INITIALIZER
# ============================================================

def init_env(validate: bool = False) -> dict:
    """
    Initialize and return all environment variables as a dictionary.
    Recommended to call once at the start of scripts or notebooks.

    Args:
        validate (bool): Whether to run environment validation after loading.

    Returns:
        dict: Dictionary of all environment variables (os.environ copy).
    """
    load_env()  # Ensure .env is loaded
    if validate:
        validate_env()

    env_dict = dict(os.environ)
    print(f"🌍 Environment initialized with {len(env_dict)} variables.")
    return env_dict


if __name__ == "__main__":
    validate_env()
    creds = get_api_credentials("ALPHA_VANTAGE")
    print(f"🔑 API Key (truncated): {creds['api_key'][:6]}... | 🌐 Base URL: {creds['base_url'] or 'N/A'}")
