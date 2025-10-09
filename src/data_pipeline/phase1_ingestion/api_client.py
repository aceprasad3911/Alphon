"""
api_client.py
------------------------------------
Unified API client for ALPHA_VANTAGE, FRED, and QUANDL.
Includes endpoint definitions, auth methods, and response schema validation.
Designed to pass standard api_tests.py without modifications.
"""

import requests
from src.utils.env_utils import load_env, get_api_credentials  # import from env_utils

# ========================
# API CLIENT
# ========================
class APIClient:
    SCHEMAS = {
        "ALPHA_VANTAGE": ["symbol", "open", "close", "high", "low", "volume"],
        "FRED": ["id", "realtime_start", "realtime_end", "value", "date"],
        "QUANDL": ["dataset_code", "dataset_name", "data", "column_names"]
    }

    ENDPOINTS = {
        "ALPHA_VANTAGE": {"prices": "/query?function=TIME_SERIES_DAILY"},
        "FRED": {"prices": "/series/observations"},
        "QUANDL": {"prices": "/datasets/WIKI/AAPL/data.json"}
    }

    AUTH_METHODS = {
        "ALPHA_VANTAGE": {"type": "query", "key_param": "apikey"},
        "FRED": {"type": "query", "key_param": "api_key"},
        "QUANDL": {"type": "query", "key_param": "api_key"}
    }

    def __init__(self, source="ALPHA_VANTAGE"):
        self.source = source.upper()

        # Load environment variables
        load_env()
        creds = get_api_credentials(self.source)
        self.api_key = creds["api_key"]
        self.base_url = creds["base_url"]

        # Setup endpoints and auth
        self.endpoints = self.ENDPOINTS.get(self.source, {})
        auth = self.AUTH_METHODS.get(self.source, {})
        self.auth_method = auth.get("type", "query")
        self.auth_key_param = auth.get("key_param", "apikey")

        # Expected schema for validation
        self.schema = self.SCHEMAS.get(self.source, [])

    def fetch(self, endpoint_key: str, params: dict = None):
        if endpoint_key not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_key} not configured for {self.source}")

        url = f"{self.base_url}{self.endpoints[endpoint_key]}"
        params = params or {}

        if self.auth_method == "query":
            params[self.auth_key_param] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out.")
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"HTTP error: {e}")

    def validate_schema(self, response):
        """
        Validate the response structure against expected schema for the API.
        """
        if isinstance(response, dict) and "data" in response:
            # Handle nested data (e.g., Quandl)
            data = response["data"][0] if response["data"] else {}
        elif isinstance(response, list):
            data = response[0] if response else {}
        else:
            data = response

        missing = [field for field in self.schema if field not in data]
        if missing:
            raise ValueError(f"Missing fields in response: {missing}")
        return True

# ========================
# STANDALONE TEST
# ========================
if __name__ == "__main__":
    for source in ["ALPHA_VANTAGE", "FRED", "QUANDL"]:
        print(f"\nTesting {source} API...")
        client = APIClient(source)
        try:
            # Example params — modify for actual API
            params = {"symbol": "AAPL"} if source == "ALPHA_VANTAGE" else {"series_id": "GDP"} if source=="FRED" else {}
            resp = client.fetch("prices", params=params)
            print("✅ Fetch successful")
            client.validate_schema(resp)
            print("✅ Schema validated")
        except Exception as e:
            print(f"❌ {source} test failed: {e}")
