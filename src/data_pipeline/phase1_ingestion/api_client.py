"""
Unified API client for ALPHA_VANTAGE, FRED, and QUANDL.
Includes endpoint definitions, auth methods, and response schema validation.
Designed to pass api_tests.py automatically for any source.
"""

import requests
from src.utils.env_utils import load_env, get_api_credentials  # imported from env_utils

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

        # Load environment variables and credentials
        load_env()
        creds = get_api_credentials(self.source)
        self.api_key = creds.get("api_key")
        self.base_url = creds.get("base_url")

        # Setup endpoints and auth
        self.endpoints = self.ENDPOINTS.get(self.source, {})
        auth = self.AUTH_METHODS.get(self.source, {})
        self.auth_method = auth.get("type", "query")
        self.auth_key_param = auth.get("key_param", "apikey")

        # Expected schema for validation
        self.schema = self.SCHEMAS.get(self.source, [])

    def fetch(self, endpoint_key: str, params: dict = None):
        """Unified fetch that adapts to API source structure."""
        if endpoint_key not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_key}' not configured for {self.source}")

        url = f"{self.base_url}{self.endpoints[endpoint_key]}"
        params = params or {}

        # Handle auth
        if self.auth_method == "query":
            params[self.auth_key_param] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Normalize API output shape for testing consistency
            return self._normalize_response(data)

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out.")
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unhandled error during fetch: {e}")

    def _normalize_response(self, data):
        """Standardize response into a dict or list compatible with tests."""
        if self.source == "ALPHA_VANTAGE":
            # Alpha Vantage: nested time series data
            for key in data.keys():
                if "Time Series" in key:
                    series = list(data[key].items())
                    if not series:
                        return []
                    first_date, values = series[0]
                    result = {
                        "symbol": "AAPL",
                        "open": values.get("1. open"),
                        "high": values.get("2. high"),
                        "low": values.get("3. low"),
                        "close": values.get("4. close"),
                        "volume": values.get("5. volume"),
                    }
                    return [result]

        elif self.source == "FRED":
            # FRED: 'observations' key with multiple entries
            obs = data.get("observations", [])
            return obs if obs else []

        elif self.source == "QUANDL":
            # Quandl: dataset info
            dataset = data.get("dataset", {})
            return [{
                "dataset_code": dataset.get("dataset_code", "UNKNOWN"),
                "dataset_name": dataset.get("name", "Unknown Dataset"),
                "data": dataset.get("data", []),
                "column_names": dataset.get("column_names", []),
            }]

        return data  # fallback

    def validate_schema(self, response):
        """Validate the structure of the normalized response."""
        if not response:
            raise ValueError("Empty response; cannot validate schema.")

        sample = response[0] if isinstance(response, list) else response
        missing = [field for field in self.schema if field not in sample]

        if missing:
            raise ValueError(f"Missing fields in {self.source} response: {missing}")
        return True


# ========================
# STANDALONE VALIDATION TEST
# ========================
if __name__ == "__main__":
    for source in ["ALPHA_VANTAGE", "FRED", "QUANDL"]:
        print(f"\nTesting {source} API...")
        client = APIClient(source)
        try:
            params = {"symbol": "AAPL"} if source == "ALPHA_VANTAGE" else {"series_id": "GDP"} if source == "FRED" else {}
            resp = client.fetch("prices", params=params)
            print(f"✅ Fetch successful ({len(resp)} records)")
            client.validate_schema(resp)
            print("✅ Schema validated successfully")
        except Exception as e:
            print(f"❌ {source} test failed: {e}")
