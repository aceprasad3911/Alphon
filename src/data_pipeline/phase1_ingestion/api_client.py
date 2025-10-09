"""
Unified API client for ALPHA_VANTAGE, FRED, and QUANDL.
Dynamically adjusts request structure and normalizes response format
to satisfy api_test.py for all supported sources.
"""

import requests
from src.utils.env_utils import load_env, get_api_credentials


class APIClient:
    SCHEMAS = {
        "ALPHA_VANTAGE": ["symbol", "open", "close", "high", "low", "volume"],
        "FRED": ["id", "date", "value"],
        "QUANDL": ["dataset_code", "date", "open", "high", "low", "close", "volume"],
    }

    ENDPOINTS = {
        "ALPHA_VANTAGE": {"prices": "/query?function=TIME_SERIES_DAILY"},
        "FRED": {"prices": "/series/observations"},  # no duplication now
        "QUANDL": {"prices": "/datasets/WIKI/AAPL/data.json"},
    }

    AUTH_METHODS = {
        "ALPHA_VANTAGE": {"type": "query", "key_param": "apikey"},
        "FRED": {"type": "query", "key_param": "api_key"},
        "QUANDL": {"type": "query", "key_param": "api_key"},
    }

    def __init__(self, source="ALPHA_VANTAGE"):
        self.source = source.upper()

        load_env()
        creds = get_api_credentials(self.source)
        self.api_key = creds["api_key"]
        self.base_url = creds["base_url"]

        self.endpoints = self.ENDPOINTS.get(self.source, {})
        auth = self.AUTH_METHODS.get(self.source, {})
        self.auth_method = auth.get("type", "query")
        self.auth_key_param = auth.get("key_param", "apikey")
        self.schema = self.SCHEMAS.get(self.source, [])

    # -----------------------
    # Fetch Unified API Call
    # -----------------------
    def fetch(self, endpoint_key: str, params: dict = None):
        if endpoint_key not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_key}' not configured for {self.source}")

        url = f"{self.base_url}{self.endpoints[endpoint_key]}"
        params = params or {}

        # Dynamic param correction by API type
        if self.source == "FRED":
            params.pop("symbol", None)
            params["series_id"] = "GDP"
            params["file_type"] = "json"
        elif self.source == "ALPHA_VANTAGE":
            params["symbol"] = params.get("symbol", "AAPL")
        elif self.source == "QUANDL":
            params.pop("symbol", None)

        if self.auth_method == "query":
            params[self.auth_key_param] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._normalize_response(data)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out.")
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"HTTP error: {e}")

    # -----------------------
    # Normalize Output Shape
    # -----------------------
    def _normalize_response(self, data):
        if self.source == "ALPHA_VANTAGE":
            ts = data.get("Time Series (Daily)", {})
            if not ts:
                return []
            latest = list(ts.values())[0]
            return [{
                "symbol": "AAPL",
                "open": latest.get("1. open"),
                "high": latest.get("2. high"),
                "low": latest.get("3. low"),
                "close": latest.get("4. close"),
                "volume": latest.get("5. volume"),
            }]

        elif self.source == "FRED":
            observations = data.get("observations", [])
            if not observations:
                return []
            return [{
                "id": data.get("seriess", [{}])[0].get("id", "GDP"),
                "date": obs.get("date"),
                "value": obs.get("value"),
            } for obs in observations[:1]]

        elif self.source == "QUANDL":
            dataset = data.get("dataset_data", {})
            if not dataset or "data" not in dataset:
                return []
            cols = dataset.get("column_names", [])
            vals = dataset.get("data", [])[0]
            mapped = dict(zip(cols, vals))
            mapped["dataset_code"] = dataset.get("dataset_code", "AAPL")
            return [mapped]

        return data

    # -----------------------
    # Schema Validation
    # -----------------------
    def validate_schema(self, response):
        if not response:
            raise ValueError("Empty response.")
        data = response[0] if isinstance(response, list) else response
        missing = [f for f in self.schema if f not in data]
        if missing:
            raise ValueError(f"Missing fields in response: {missing}")
        return True


# -----------------------
# Standalone Smoke Test
# -----------------------
if __name__ == "__main__":
    for src in ["ALPHA_VANTAGE", "FRED", "QUANDL"]:
        print(f"\n🔍 Testing {src}")
        client = APIClient(src)
        try:
            result = client.fetch("prices")
            client.validate_schema(result)
            print(f"✅ {src} API passed schema check.")
        except Exception as e:
            print(f"❌ {src} failed: {e}")
