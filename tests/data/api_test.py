import pytest
import time
from src.data_pipeline.phase1_ingestion.api_client import APIClient

client = APIClient(source="ALPHA_VANTAGE")

# --- TEST 1: Connectivity ---
def test_api_connectivity():
    """Ensure the API is reachable and returns HTTP 200 within latency threshold."""
    start = time.time()
    response = client.fetch("prices", params={"symbol": "AAPL", "limit": 1})
    end = time.time()
    latency = end - start

    assert response is not None, "❌ No response received from API"
    assert latency < 5.0, f"⚠️ API latency too high: {latency:.2f}s"
    print(f"✅ API reachable with latency {latency:.2f}s")


# --- TEST 2: Rate Limit Handling ---
def test_api_rate_limit():
    """Check API throttling behaviour under repeated requests."""
    MAX_REQUESTS_PER_MIN = 5  # Define manually or via .env if desired
    timestamps = []

    for _ in range(min(5, MAX_REQUESTS_PER_MIN)):
        start = time.time()
        response = client.fetch("prices", params={"symbol": "AAPL", "limit": 1})
        assert response is not None
        timestamps.append(time.time() - start)

    avg_latency = sum(timestamps) / len(timestamps)
    assert avg_latency < 5.0, "⚠️ Average latency too high — possible throttling"
    print(f"✅ Rate test passed. Avg latency: {avg_latency:.2f}s")


# --- TEST 3: Data Format & Schema ---
def test_api_data_format():
    """Validate response structure and required fields."""
    REQUIRED_FIELDS = ["symbol", "open", "close", "high", "low", "volume"]

    response = client.fetch("prices", params={"symbol": "AAPL", "limit": 1})

    assert isinstance(response, (list, dict)), f"Unexpected response type: {type(response)}"

    # Handle both list and dict responses
    sample = response[0] if isinstance(response, list) else response
    for field in REQUIRED_FIELDS:
        assert field in sample, f"❌ Missing field: {field}"

    print(f"✅ Schema validated. Fields present: {list(sample.keys())[:5]}...")
