import pytest
import time
from src.data_pipeline.phase1_ingestion.api_client import APIClient

# Change the source here (ALPHA_VANTAGE, FRED, or QUANDL)
client = APIClient(source="ALPHA_VANTAGE")

# --- TEST 1: Connectivity ---
def test_api_connectivity():
    """Ensure the API is reachable and returns HTTP 200 within latency threshold."""
    start = time.time()
    params = {"symbol": "AAPL"} if client.source == "ALPHA_VANTAGE" else {"series_id": "GDP"}
    response = client.fetch("prices", params=params)
    end = time.time()
    latency = end - start

    assert response is not None, "❌ No response received from API"
    assert latency < 5.0, f"⚠️ API latency too high: {latency:.2f}s"
    print(f"✅ API reachable with latency {latency:.2f}s")


# --- TEST 2: Rate Limit Handling ---
def test_api_rate_limit():
    """Check API throttling behaviour under repeated requests."""
    MAX_REQUESTS_PER_MIN = 5
    timestamps = []
    params = {"symbol": "AAPL"} if client.source == "ALPHA_VANTAGE" else {"series_id": "GDP"}

    for _ in range(min(5, MAX_REQUESTS_PER_MIN)):
        start = time.time()
        response = client.fetch("prices", params=params)
        assert response is not None
        timestamps.append(time.time() - start)

    avg_latency = sum(timestamps) / len(timestamps)
    assert avg_latency < 5.0, "⚠️ Average latency too high — possible throttling"
    print(f"✅ Rate test passed. Avg latency: {avg_latency:.2f}s")


# --- TEST 3: Data Format & Schema ---
def test_api_data_format():
    """Validate response structure dynamically based on API schema."""
    response = client.fetch(
        "prices",
        params={"symbol": "AAPL"} if client.source == "ALPHA_VANTAGE" else {"series_id": "GDP"}
    )

    assert isinstance(response, (list, dict)), f"Unexpected response type: {type(response)}"

    # Extract sample data depending on structure
    sample = None
    if isinstance(response, list) and response:
        sample = response[0]
    elif isinstance(response, dict):
        # handle Quandl and FRED formats
        if "observations" in response:  # FRED
            sample = response["observations"][0]
        elif "dataset_data" in response and "data" in response["dataset_data"]:  # Quandl
            sample = dict(zip(response["dataset_data"]["column_names"], response["dataset_data"]["data"][0]))
        else:
            sample = response

    assert sample, "❌ No valid sample data found in response"

    # Dynamically use schema from the API client
    required_fields = client.schema
    for field in required_fields:
        assert field in sample, f"❌ Missing field: {field}"

    print(f"✅ Schema validated for {client.source}. Fields present: {list(sample.keys())[:5]}...")
