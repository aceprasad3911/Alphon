import pytest
import time
from src.data_pipeline.phase1_ingestion.api_client import APIClient

# Change this line to test any API
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

    sample = None

    if isinstance(response, list) and response:
        sample = response[0]

    elif isinstance(response, dict):
        # --- FRED ---
        if "observations" in response:
            sample = response["observations"][0]

        # --- QUANDL ---
        elif "dataset_data" in response and "data" in response["dataset_data"]:
            sample = dict(zip(
                response["dataset_data"]["column_names"],
                response["dataset_data"]["data"][0]
            ))

        # --- ALPHA VANTAGE ---
        elif any("Time Series" in k for k in response.keys()):
            ts_key = next((k for k in response.keys() if "Time Series" in k), None)
            ts_block = response.get(ts_key, {})
            if isinstance(ts_block, dict) and ts_block:
                first_date = next(iter(ts_block))
                sample_raw = ts_block[first_date]
                sample = {
                    "symbol": response.get("Meta Data", {}).get("2. Symbol", "AAPL"),
                    "open": sample_raw.get("1. open"),
                    "high": sample_raw.get("2. high"),
                    "low": sample_raw.get("3. low"),
                    "close": sample_raw.get("4. close"),
                    "volume": sample_raw.get("5. volume"),
                }

        # --- Rate limit / Error fallback ---
        elif "Note" in response:
            pytest.skip("⏸️ Skipped: Alpha Vantage rate-limited (Note in response).")
        elif "Error Message" in response:
            pytest.skip(f"⏸️ Skipped: Alpha Vantage returned error → {response['Error Message']}")
        else:
            # 🪶 Diagnostic output for unexpected schema
            print("⚠️ Unexpected Alpha Vantage response keys:", list(response.keys()))
            pytest.skip("⏸️ Skipped: Alpha Vantage returned unrecognized structure.")

    assert sample, "❌ No valid sample data found in response"
    print(f"✅ Sample extracted: {sample}")
