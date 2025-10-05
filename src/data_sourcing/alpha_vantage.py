# src/data_sourcing/alpha_vantage_api_notes.txt

# Handles data fetching for Alpha Vantage API

import requests
import pandas as pd
import logging
import time
from typing import Dict, Any
from .base import DataSource

logger = logging.getLogger(__name__)

class AlphaVantageSource(DataSource):
    """
    Data source handler for Alpha Vantage API.
    Handles API key, rate limiting, and different function calls.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url")
        self.data_frequency = self.config.get("data_frequency", "daily")
        self.output_size = self.config.get("output_size", "full")
        self.premium_tier = self.config.get("premium_tier", False)

        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found in config.")

        # Rate limiting: 5 calls per minute for free tier, 500 calls per day
        self.last_call_time = 0
        self.call_count_minute = 0
        self.call_count_day = 0
        self.minute_reset_time = time.time()
        self.day_reset_time = time.time()

    def _apply_rate_limit(self):
        """Applies rate limiting based on Alpha Vantage tier."""
        current_time = time.time()

        # Reset minute count
        if current_time - self.minute_reset_time >= 60:
            self.call_count_minute = 0
            self.minute_reset_time = current_time

        # Reset daily count (assuming daily reset at midnight UTC for simplicity)
        # TODO: Implement more robust daily reset logic if needed
        if current_time - self.day_reset_time >= 24 * 3600:
            self.call_count_day = 0
            self.day_reset_time = current_time

        if not self.premium_tier:
            # Free tier limits
            if self.call_count_minute >= 5:
                sleep_time = 60 - (current_time - self.minute_reset_time) + 1 # Wait until next minute
                logger.warning(f"Alpha Vantage minute limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                self.call_count_minute = 0 # Reset after sleeping
                self.minute_reset_time = time.time() # Reset timer
            if self.call_count_day >= 500:
                logger.error("Alpha Vantage daily limit reached. Cannot make more calls today.")
                raise Exception("Alpha Vantage daily API limit exceeded.")
        else:
            # TODO: Implement premium tier rate limits if applicable
            pass

        self.call_count_minute += 1
        self.call_count_day += 1
        self.last_call_time = current_time

    def fetch(self, function: str, symbol: str, **kwargs) -> pd.DataFrame:
        """
        Fetches data from Alpha Vantage for a given function and symbol.
        Args:
            function (str): Alpha Vantage API function (e.g., "TIME_SERIES_DAILY", "GLOBAL_QUOTE").
            symbol (str): Ticker symbol.
            **kwargs: Additional parameters specific to the function.
        Returns:
            pd.DataFrame: Raw data from Alpha Vantage.
        """
        self._apply_rate_limit()

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": self.output_size,
            **kwargs
        }

        logger.info(f"Fetching {function} for {symbol} from Alpha Vantage.")
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
                raise Exception(f"Alpha Vantage API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"Alpha Vantage API Note: {data['Note']}") # Often rate limit warnings

            # TODO: Handle different data structures based on 'function'
            if function == "TIME_SERIES_DAILY" or function == "TIME_SERIES_DAILY_ADJUSTED":
                key = "Time Series (Daily)"
                if key not in data:
                    logger.warning(f"Expected key '{key}' not found in Alpha Vantage response for {symbol}.")
                    return pd.DataFrame()
                df = pd.DataFrame.from_dict(data[key], orient='index', dtype=float)
            elif function == "GLOBAL_QUOTE":
                key = "Global Quote"
                if key not in data or not data[key]:
                    logger.warning(f"Expected key '{key}' not found or empty in Alpha Vantage response for {symbol}.")
                    return pd.DataFrame()
                df = pd.DataFrame([data[key]])
            else:
                logger.warning(f"Unsupported Alpha Vantage function: {function}. Returning raw JSON.")
                return pd.DataFrame(data) # Return as is, normalization will likely fail

            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Network or HTTP error fetching from Alpha Vantage: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol} from Alpha Vantage: {e}")
            raise

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes Alpha Vantage time series data.
        Args:
            raw_data (pd.DataFrame): Raw data from Alpha Vantage.
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if raw_data.empty:
            return pd.DataFrame()

        # For time series data, index is already date-like
        raw_data.index = pd.to_datetime(raw_data.index)
        raw_data = raw_data.sort_index() # Ensure chronological order

        # Rename columns to a consistent lowercase format
        # Example: '1. open' -> 'open', '5. adjusted close' -> 'adjusted_close'
        new_columns = {}
        for col in raw_data.columns:
            parts = col.split('. ')
            if len(parts) > 1:
                new_columns[col] = parts[1].lower().replace(' ', '_')
            else:
                new_columns[col] = col.lower().replace(' ', '_')
        normalized_data = raw_data.rename(columns=new_columns)

        # Select common columns
        common_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        normalized_data = normalized_data[[col for col in common_cols if col in normalized_data.columns]]

        logger.debug("Alpha Vantage data normalized.")
        return normalized_data

# TODO: Implement specific fetch methods for different Alpha Vantage functions (e.g., fundamental data).
# TODO: Add more robust error handling for different API response structures.
# TODO: Consider using a dedicated Alpha Vantage Python client library if available and more robust.
