# src/data_sources/yfinance.py

# Handles data fetching for YFinance API

import yfinance as yf
import pandas as pd
import logging
from typing import Dict, Any
from .base import DataSource

logger = logging.getLogger(__name__)

class YahooFinanceSource(DataSource):
    """
    Data source handler for Yahoo Finance using the yfinance library.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.default_symbols = self.config.get("default_symbols", ["AAPL"])
        self.default_start_date = self.config.get("default_start_date", "2010-01-01")
        self.default_end_date = self.config.get("default_end_date", "2023-12-31")
        self.data_frequency = self.config.get("data_frequency", "daily")

    def fetch(self, symbol: str = None, start_date: str = None, end_date: str = None,
              interval: str = None) -> pd.DataFrame:
        """
        Fetches historical market data from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol (e.g., "AAPL"). Defaults to config.
            start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to config.
            end_date (str): End date in 'YYYY-MM-DD' format. Defaults to config.
            interval (str): Data interval (e.g., "1d", "1wk", "1mo"). Defaults to config.
        Returns:
            pd.DataFrame: Raw OHLCV data.
        """
        symbol = symbol if symbol is not None else self.default_symbols[0]
        start_date = start_date if start_date is not None else self.default_start_date
        end_date = end_date if end_date is not None else self.default_end_date
        interval = interval if interval is not None else self._map_frequency_to_interval(self.data_frequency)

        logger.info(f"Fetching {symbol} from Yahoo Finance from {start_date} to {end_date} at {interval} interval.")
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            if data.empty:
                logger.warning(f"No data found for {symbol} from Yahoo Finance.")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol} from Yahoo Finance: {e}")
            raise

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes Yahoo Finance data to a consistent format.
        Renames columns to lowercase and sets 'Date' as datetime index.
        Args:
            raw_data (pd.DataFrame): Raw data from yfinance.download().
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if raw_data.empty:
            return pd.DataFrame()

        # Rename columns to a consistent lowercase format
        raw_data.columns = [col.lower().replace(' ', '_') for col in raw_data.columns]

        # Ensure 'date' is the index and is datetime type
        if 'date' in raw_data.columns:
            raw_data = raw_data.set_index('date')
        raw_data.index = pd.to_datetime(raw_data.index)

        # Select and reorder common columns
        common_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        normalized_data = raw_data[[col for col in common_cols if col in raw_data.columns]]

        logger.debug("Yahoo Finance data normalized.")
        return normalized_data

    def _map_frequency_to_interval(self, frequency: str) -> str:
        """Maps a general frequency string to yfinance interval string."""
        if frequency == "daily":
            return "1d"
        elif frequency == "weekly":
            return "1wk"
        elif frequency == "monthly":
            return "1mo"
        # TODO: Add more mappings if needed (e.g., intraday)
        else:
            logger.warning(f"Unsupported frequency: {frequency}. Defaulting to '1d'.")
            return "1d"

# TODO: Add error handling for specific yfinance exceptions (e.g., network issues).
# TODO: Implement a method to fetch multiple symbols efficiently.
# TODO: Consider caching fetched data to avoid repeated API calls.
