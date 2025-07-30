# src/data_sources/quandl.py

# Handles data fetching for Quandl API

import quandl
import pandas as pd
import logging
from typing import Dict, Any
from .base import DataSource

logger = logging.getLogger(__name__)

class QuandlSource(DataSource):
    """
    Data source handler for Quandl (Nasdaq Data Link) API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url") # Not directly used by quandl library, but good for config
        self.default_dataset_code = self.config.get("default_dataset_code", "WIKI/AAPL")
        self.data_frequency = self.config.get("data_frequency", "daily")

        if self.api_key:
            quandl.ApiConfig.api_key = self.api_key
        else:
            logger.warning("Quandl API key not found in config. Free tier limits apply.")

    def fetch(self, dataset_code: str = None, start_date: str = None, end_date: str = None,
              collapse: str = None, **kwargs) -> pd.DataFrame:
        """
        Fetches data from Quandl.
        Args:
            dataset_code (str): Quandl dataset code (e.g., "WIKI/AAPL"). Defaults to config.
            start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to None.
            end_date (str): End date in 'YYYY-MM-DD' format. Defaults to None.
            collapse (str): Data frequency (e.g., "daily", "weekly", "monthly"). Defaults to config.
            **kwargs: Additional parameters for quandl.get().
        Returns:
            pd.DataFrame: Raw data from Quandl.
        """
        dataset_code = dataset_code if dataset_code is not None else self.default_dataset_code
        collapse = collapse if collapse is not None else self.data_frequency

        logger.info(f"Fetching {dataset_code} from Quandl from {start_date} to {end_date} with collapse={collapse}.")
        try:
            data = quandl.get(
                dataset_code,
                start_date=start_date,
                end_date=end_date,
                collapse=collapse,
                **kwargs
            )
            if data.empty:
                logger.warning(f"No data found for {dataset_code} from Quandl.")
            return data
        except quandl.QuandlException as e:
            logger.error(f"Quandl API Error for {dataset_code}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch data for {dataset_code} from Quandl: {e}")
            raise

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes Quandl data to a consistent format.
        Args:
            raw_data (pd.DataFrame): Raw data from quandl.get().
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if raw_data.empty:
            return pd.DataFrame()

        # Quandl data usually has 'Date' as index and good column names
        raw_data.index = pd.to_datetime(raw_data.index)
        raw_data = raw_data.sort_index()

        # Rename columns to lowercase and replace spaces/special chars
        new_columns = {}
        for col in raw_data.columns:
            new_columns[col] = col.lower().replace(' ', '_').replace('.', '').replace('-', '_')
        normalized_data = raw_data.rename(columns=new_columns)

        # TODO: Identify common columns based on expected dataset types (e.g., 'adj_close' for WIKI)
        # For WIKI dataset, 'Adj. Close' is common
        if 'adj_close' in normalized_data.columns:
            normalized_data['close'] = normalized_data['adj_close'] # Create a 'close' column if only 'adj_close' exists

        logger.debug("Quandl data normalized.")
        return normalized_data

# TODO: Implement specific fetch methods for different Quandl datasets (e.g., economic data).
# TODO: Handle different column naming conventions for various Quandl datasets.
# TODO: Add more robust error handling for Quandl-specific exceptions.
