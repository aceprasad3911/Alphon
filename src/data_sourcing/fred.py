# src/data_sourcing/fred_api_calls.py

# Handles data fetching for FRED API

from fredapi import Fred
import pandas as pd
import logging
from typing import Dict, Any, List
from .base import DataSource

logger = logging.getLogger(__name__)


class FREDSource(DataSource):
    """
    Data source handler for FRED (Federal Reserve Economic Data) API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self.config.get("api_key")  # FRED API key is optional for some calls, but good practice
        self.default_series_ids = self.config.get("default_series_ids", ["GDP", "CPIAUCSL"])
        self.data_frequency = self.config.get("data_frequency",
                                              "daily")  # Not directly used by FRED, but for consistency

        if not self.api_key:
            logger.warning("FRED API key not found in config. Rate limits may apply.")
        self.fred = Fred(api_key=self.api_key)

    def fetch(self, series_id: str = None, start_date: str = None, end_date: str = None,
              observation_start: str = None, observation_end: str = None,
              frequency: str = None, **kwargs) -> pd.DataFrame:
        """
        Fetches economic data series from FRED.
        Args:
            series_id (str): FRED series ID (e.g., "GDP", "CPIAUCSL"). Defaults to config.
            start_date (str): Start date for observations (YYYY-MM-DD).
            end_date (str): End date for observations (YYYY-MM-DD).
            observation_start (str): Alias for start_date.
            observation_end (str): Alias for end_date.
            frequency (str): Data frequency (e.g., "d", "w", "m", "q", "a").
            **kwargs: Additional parameters for fredapi.Fred.get_series().
        Returns:
            pd.DataFrame: Raw data from FRED.
        """
        series_id = series_id if series_id is not None else self.default_series_ids[0]
        obs_start = observation_start or start_date
        obs_end = observation_end or end_date

        logger.info(f"Fetching series '{series_id}' from FRED from {obs_start} to {obs_end}.")
        try:
            data = self.fred.get_series(
                series_id,
                observation_start=obs_start,
                observation_end=obs_end,
                frequency=frequency,
                **kwargs
            )
            if data is None or data.empty:
                logger.warning(f"No data found for FRED series '{series_id}'.")
                return pd.DataFrame()

            # get_series returns a Series, convert to DataFrame
            df = data.to_frame(name=series_id)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for FRED series '{series_id}': {e}")
            raise

    def fetch_multiple_series(self, series_ids: List[str] = None, start_date: str = None, end_date: str = None,
                              **kwargs) -> pd.DataFrame:
        """
        Fetches multiple economic data series from FRED and combines them.
        Args:
            series_ids (List[str]): List of FRED series IDs. Defaults to config.
            start_date (str): Start date for observations (YYYY-MM-DD).
            end_date (str): End date for observations (YYYY-MM-DD).
            **kwargs: Additional parameters for fredapi.Fred.get_series().
        Returns:
            pd.DataFrame: Combined DataFrame of multiple series.
        """
        series_ids = series_ids if series_ids is not None else self.default_series_ids
        all_data = []
        for sid in series_ids:
            try:
                data = self.fetch(series_id=sid, start_date=start_date, end_date=end_date, **kwargs)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                logger.warning(f"Skipping series {sid} due to error: {e}")
                continue

        if not all_data:
            logger.warning("No data fetched for any of the specified FRED series.")
            return pd.DataFrame()

        # Combine all series, aligning by date index
        combined_df = pd.concat(all_data, axis=1)
        return self.normalize(combined_df)  # Normalize the combined DataFrame

    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes FRED data.
        Ensures datetime index and consistent column names.
        Args:
            raw_data (pd.DataFrame): Raw data from FRED.
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if raw_data.empty:
            return pd.DataFrame()

        # FRED data usually has datetime index already
        raw_data.index = pd.to_datetime(raw_data.index)
        raw_data = raw_data.sort_index()

        # Rename columns to lowercase
        normalized_data = raw_data.rename(columns=lambda x: x.lower())

        logger.debug("FRED data normalized.")
        return normalized_data

# TODO: Add methods to search for FRED series IDs.
# TODO: Implement more robust error handling for FRED API limits or invalid series IDs.
