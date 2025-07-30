# src/data_sources/base.py

# Abstract base class for all data source handlers

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """
    Abstract base class for all data sources.
    Defines the common interface for fetching and normalizing financial data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the data source with its specific configuration.
        Args:
            config (Dict[str, Any]): Configuration dictionary for the data source.
        """
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__} with config: {config}")

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Abstract method to fetch raw data from the source.
        Specific parameters will vary by data source (e.g., symbol, start_date, end_date).
        Returns:
            pd.DataFrame: Raw data fetched from the source.
        """
        pass

    @abstractmethod
    def normalize(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to normalize raw data into a consistent format.
        This typically involves renaming columns, setting datetime index, and basic type conversions.
        Args:
            raw_data (pd.DataFrame): The raw data fetched from the source.
        Returns:
            pd.DataFrame: Normalized data with consistent column names and format.
        """
        pass

    def get_data(self, **kwargs) -> pd.DataFrame:
        """
        Combines fetching and normalizing data.
        Args:
            **kwargs: Arguments to pass to the fetch method.
        Returns:
            pd.DataFrame: Normalized data.
        """
        try:
            raw_data = self.fetch(**kwargs)
            normalized_data = self.normalize(raw_data)
            logger.info(f"Successfully fetched and normalized data from {self.__class__.__name__}.")
            return normalized_data
        except Exception as e:
            logger.error(f"Error fetching or normalizing data from {self.__class__.__name__}: {e}")
            raise

# TODO: Consider adding a rate-limiting mechanism here if it's common across many sources.
# TODO: Add a method for saving raw data to the data/raw directory.
