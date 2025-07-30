# src/utils/date_utils.py

# Helper functions for date and time manipulations

import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Union

logger = logging.getLogger(__name__)


def get_trading_days_in_range(start_date: Union[str, datetime],
                              end_date: Union[str, datetime],
                              freq: str = 'B') -> pd.DatetimeIndex:
    """
    Generates a DatetimeIndex of trading days within a specified range.
    Args:
        start_date (Union[str, datetime]): The start date (inclusive).
        end_date (Union[str, datetime]): The end date (inclusive).
        freq (str): Frequency string for pandas date_range (e.g., 'B' for business days, 'D' for calendar days).
    Returns:
        pd.DatetimeIndex: A DatetimeIndex of trading days.
    """
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        trading_days = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        logger.debug(f"Generated {len(trading_days)} trading days from {start_date} to {end_date} with freq {freq}.")
        return trading_days
    except Exception as e:
        logger.error(f"Error generating trading days for range {start_date} to {end_date}: {e}")
        raise


def get_last_n_trading_days(end_date: Union[str, datetime], n: int, freq: str = 'B') -> pd.DatetimeIndex:
    """
    Generates a DatetimeIndex of the last N trading days up to an end date.
    Args:
        end_date (Union[str, datetime]): The end date (inclusive).
        n (int): The number of trading days to retrieve.
        freq (str): Frequency string for pandas date_range (e.g., 'B' for business days).
    Returns:
        pd.DatetimeIndex: A DatetimeIndex of the last N trading days.
    """
    try:
        end_dt = pd.to_datetime(end_date)
        # Generate more days than needed to ensure we get 'n' business days
        # A buffer of 2*n is usually safe for business days
        start_buffer = end_dt - timedelta(days=n * 2)
        all_days = pd.date_range(start=start_buffer, end=end_dt, freq=freq)

        if len(all_days) < n:
            # If still not enough, extend the buffer
            start_buffer = end_dt - timedelta(days=n * 3)
            all_days = pd.date_range(start=start_buffer, end=end_dt, freq=freq)

        trading_days = all_days[-n:]
        logger.debug(f"Generated last {n} trading days ending {end_date} with freq {freq}.")
        return trading_days
    except Exception as e:
        logger.error(f"Error generating last {n} trading days ending {end_date}: {e}")
        raise

# TODO: Add functions for converting between different time frequencies (e.g., daily to weekly).
# TODO: Implement a function to check if a given date is a trading holiday (requires holiday calendar data).
