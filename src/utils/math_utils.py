# src/utils/math_utils.py

# Custom mathematical functions (e.g., for fractional differencing)

import numpy as np
import pandas as pd
import logging
from typing import Union, List

logger = logging.getLogger(__name__)


def calculate_hurst_exponent(series: Union[pd.Series, np.ndarray], lags: List[int] = None) -> float:
    """
    Calculates the Hurst Exponent for a time series using Rescaled Range (R/S) analysis.
    A value near 0.5 indicates a random walk, >0.5 indicates persistence, <0.5 indicates anti-persistence.
    Args:
        series (Union[pd.Series, np.ndarray]): The input time series.
        lags (List[int]): List of lags to use for R/S calculation. If None, defaults to a range.
    Returns:
        float: The calculated Hurst Exponent. Returns NaN if calculation fails.
    """
    if isinstance(series, pd.Series):
        series = series.values

    if len(series) < 10:  # Minimum length for meaningful calculation
        logger.warning("Series too short for Hurst Exponent calculation. Returning NaN.")
        return np.nan

    # Calculate the differences
    diff_series = np.diff(series)
    if len(diff_series) == 0:
        logger.warning("Differenced series is empty. Returning NaN for Hurst Exponent.")
        return np.nan

    # Define lags if not provided
    if lags is None:
        max_lag = int(len(diff_series) / 2)
        if max_lag < 2:
            logger.warning("Not enough data points for specified lags. Returning NaN for Hurst Exponent.")
            return np.nan
        lags = list(range(2, max_lag + 1))

    rs_values = []
    for lag in lags:
        if lag == 0: continue  # Avoid division by zero
        # Divide series into segments of length 'lag'
        num_segments = len(diff_series) // lag
        if num_segments == 0: continue

        segments = np.array_split(diff_series[:num_segments * lag], num_segments)

        for segment in segments:
            if len(segment) < 2: continue  # Need at least 2 points for std dev

            # Calculate mean-adjusted series
            mean_adjusted = segment - np.mean(segment)

            # Calculate cumulative sum
            cumulative_sum = np.cumsum(mean_adjusted)

            # Calculate range (max - min)
            range_val = np.max(cumulative_sum) - np.min(cumulative_sum)

            # Calculate standard deviation
            std_dev = np.std(segment)

            if std_dev > 0:
                rs_values.append(range_val / std_dev)
            else:
                rs_values.append(np.nan)  # Avoid division by zero

    rs_values = np.array(rs_values)
    rs_values = rs_values[~np.isnan(rs_values)]  # Remove NaNs

    if len(rs_values) < 2:
        logger.warning("Not enough valid R/S values to compute Hurst Exponent. Returning NaN.")
        return np.nan

    try:
        # Log-log plot of R/S vs. lag
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)

        # Fit a linear regression to find the slope (Hurst Exponent)
        # Filter out inf/-inf values that can arise from log(0) or log(small_number)
        valid_indices = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid_indices) < 2:
            logger.warning("Not enough finite log values for Hurst Exponent regression. Returning NaN.")
            return np.nan

        poly_fit = np.polyfit(log_lags[valid_indices], log_rs[valid_indices], 1)
        hurst_exponent = poly_fit[0]
        logger.debug(f"Hurst Exponent calculated: {hurst_exponent:.4f}")
        return hurst_exponent
    except Exception as e:
        logger.error(f"Error during Hurst Exponent regression: {e}. Returning NaN.")
        return np.nan

# TODO: Implement fractional differencing (requires more complex algorithms).
# TODO: Add functions for statistical moments (skewness, kurtosis) if not using scipy.stats.
# TODO: Add functions for signal processing (e.g., detrending, smoothing).
