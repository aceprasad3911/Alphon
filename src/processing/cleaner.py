# src/processing/cleaner.py

# Contains functions for data cleaning, temporal alignment, outlier handling

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame,
               fill_method: str = 'ffill',
               outlier_threshold: float = 3.0,
               columns_to_clean: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Performs basic data cleaning on a DataFrame.
    Includes handling missing values and basic outlier detection.

    Args:
        df (pd.DataFrame): Input DataFrame to clean. Expected to have a DatetimeIndex.
        fill_method (str): Method for filling missing values ('ffill', 'bfill', 'mean', 'median', 'interpolate').
        outlier_threshold (float): Z-score threshold for outlier detection. Outliers are replaced with NaN.
        columns_to_clean (Optional[List[str]]): Specific columns to apply cleaning to. If None, applies to all numeric.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        logger.warning("Attempted to clean an empty DataFrame.")
        return pd.DataFrame()

    cleaned_df = df.copy()

    # Ensure index is datetime
    if not isinstance(cleaned_df.index, pd.DatetimeIndex):
        try:
            cleaned_df.index = pd.to_datetime(cleaned_df.index)
            cleaned_df = cleaned_df.sort_index()
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}")
            return df # Return original if index conversion fails

    # Handle missing values
    initial_na_count = cleaned_df.isnull().sum().sum()
    if initial_na_count > 0:
        logger.info(f"Handling {initial_na_count} missing values using '{fill_method}' method.")
        if fill_method == 'ffill':
            cleaned_df = cleaned_df.ffill()
        elif fill_method == 'bfill':
            cleaned_df = cleaned_df.bfill()
        elif fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'interpolate':
            cleaned_df = cleaned_df.interpolate(method='time') # Good for time series
        else:
            logger.warning(f"Unknown fill method '{fill_method}'. Skipping missing value imputation.")
        # After ffill/bfill/interpolate, there might still be NaNs at the start/end
        cleaned_df = cleaned_df.dropna(how='all') # Drop rows that are entirely NaN

    # Outlier detection and replacement (using Z-score)
    numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    cols_to_process = columns_to_clean if columns_to_clean is not None else numeric_cols

    for col in cols_to_process:
        if col in numeric_cols:
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            if std > 0:
                z_scores = np.abs((cleaned_df[col] - mean) / std)
                outliers_count = (z_scores > outlier_threshold).sum()
                if outliers_count > 0:
                    logger.info(f"Found {outliers_count} outliers in column '{col}' (Z-score > {outlier_threshold}). Replacing with NaN.")
                    cleaned_df.loc[z_scores > outlier_threshold, col] = np.nan
                    # After replacing with NaN, re-fill them
                    if fill_method == 'ffill':
                        cleaned_df[col] = cleaned_df[col].ffill()
                    elif fill_method == 'bfill':
                        cleaned_df[col] = cleaned_df[col].bfill()
                    elif fill_method == 'interpolate':
                        cleaned_df[col] = cleaned_df[col].interpolate(method='time')
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(mean) # Fallback to mean if other fill methods not suitable

    logger.info(f"Data cleaning complete. Remaining NaNs: {cleaned_df.isnull().sum().sum()}")
    return cleaned_df

def align_data(data_frames: Dict[str, pd.DataFrame],
               freq: str = 'D',
               how: str = 'outer',
               fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Aligns multiple DataFrames to a common datetime index and frequency.
    Args:
        data_frames (Dict[str, pd.DataFrame]): A dictionary of DataFrames, where keys are identifiers
                                               (e.g., 'prices', 'fundamentals', 'macro').
                                               Each DataFrame must have a DatetimeIndex.
        freq (str): Target frequency for alignment (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly).
        how (str): How to join the DataFrames ('inner', 'outer', 'left', 'right').
        fill_method (str): Method for filling NaNs introduced by reindexing/merging ('ffill', 'bfill', 'mean', 'median', 'interpolate').

    Returns:
        pd.DataFrame: A single, aligned DataFrame.
    """
    if not data_frames:
        logger.warning("No DataFrames provided for alignment.")
        return pd.DataFrame()

    # Ensure all DataFrames have DatetimeIndex and sort them
    for name, df in data_frames.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                data_frames[name] = df.sort_index()
            except Exception as e:
                logger.error(f"DataFrame '{name}' index could not be converted to DatetimeIndex: {e}")
                raise ValueError(f"DataFrame '{name}' must have a DatetimeIndex.")

    # Create a common date range
    min_date = min(df.index.min() for df in data_frames.values())
    max_date = max(df.index.max() for df in data_frames.values())
    full_date_range = pd.date_range(start=min_date, end=max_date, freq=freq)

    aligned_df = pd.DataFrame(index=full_date_range)

    for name, df in data_frames.items():
        # Reindex each DataFrame to the full date range
        reindexed_df = df.reindex(full_date_range)

        # Fill NaNs introduced by reindexing (e.g., for lower frequency data)
        if fill_method == 'ffill':
            reindexed_df = reindexed_df.ffill()
        elif fill_method == 'bfill':
            reindexed_df = reindexed_df.bfill()
        elif fill_method == 'mean':
            reindexed_df = reindexed_df.fillna(reindexed_df.mean(numeric_only=True))
        elif fill_method == 'median':
            reindexed_df = reindexed_df.fillna(reindexed_df.median(numeric_only=True))
        elif fill_method == 'interpolate':
            reindexed_df = reindexed_df.interpolate(method='time')
        else:
            logger.warning(f"Unknown fill method '{fill_method}' for alignment. NaNs may remain.")

        # Prefix columns to avoid name clashes
        prefixed_df = reindexed_df.add_prefix(f"{name}_")
        aligned_df = aligned_df.merge(prefixed_df, left_index=True, right_index=True, how=how)

    logger.info(f"Data alignment complete to frequency '{freq}' using '{how}' join. Final shape: {aligned_df.shape}")
    return aligned_df

# TODO: Implement corporate action adjustments (e.g., for non-adjusted prices if needed).
# TODO: Add more sophisticated outlier detection methods (e.g., IQR, Isolation Forest).
# TODO: Consider handling multi-index DataFrames for multiple assets.
