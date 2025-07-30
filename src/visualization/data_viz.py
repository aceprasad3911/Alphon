# src/visualization/data_viz.py

# Visualizations for raw and processed data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

def plot_data_distribution(df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           title: str = "Data Distribution",
                           save_path: Optional[str] = None):
    """
    Plots histograms and KDEs for numerical columns in a DataFrame to visualize their distribution.
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (Optional[List[str]]): List of columns to plot. If None, plots all numeric columns.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot (e.g., "output/distribution.png"). If None, displays plot.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot plot data distribution.")
        return

    data_to_plot = df[columns] if columns else df.select_dtypes(include=['number'])

    if data_to_plot.empty:
        logger.warning("No numeric columns to plot for data distribution.")
        return

    num_cols = len(data_to_plot.columns)
    if num_cols == 0:
        logger.warning("No columns selected for plotting distribution.")
        return

    fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 4 * num_cols))
    if num_cols == 1: # Ensure axes is iterable even for single plot
        axes = [axes]

    for i, col in enumerate(data_to_plot.columns):
        sns.histplot(data_to_plot[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=16) # Adjust suptitle position

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Data distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_time_series(df: pd.DataFrame,
                     columns: Optional[List[str]] = None,
                     title: str = "Time Series Plot",
                     save_path: Optional[str] = None):
    """
    Plots selected time series columns from a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        columns (Optional[List[str]]): List of columns to plot. If None, plots all columns.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot (e.g., "output/time_series.png").
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot plot time series.")
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame must have a DatetimeIndex to plot time series.")
        return

    data_to_plot = df[columns] if columns else df

    if data_to_plot.empty:
        logger.warning("No columns selected or available for time series plotting.")
        return

    plt.figure(figsize=(15, 7))
    for col in data_to_plot.columns:
        plt.plot(data_to_plot.index, data_to_plot[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Time series plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# TODO: Add functions for plotting correlation matrices.
# TODO: Add functions for plotting rolling statistics (mean, std).
# TODO: Consider using Plotly for interactive plots.
