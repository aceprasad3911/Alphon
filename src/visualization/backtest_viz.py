# src/visualization/backtest_viz.py

# Visualizations for backtest results (equity curves, drawdowns)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def plot_equity_curve(equity_curve: pd.Series,
                      benchmark_curve: Optional[pd.Series] = None,
                      title: str = "Portfolio Equity Curve",
                      save_path: Optional[str] = None):
    """
    Plots the portfolio's equity curve over time.
    Args:
        equity_curve (pd.Series): Series of portfolio total value with DatetimeIndex.
        benchmark_curve (Optional[pd.Series]): Series of benchmark total value with DatetimeIndex.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot.
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty. Cannot plot equity curve.")
        return
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        logger.error("Equity curve must have a DatetimeIndex.")
        return

    plt.figure(figsize=(15, 7))
    plt.plot(equity_curve.index, equity_curve, label='Portfolio', color='blue')

    if benchmark_curve is not None:
        if not isinstance(benchmark_curve.index, pd.DatetimeIndex):
            logger.warning("Benchmark curve must have a DatetimeIndex. Skipping benchmark plot.")
        else:
            # Align benchmark to portfolio's start value for comparison
            aligned_benchmark = benchmark_curve.reindex(equity_curve.index, method='ffill').dropna()
            if not aligned_benchmark.empty:
                aligned_benchmark = aligned_benchmark / aligned_benchmark.iloc[0] * equity_curve.iloc[0]
                plt.plot(aligned_benchmark.index, aligned_benchmark, label='Benchmark', color='orange', linestyle='--')
            else:
                logger.warning("Aligned benchmark curve is empty. Skipping benchmark plot.")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Equity curve plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_drawdown(equity_curve: pd.Series,
                  title: str = "Portfolio Drawdown",
                  save_path: Optional[str] = None):
    """
    Plots the portfolio's drawdown over time.
    Args:
        equity_curve (pd.Series): Series of portfolio total value with DatetimeIndex.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot.
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty. Cannot plot drawdown.")
        return
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        logger.error("Equity curve must have a DatetimeIndex.")
        return

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1

    plt.figure(figsize=(15, 7))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.5)
    plt.plot(drawdown.index, drawdown, color='red')

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Format as percentage
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Drawdown plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# TODO: Add functions for plotting rolling Sharpe ratio.
# TODO: Add functions for plotting daily returns distribution.
# TODO: Add functions for plotting trade volume/frequency.
