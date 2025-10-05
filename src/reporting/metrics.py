# src/reporting/metrics.py

# Functions for calculating key performance metrics (Sharpe, Drawdown, IC, etc.)

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def calculate_performance_metrics(equity_curve: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculates a comprehensive set of performance metrics for an equity curve.
    Args:
        equity_curve (pd.Series): A pandas Series representing the portfolio's total value over time.
                                  Index must be DatetimeIndex.
        benchmark_returns (Optional[pd.Series]): Daily returns of a benchmark index.
                                                 Index must be DatetimeIndex, aligned with equity_curve.
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).
    Returns:
        Dict[str, Any]: A dictionary of calculated performance metrics.
    """
    if equity_curve.empty:
        logger.warning("Equity curve is empty. Cannot calculate performance metrics.")
        return {}

    # Ensure equity curve is numeric and has DatetimeIndex
    equity_curve = pd.to_numeric(equity_curve, errors='coerce').dropna()
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        try:
            equity_curve.index = pd.to_datetime(equity_curve.index)
        except Exception as e:
            logger.error(f"Equity curve index could not be converted to DatetimeIndex: {e}")
            return {}
    equity_curve = equity_curve.sort_index()

    # Calculate daily returns
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        logger.warning("Returns series is empty after calculating. Cannot calculate performance metrics.")
        return {}

    # Annualization factor (assuming daily data)
    trading_days_per_year = 252  # Common for equities
    if returns.index.freq == 'W':
        trading_days_per_year = 52
    elif returns.index.freq == 'M':
        trading_days_per_year = 12
    elif returns.index.freq == 'Q':
        trading_days_per_year = 4
    elif returns.index.freq == 'A':
        trading_days_per_year = 1
    # TODO: More robust frequency detection or pass as argument

    # Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualized Return
    num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else np.nan

    # Annualized Volatility (Standard Deviation of Returns)
    annualized_volatility = returns.std() * np.sqrt(trading_days_per_year)

    # Sharpe Ratio
    # (Annualized Portfolio Return - Annualized Risk-Free Rate) / Annualized Portfolio Volatility
    excess_returns = returns - (risk_free_rate / trading_days_per_year)
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(
        trading_days_per_year) if returns.std() > 0 else np.nan

    # Maximum Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown * 100

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else np.nan

    # Sortino Ratio
    # (Annualized Portfolio Return - Annualized Risk-Free Rate) / Annualized Downside Volatility
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(
        trading_days_per_year) if not downside_returns.empty else np.nan
    sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(
        trading_days_per_year) if downside_volatility > 0 else np.nan

    # VaR (Value at Risk) - 95% confidence
    var_95 = returns.quantile(0.05)  # Daily VaR

    # CVaR (Conditional Value at Risk) - 95% confidence
    cvar_95 = returns[returns <= var_95].mean()  # Daily CVaR

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar_ratio": calmar_ratio,
        "sortino_ratio": sortino_ratio,
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "num_trading_days": len(returns),
        "start_date": equity_curve.index.min().strftime('%Y-%m-%d'),
        "end_date": equity_curve.index.max().strftime('%Y-%m-%d')
    }

    # Metrics requiring a benchmark
    if benchmark_returns is not None:
        # Align benchmark returns with portfolio returns
        aligned_benchmark_returns = benchmark_returns.reindex(returns.index).dropna()
        aligned_returns = returns.reindex(aligned_benchmark_returns.index).dropna()

        if not aligned_benchmark_returns.empty and not aligned_returns.empty:
            # Beta
            covariance = aligned_returns.cov(aligned_benchmark_returns)
            benchmark_variance = aligned_benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan

            # Alpha (Jensen's Alpha)
            # Alpha = Portfolio Return - (Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate))
            alpha = (aligned_returns.mean() - (risk_free_rate / trading_days_per_year) -
                     beta * (aligned_benchmark_returns.mean() - (
                                risk_free_rate / trading_days_per_year))) * trading_days_per_year

            # Tracking Error
            tracking_error = (aligned_returns - aligned_benchmark_returns).std() * np.sqrt(trading_days_per_year)

            metrics.update({
                "beta": beta,
                "alpha": alpha,
                "tracking_error": tracking_error
            })
        else:
            logger.warning("Benchmark returns could not be aligned or are empty. Skipping benchmark-dependent metrics.")

    logger.info("Performance metrics calculated.")
    return metrics

# TODO: Add turnover calculation (requires trade log).
# TODO: Add win rate, profit factor (requires trade log).
# TODO: Implement Information Coefficient (IC) calculation (requires signals and future returns).
