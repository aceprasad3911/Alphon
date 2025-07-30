# src/reporting/report_generator.py

# Generates comprehensive backtest and performance reports

# TODO: Add user (y/n) decision to append report to reports folder

import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.visualization.backtest_viz import plot_equity_curve, plot_drawdown  # Assuming these exist
from src.reporting.metrics import calculate_performance_metrics

logger = logging.getLogger(__name__)


def generate_backtest_report(
        portfolio_history: pd.DataFrame,
        trade_log: pd.DataFrame,
        output_dir: str = "reports/",
        report_name: str = None,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "Alpha Strategy"
) -> str:
    """
    Generates a comprehensive backtest report including performance metrics and visualizations.
    Args:
        portfolio_history (pd.DataFrame): DataFrame with portfolio value history.
        trade_log (pd.DataFrame): DataFrame with trade details.
        output_dir (str): Directory to save the report and plots.
        report_name (str): Name of the report file (e.g., "my_strategy_report").
                           If None, a timestamped name will be used.
        benchmark_returns (Optional[pd.Series]): Daily returns of a benchmark index.
        strategy_name (str): Name of the strategy for the report title.
    Returns:
        str: Path to the generated report file.
    """
    if portfolio_history.empty:
        logger.error("Portfolio history is empty. Cannot generate report.")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    if report_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"backtest_report_{timestamp}"

    report_path = os.path.join(output_dir, f"{report_name}.txt")  # Using .txt for simplicity, could be .md or .pdf
    plots_dir = os.path.join(output_dir, f"{report_name}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(f"Generating backtest report: {report_path}")

    # 1. Calculate Performance Metrics
    metrics = calculate_performance_metrics(portfolio_history['total_value'], benchmark_returns=benchmark_returns)

    with open(report_path, 'w') as f:
        f.write(f"--- Backtest Report for {strategy_name} ---\n")
        f.write(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Backtest Period: {metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}\n")
        f.write("\n")

        f.write("--- Performance Metrics ---\n")
        for key, value in metrics.items():
            if isinstance(value, (float, np.float_)):
                if 'pct' in key or 'return' in key or 'drawdown' in key:
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2%}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        # 2. Generate Plots
        logger.info("Generating plots...")

        # Equity Curve
        equity_curve_path = os.path.join(plots_dir, "equity_curve.png")
        plot_equity_curve(portfolio_history['total_value'], title=f"{strategy_name} Equity Curve",
                          save_path=equity_curve_path)
        f.write(f"--- Visualizations ---\n")
        f.write(f"Equity Curve: {equity_curve_path}\n")

        # Drawdown Plot
        drawdown_path = os.path.join(plots_dir, "drawdown.png")
        plot_drawdown(portfolio_history['total_value'], title=f"{strategy_name} Drawdown", save_path=drawdown_path)
        f.write(f"Drawdown Plot: {drawdown_path}\n")

        # TODO: Add more plots:
        # - Daily Returns Distribution (Histogram)
        # - Rolling Sharpe Ratio
        # - Trade Volume over time
        # - Position sizing over time

        f.write("\n--- Trade Log Summary ---\n")
        if not trade_log.empty:
            f.write(f"Total Trades: {len(trade_log)}\n")
            f.write(f"Total Buy Trades: {len(trade_log[trade_log['type'] == 'BUY'])}\n")
            f.write(f"Total Sell Trades: {len(trade_log[trade_log['type'] == 'SELL'])}\n")
            f.write(f"Total Commission Paid: ${trade_log['commission'].sum():,.2f}\n")
            f.write(f"Total Slippage Cost: ${trade_log['slippage'].sum():,.2f}\n")
            f.write("\nSample of Trade Log:\n")
            f.write(trade_log.head().to_string())
            f.write("\n...\n")
            f.write(trade_log.tail().to_string())
        else:
            f.write("No trades recorded.\n")
        f.write("\n")

        f.write("--- End of Report ---\n")

    logger.info(f"Backtest report saved to {report_path}")
    return report_path

# TODO: Integrate with a PDF generation library (e.g., ReportLab, FPDF) for professional reports.
# TODO: Add more detailed trade analysis (e.g., win/loss ratio, average trade profit).
# TODO: Include a section for strategy description and assumptions.
