# src/backtesting/sensitivity_analysis.py

# Tools for testing model sensitivity to parameters

import pandas as pd
import logging
from typing import Dict, Any, List, Callable
from .engine import BacktestingEngine
from .strategy_manager import AlphaStrategy
from src.reporting.metrics import calculate_performance_metrics  # Assuming this exists

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on key strategy parameters by running multiple backtests.
    """

    def __init__(self, config: Dict[str, Any], backtesting_engine: BacktestingEngine):
        """
        Initializes the sensitivity analyzer.
        Args:
            config (Dict[str, Any]): Configuration containing parameters to test.
            backtesting_engine (BacktestingEngine): An instance of the backtesting engine.
        """
        self.config = config
        self.backtesting_engine = backtesting_engine
        self.sensitivity_params = config.get("sensitivity_params", {})
        logger.info(f"SensitivityAnalyzer initialized with parameters: {list(self.sensitivity_params.keys())}")

    def run_sensitivity_tests(self,
                              alpha_signals: pd.DataFrame,
                              prices: pd.DataFrame,
                              base_strategy_config: Dict[str, Any],
                              strategy_class: type) -> Dict[str, Dict[Any, Dict[str, Any]]]:
        """
        Runs backtests for different values of specified parameters and collects performance metrics.
        Args:
            alpha_signals (pd.DataFrame): DataFrame of alpha signals.
            prices (pd.DataFrame): DataFrame of asset prices.
            base_strategy_config (Dict[str, Any]): The base configuration for the strategy.
            strategy_class (type): The class of the strategy to test.
        Returns:
            Dict[str, Dict[Any, Dict[str, Any]]]: Nested dictionary of results.
                                                  Outer key: parameter name.
                                                  Inner key: parameter value.
                                                  Innermost value: performance metrics.
        """
        sensitivity_results = {}
        logger.info("Starting sensitivity analysis.")

        for param_name, param_values in self.sensitivity_params.items():
            logger.info(f"\n--- Testing Sensitivity for parameter: '{param_name}' ---")
            param_results = {}
            for value in param_values:
                logger.info(f"  Testing with {param_name}={value}")

                # Create a modified strategy config for the current test
                current_strategy_config = base_strategy_config.copy()
                current_strategy_config[param_name] = value

                strategy_instance = strategy_class(current_strategy_config)

                # Run backtest with the modified strategy
                backtest_output = self.backtesting_engine.run_backtest(
                    alpha_signals=alpha_signals,
                    prices=prices,
                    strategy=strategy_instance,
                    start_date=self.config.get("start_date"),  # Use global backtest dates
                    end_date=self.config.get("end_date"),
                    rebalance_frequency=self.config.get("rebalance_frequency", "monthly")
                )

                portfolio_history = backtest_output["portfolio_history"]

                if not portfolio_history.empty:
                    metrics = calculate_performance_metrics(portfolio_history['total_value'])
                    param_results[value] = metrics
                    logger.info(
                        f"    Sharpe Ratio: {metrics.get('sharpe_ratio'):.4f}, Max Drawdown: {metrics.get('max_drawdown_pct'):.2%}")
                else:
                    logger.warning(f"    Backtest for {param_name}={value} yielded no portfolio history.")
                    param_results[value] = {"status": "No data/history"}

            sensitivity_results[param_name] = param_results

        logger.info("Sensitivity analysis complete.")
        return sensitivity_results

# TODO: Implement visualization for sensitivity analysis results (e.g., heatmaps, line plots).
# TODO: Extend to test sensitivity to model hyperparameters (requires re-training models).
# TODO: Add support for multi-parameter sensitivity analysis.
