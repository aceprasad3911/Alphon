# src/backtesting/stress_testing.py

# Logic for evaluating performance under different market regimes

import pandas as pd
import logging
from typing import Dict, Any, List
from .engine import BacktestingEngine
from .strategy_manager import AlphaStrategy
from src.reporting.metrics import calculate_performance_metrics # Assuming this exists

logger = logging.getLogger(__name__)

class MarketRegimeTester:
    """
    Evaluates alpha signal performance under different predefined market regimes.
    """
    def __init__(self, config: Dict[str, Any], backtesting_engine: BacktestingEngine):
        """
        Initializes the market regime tester.
        Args:
            config (Dict[str, Any]): Configuration containing market regime definitions.
            backtesting_engine (BacktestingEngine): An instance of the backtesting engine.
        """
        self.config = config
        self.backtesting_engine = backtesting_engine
        self.market_regimes = config.get("market_regimes", [])
        logger.info(f"MarketRegimeTester initialized with {len(self.market_regimes)} regimes.")

    def run_regime_tests(self,
                         alpha_signals: pd.DataFrame,
                         prices: pd.DataFrame,
                         strategy: AlphaStrategy) -> Dict[str, Dict[str, Any]]:
        """
        Runs backtests for each defined market regime and collects performance metrics.
        Args:
            alpha_signals (pd.DataFrame): DataFrame of alpha signals.
            prices (pd.DataFrame): DataFrame of asset prices.
            strategy (AlphaStrategy): The trading strategy to test.
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are regime names and values are
                                       dictionaries of performance metrics for that regime.
        """
        regime_results = {}
        logger.info("Starting market regime stress testing.")

        for regime in self.market_regimes:
            regime_name = regime.get("name", "Unnamed Regime")
            start_date = regime.get("start")
            end_date = regime.get("end")

            if not start_date or not end_date:
                logger.warning(f"Skipping regime '{regime_name}' due to missing start/end dates.")
                continue

            logger.info(f"\n--- Testing Regime: {regime_name} ({start_date} to {end_date}) ---")

            # Filter signals and prices for the current regime
            regime_signals = alpha_signals[(alpha_signals.index >= pd.to_datetime(start_date)) &
                                           (alpha_signals.index <= pd.to_datetime(end_date))]
            regime_prices = prices[(prices.index >= pd.to_datetime(start_date)) &
                                   (prices.index <= pd.to_datetime(end_date))]

            if regime_signals.empty or regime_prices.empty:
                logger.warning(f"No data for regime '{regime_name}'. Skipping.")
                continue

            # Run backtest for this regime
            backtest_output = self.backtesting_engine.run_backtest(
                alpha_signals=regime_signals,
                prices=regime_prices,
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency=self.config.get("rebalance_frequency", "monthly") # Use global rebalance freq
            )

            portfolio_history = backtest_output["portfolio_history"]

            if not portfolio_history.empty:
                # Calculate performance metrics for this regime
                metrics = calculate_performance_metrics(portfolio_history['total_value'])
                regime_results[regime_name] = metrics
                logger.info(f"Performance for {regime_name}: Sharpe Ratio={metrics.get('sharpe_ratio'):.4f}, "
                            f"Max Drawdown={metrics.get('max_drawdown_pct'):.2%}")
            else:
                logger.warning(f"Backtest for regime '{regime_name}' yielded no portfolio history.")
                regime_results[regime_name] = {"status": "No data/history"}

        logger.info("Market regime stress testing complete.")
        return regime_results

# TODO: Implement methods to automatically identify market regimes (e.g., using VIX, economic indicators).
# TODO: Add visualization for regime-specific performance.
