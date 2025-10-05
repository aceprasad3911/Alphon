# src/backtesting/strategy_manager.py

# Defines how alpha signals are converted into trading strategies

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AlphaStrategy(ABC):
    """
    Abstract base class for defining alpha signal-based trading strategies.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__} strategy.")

    @abstractmethod
    def generate_weights(self,
                         alpha_signals: pd.Series,
                         current_prices: pd.Series,
                         current_portfolio_value: float) -> pd.Series:
        """
        Abstract method to generate target portfolio weights based on alpha signals.
        Args:
            alpha_signals (pd.Series): Alpha signals for available assets at the current time.
                                       Higher value implies stronger conviction.
            current_prices (pd.Series): Current prices for available assets.
            current_portfolio_value (float): Current total value of the portfolio (cash + market value).
        Returns:
            pd.Series: Target portfolio weights (summing to 1 for long-only, or allowing short positions).
                       Index should be asset symbols.
        """
        pass

class TopNLongOnlyStrategy(AlphaStrategy):
    """
    A simple long-only strategy that invests equally in the top N assets
    based on their alpha signals.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.top_n = config.get("top_n", 5)
        self.min_signal_threshold = config.get("min_signal_threshold", 0.0)
        logger.info(f"TopNLongOnlyStrategy: Top N={self.top_n}, Min Signal Threshold={self.min_signal_threshold}")

    def generate_weights(self,
                         alpha_signals: pd.Series,
                         current_prices: pd.Series,
                         current_portfolio_value: float) -> pd.Series:
        """
        Generates equal weights for the top N assets with signals above a threshold.
        Args:
            alpha_signals (pd.Series): Alpha signals.
            current_prices (pd.Series): Current prices.
            current_portfolio_value (float): Current portfolio value.
        Returns:
            pd.Series: Target portfolio weights.
        """
        if alpha_signals.empty:
            return pd.Series(0.0, index=current_prices.index)

        # Filter by minimum signal threshold
        filtered_signals = alpha_signals[alpha_signals >= self.min_signal_threshold]

        if filtered_signals.empty:
            logger.debug("No signals above minimum threshold. All cash.")
            return pd.Series(0.0, index=current_prices.index)

        # Select top N assets
        top_assets = filtered_signals.nlargest(self.top_n).index.tolist()

        if not top_assets:
            logger.debug("No assets selected for investment after filtering. All cash.")
            return pd.Series(0.0, index=current_prices.index)

        # Assign equal weights to selected assets
        weights = pd.Series(0.0, index=current_prices.index)
        weight_per_asset = 1.0 / len(top_assets)
        for asset in top_assets:
            if asset in weights.index:
                weights[asset] = weight_per_asset
            else:
                logger.warning(f"Asset {asset} from top signals not found in current prices index.")

        # Ensure weights sum to 1 (or close to it due to floating point)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        logger.debug(f"Generated weights: {weights[weights > 0].to_dict()}")
        return weights

class SignalProportionalStrategy(AlphaStrategy):
    """
    A strategy where weights are proportional to the alpha signal strength,
    after applying a threshold and normalization.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_signal_threshold = config.get("min_signal_threshold", 0.0)
        self.max_position_weight = config.get("max_position_weight", 0.2) # Max weight for a single asset
        logger.info(f"SignalProportionalStrategy: Min Signal Threshold={self.min_signal_threshold}, Max Position Weight={self.max_position_weight}")

    def generate_weights(self,
                         alpha_signals: pd.Series,
                         current_prices: pd.Series,
                         current_portfolio_value: float) -> pd.Series:
        """
        Generates weights proportional to signal strength.
        Args:
            alpha_signals (pd.Series): Alpha signals.
            current_prices (pd.Series): Current prices.
            current_portfolio_value (float): Current portfolio value.
        Returns:
            pd.Series: Target portfolio weights.
        """
        if alpha_signals.empty:
            return pd.Series(0.0, index=current_prices.index)

        # Filter by minimum signal threshold
        filtered_signals = alpha_signals[alpha_signals >= self.min_signal_threshold]

        if filtered_signals.empty:
            logger.debug("No signals above minimum threshold. All cash.")
            return pd.Series(0.0, index=current_prices.index)

        # Normalize signals to sum to 1
        total_signal = filtered_signals.sum()
        if total_signal == 0:
            logger.debug("Sum of filtered signals is zero. All cash.")
            return pd.Series(0.0, index=current_prices.index)

        raw_weights = filtered_signals / total_signal

        # Apply max position weight constraint
        if self.max_position_weight > 0:
            raw_weights = raw_weights.clip(upper=self.max_position_weight)
            # Re-normalize after clipping
            raw_weights = raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else raw_weights

        weights = pd.Series(0.0, index=current_prices.index)
        weights.update(raw_weights) # Update only the assets that have weights

        logger.debug(f"Generated weights: {weights[weights > 0].to_dict()}")
        return weights

# TODO: Implement more complex strategies:
# - Long/Short strategy
# - Sector rotation strategy
# - Risk-parity strategy (requires volatility estimates)
# - Strategies incorporating transaction costs directly in optimization
# - Strategies using numerical optimization (e.g., CVXPY) for portfolio construction.
