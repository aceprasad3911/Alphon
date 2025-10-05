# src/backtesting/slippage_commissions.py

# Models realistic trading costs

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CommissionModel(ABC):
    """
    Abstract base class for commission models.
    """
    @abstractmethod
    def calculate(self, trade_value: float, shares: float) -> float:
        """
        Calculates the commission for a trade.
        Args:
            trade_value (float): The absolute value of the trade (price * shares).
            shares (float): The absolute number of shares traded.
        Returns:
            float: The commission amount (always positive).
        """
        pass

class FixedCommission(CommissionModel):
    """
    A simple commission model with a fixed rate per trade value.
    """
    def __init__(self, rate: float):
        """
        Args:
            rate (float): Commission rate as a percentage of trade value (e.g., 0.0005 for 0.05%).
        """
        if not (0 <= rate < 1):
            raise ValueError("Commission rate must be between 0 and 1 (exclusive of 1).")
        self.rate = rate
        logger.info(f"FixedCommission model initialized with rate: {rate*100:.2f}%")

    def calculate(self, trade_value: float, shares: float) -> float:
        """
        Calculates commission as a fixed percentage of the trade value.
        Args:
            trade_value (float): The absolute value of the trade.
            shares (float): The absolute number of shares traded.
        Returns:
            float: The calculated commission.
        """
        commission = abs(trade_value) * self.rate
        logger.debug(f"Calculated fixed commission: {commission:.4f} for trade value {trade_value:.2f}")
        return commission

class PerShareCommission(CommissionModel):
    """
    A commission model with a fixed amount per share traded.
    """
    def __init__(self, rate_per_share: float, min_commission: float = 0.0, max_commission: float = float('inf')):
        """
        Args:
            rate_per_share (float): Commission amount per share.
            min_commission (float): Minimum commission per trade.
            max_commission (float): Maximum commission per trade.
        """
        if rate_per_share < 0:
            raise ValueError("Rate per share cannot be negative.")
        self.rate_per_share = rate_per_share
        self.min_commission = min_commission
        self.max_commission = max_commission
        logger.info(f"PerShareCommission model initialized with rate: ${rate_per_share:.4f}/share")

    def calculate(self, trade_value: float, shares: float) -> float:
        """
        Calculates commission as a fixed amount per share, with min/max limits.
        Args:
            trade_value (float): The absolute value of the trade.
            shares (float): The absolute number of shares traded.
        Returns:
            float: The calculated commission.
        """
        commission = abs(shares) * self.rate_per_share
        commission = max(self.min_commission, min(commission, self.max_commission))
        logger.debug(f"Calculated per-share commission: {commission:.4f} for {shares} shares")
        return commission

class SlippageModel(ABC):
    """
    Abstract base class for slippage models.
    """
    @abstractmethod
    def calculate(self, trade_value: float, shares: float) -> float:
        """
        Calculates the slippage cost for a trade.
        Slippage is typically an additional cost incurred due to market impact.
        Args:
            trade_value (float): The absolute value of the trade.
            shares (float): The absolute number of shares traded.
        Returns:
            float: The slippage amount (always positive, added to cost for buys, subtracted from proceeds for sells).
        """
        pass

class PercentageSlippage(SlippageModel):
    """
    A simple slippage model with a fixed percentage of trade value.
    """
    def __init__(self, bps: float):
        """
        Args:
            bps (float): Slippage in basis points (e.g., 1 for 1 bps = 0.0001).
        """
        if not (0 <= bps):
            raise ValueError("Slippage basis points must be non-negative.")
        self.rate = bps / 10000.0 # Convert basis points to a decimal rate
        logger.info(f"PercentageSlippage model initialized with {bps} bps ({self.rate*100:.4f}%)")

    def calculate(self, trade_value: float, shares: float) -> float:
        """
        Calculates slippage as a fixed percentage of the trade value.
        Args:
            trade_value (float): The absolute value of the trade.
            shares (float): The absolute number of shares traded.
        Returns:
            float: The calculated slippage.
        """
        slippage = abs(trade_value) * self.rate
        logger.debug(f"Calculated percentage slippage: {slippage:.4f} for trade value {trade_value:.2f}")
        return slippage

# TODO: Implement more advanced slippage models (e.g., based on volume, volatility, A-D curve).
# TODO: Consider combining commission and slippage into a single transaction cost model.
