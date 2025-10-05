# src/backtesting/__init__.py

# This file makes the 'backtesting' directory a Python package.

# Import key backtesting modules for easier access
from .engine import BacktestingEngine
from .strategy_manager import AlphaStrategy
from .slippage_commissions import FixedCommission, PercentageSlippage
from .rolling_window_validation import RollingWindowValidator
from .stress_testing import MarketRegimeTester
from .sensitivity_analysis import SensitivityAnalyzer

# TODO: Add any shared backtesting utilities or configurations here.


