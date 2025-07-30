# src/reporting/__init__.py
# This file makes the 'reporting' directory a Python package.

# Import key reporting modules for easier access
from .metrics import calculate_performance_metrics
from .report_generator import generate_backtest_report
from .interpretability_analyzer import analyze_disentanglement, interpret_signals

# TODO: Add any shared reporting utilities or configurations here.
