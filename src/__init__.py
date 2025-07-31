# src/__init__.py
# This file makes the 'src' directory a Python package.
# It can be used to define package-level variables or perform initializations.

import logging

# Configure basic logging for the entire project
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# You can also import commonly used modules or functions here for easier access
# from .utils.config_loader import load_config
# from .data_sourcing.yfinance import YahooFinanceSource

# Define package version
__version__ = "0.1.0"

# TODO: Add any other package-wide initializations or imports here.
