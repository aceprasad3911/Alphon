# src/data_sources/__init__.py

# This file makes the 'data_sources' directory a Python package.

# Import specific data source classes for easier access
from .base import DataSource
from .yfinance import YahooFinanceSource
from .alpha_vantage import AlphaVantageSource
from .quandl import QuandlSource
from .wrds import WRDSSource
from .fred import FREDSource
from .edgar import EDGARSource

# Define a dictionary to easily access data source classes by name
DATA_SOURCES = {
    "yahoo": YahooFinanceSource,
    "alpha_vantage": AlphaVantageSource,
    "quandl": QuandlSource,
    "wrds": WRDSSource,
    "fred": FREDSource,
    "edgar": EDGARSource,
}

# TODO: Add any shared data source utilities or configurations here.
