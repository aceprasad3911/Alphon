# src/backtesting/engine.py

# Core backtesting logic and simulation engine

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from .slippage_commissions import CommissionModel, SlippageModel, FixedCommission, PercentageSlippage
from .strategy_manager import AlphaStrategy

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Core backtesting engine to simulate trading strategies based on alpha signals.
    """

    def __init__(self,
                 initial_capital: float,
                 commission_model: Optional[CommissionModel] = None,
                 slippage_model: Optional[SlippageModel] = None):
        """
        Initializes the backtesting engine.
        Args:
            initial_capital (float): Starting capital for the portfolio.
            commission_model (Optional[CommissionModel]): Model for calculating commissions.
            slippage_model (Optional[SlippageModel]): Model for calculating slippage.
        """
        self.initial_capital = initial_capital
        self.commission_model = commission_model if commission_model else FixedCommission(0.0)
        self.slippage_model = slippage_model if slippage_model else PercentageSlippage(0.0)
        self.portfolio_history = pd.DataFrame()
        self.trade_log = pd.DataFrame()
        logger.info(f"BacktestingEngine initialized with capital: ${initial_capital:,.2f}")

    def run_backtest(self,
                     alpha_signals: pd.DataFrame,  # DataFrame of alpha signals (index=date, columns=symbols)
                     prices: pd.DataFrame,  # DataFrame of asset prices (index=date, columns=symbols)
                     strategy: AlphaStrategy,  # An instance of AlphaStrategy
                     start_date: str = None,
                     end_date: str = None,
                     rebalance_frequency: str = 'monthly') -> Dict[str, pd.DataFrame]:
        """
        Runs the backtest simulation.
        Args:
            alpha_signals (pd.DataFrame): DataFrame of alpha signals. Index is date, columns are asset symbols.
                                          Higher signal implies stronger buy recommendation.
            prices (pd.DataFrame): DataFrame of asset prices (e.g., 'close' prices). Index is date, columns are asset symbols.
            strategy (AlphaStrategy): An instance of AlphaStrategy defining how signals are converted to positions.
            start_date (str): Start date for the backtest (YYYY-MM-DD).
            end_date (str): End date for the backtest (YYYY-MM-DD).
            rebalance_frequency (str): How often to rebalance the portfolio ('daily', 'weekly', 'monthly', 'quarterly', 'annual').
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing 'portfolio_history' and 'trade_log'.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date} with {rebalance_frequency} rebalancing.")

        # Align data and filter by date range
        combined_data = pd.concat([prices.add_prefix('price_'), alpha_signals.add_prefix('signal_')], axis=1)
        if start_date:
            combined_data = combined_data[combined_data.index >= pd.to_datetime(start_date)]
        if end_date:
            combined_data = combined_data[combined_data.index <= pd.to_datetime(end_date)]

        if combined_data.empty:
            logger.error("No data available for the specified backtest period.")
            return {"portfolio_history": pd.DataFrame(), "trade_log": pd.DataFrame()}

        # Initialize portfolio
        current_capital = self.initial_capital
        current_positions = pd.Series(0, index=prices.columns, dtype=float)  # Number of shares
        portfolio_value_history = []
        trade_records = []

        # Determine rebalance dates
        rebalance_dates = pd.date_range(start=combined_data.index.min(),
                                        end=combined_data.index.max(),
                                        freq=rebalance_frequency[0].upper())  # 'M', 'W', 'D', 'Q', 'A'
        rebalance_dates = rebalance_dates[rebalance_dates.isin(combined_data.index)]  # Only use dates present in data

        if rebalance_dates.empty:
            logger.warning("No rebalance dates found. Backtest will run without rebalancing.")
            rebalance_dates = [combined_data.index.min()]  # Rebalance once at start

        last_rebalance_date = None

        for i, current_date in enumerate(combined_data.index):
            # Update portfolio value
            current_prices = combined_data.loc[current_date].filter(like='price_').rename(
                lambda x: x.replace('price_', ''))
            current_market_value = (current_positions * current_prices).sum()
            total_portfolio_value = current_capital + current_market_value
            portfolio_value_history.append({
                'date': current_date,
                'capital': current_capital,
                'market_value': current_market_value,
                'total_value': total_portfolio_value
            })

            # Check for rebalance
            if current_date in rebalance_dates or i == 0:  # Rebalance on first day too
                if last_rebalance_date is None or current_date > last_rebalance_date:
                    logger.debug(f"Rebalancing on {current_date.strftime('%Y-%m-%d')}")

                    # Get signals for current date
                    current_signals = combined_data.loc[current_date].filter(like='signal_').rename(
                        lambda x: x.replace('signal_', ''))
                    current_signals = current_signals.dropna()  # Only consider assets with signals

                    if current_signals.empty:
                        logger.warning(f"No valid signals on {current_date}. Skipping rebalance.")
                        last_rebalance_date = current_date
                        continue

                    # Calculate target weights based on strategy
                    target_weights = strategy.generate_weights(current_signals, current_prices, total_portfolio_value)

                    # Calculate target positions (number of shares)
                    target_positions = (target_weights * total_portfolio_value / current_prices).fillna(0)
                    target_positions = target_positions.round(0)  # Round to whole shares

                    # Calculate trades needed
                    trades = target_positions - current_positions.reindex(target_positions.index, fill_value=0)
                    trades = trades[trades != 0]  # Only consider non-zero trades

                    if not trades.empty:
                        logger.debug(f"Trades on {current_date}: {trades.to_dict()}")
                        for asset, shares_to_trade in trades.items():
                            trade_price = current_prices.get(asset)
                            if pd.isna(trade_price):
                                logger.warning(f"Price for {asset} not available on {current_date}. Skipping trade.")
                                continue

                            trade_value = shares_to_trade * trade_price
                            commission = self.commission_model.calculate(trade_value, shares_to_trade)
                            slippage = self.slippage_model.calculate(trade_value, shares_to_trade)

                            net_trade_value = trade_value + commission + slippage  # Cost for buying, gain for selling

                            if shares_to_trade > 0:  # Buying
                                if current_capital >= net_trade_value:
                                    current_capital -= net_trade_value
                                    current_positions[asset] += shares_to_trade
                                    trade_records.append({
                                        'date': current_date,
                                        'asset': asset,
                                        'type': 'BUY',
                                        'shares': shares_to_trade,
                                        'price': trade_price,
                                        'value': trade_value,
                                        'commission': commission,
                                        'slippage': slippage,
                                        'net_value': net_trade_value
                                    })
                                else:
                                    logger.warning(
                                        f"Insufficient capital to buy {shares_to_trade} of {asset} on {current_date}.")
                            else:  # Selling
                                current_capital -= net_trade_value  # Selling adds to capital, so subtract negative cost
                                current_positions[asset] += shares_to_trade
                                trade_records.append({
                                    'date': current_date,
                                    'asset': asset,
                                    'type': 'SELL',
                                    'shares': abs(shares_to_trade),
                                    'price': trade_price,
                                    'value': abs(trade_value),
                                    'commission': commission,
                                    'slippage': slippage,
                                    'net_value': net_trade_value
                                })
                    last_rebalance_date = current_date

        self.portfolio_history = pd.DataFrame(portfolio_value_history).set_index('date')
        self.trade_log = pd.DataFrame(trade_records).set_index('date')

        logger.info("Backtest simulation complete.")
        logger.info(f"Final Portfolio Value: ${self.portfolio_history['total_value'].iloc[-1]:,.2f}")

        return {"portfolio_history": self.portfolio_history, "trade_log": self.trade_log}

# TODO: Implement short selling logic.
# TODO: Handle cash management more explicitly (e.g., interest on cash).
# TODO: Add support for different order types (e.g., market, limit).
# TODO: Integrate with a more robust backtesting library like `bt` or `Backtrader` for complex scenarios.
