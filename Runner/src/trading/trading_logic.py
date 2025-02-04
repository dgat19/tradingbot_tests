"""
Optimized Trading Logic
-----------------------
Integrates predictions from a regression model and decisions from an RL agent
to determine and execute trades. Uses concurrency for efficient per-symbol
processing and maintains robust risk management features.
"""

import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Alpaca Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, AssetClass
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Regression Model & RL Agent Imports (Adjust these to match your module structures)
from models.regression_model import get_prediction  # e.g. returns (prediction, confidence)
from models.rl_agent import EnhancedStockTradingEnv  # e.g. returns (action, confidence)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionStrategy(Enum):
    """Available option trading strategies (example placeholders)."""
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"
    BUTTERFLY = "BUTTERFLY"

class TradingSystem:
    """
    Enhanced trading system optimized for integrating both a regression model
    and an RL agent to determine trades. Manages risk, position sizing, 
    and real or paper trades via Alpaca.
    """

    def __init__(self, mode: str = 'paper', max_threads: int = 5):
        """
        Initialize the trading system.
        
        :param mode: 'paper' or 'live' trading mode.
        :param max_threads: Maximum threads for concurrent symbol processing.
        """
        load_dotenv()
        self.mode = mode
        self.positions = {}
        self.trades_history = []
        self.max_daily_risk = 0.1  # Use 2% of account equity daily
        self.cached_data = defaultdict(dict)
        self.max_threads = max_threads

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self._initialize_clients()

        # Set initial risk parameters
        self.max_position_size = 0.0
        self.update_risk_parameters()

    def _initialize_clients(self) -> None:
        """Initialize Alpaca trading and data clients."""
        try:
            self.trading_client = TradingClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                paper=(self.mode == 'paper')
            )
            self.data_client = StockHistoricalDataClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY')
            )
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca clients: {e}")
            raise

    def update_risk_parameters(self) -> None:
        """Update risk parameters based on account equity."""
        try:
            equity = float(self.trading_client.get_account().equity)
            self.max_position_size = equity * self.max_daily_risk
            logger.info(f"Max position size updated to ${self.max_position_size:.2f}")
        except Exception as e:
            logger.error(f"Error updating risk parameters: {e}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the latest price for a symbol, using cached value if available.
        
        :param symbol: Stock symbol to fetch price for.
        :return: Latest closing price or None on error.
        """
        if "price" in self.cached_data[symbol]:
            return self.cached_data[symbol]["price"]
        try:
            price = self._fetch_stock_bars(symbol)
            if price:
                self.cached_data[symbol]["price"] = price
            return price
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def _fetch_stock_bars(self, symbol: str, limit: int = 1) -> Optional[float]:
        """Helper to fetch the latest stock bars via Alpaca and return the close price."""
        try:
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=limit
                )
            )
            return bars[symbol][0].close if bars and symbol in bars else None
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, days: int = 15) -> pd.DataFrame:
        """
        Fetch historical data for a symbol as a DataFrame.
        
        :param symbol: Symbol to fetch historical data for.
        :param days: Number of days to fetch.
        :return: DataFrame with columns [Open, High, Low, Close, Volume].
        """
        try:
            # Using daily timeframe in example. Adjust as needed.
            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=days
                )
            )
            df = pd.DataFrame(
                [(bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in bars[symbol]],
                columns=['Open', 'High', 'Low', 'Close', 'Volume']
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_combined_signals(self, symbol: str) -> Dict[str, float]:
        """
        Combine signals from the regression model and RL agent.

        :param symbol: Symbol to get signals for.
        :return: Dictionary containing combined signals, e.g.:
                 {
                    'regression_prediction': float,
                    'regression_confidence': float,
                    'rl_action': str,
                    'rl_confidence': float
                 }
        """
        try:
            # Example: regression model returns (prediction, confidence)
            reg_prediction, reg_confidence = get_prediction(symbol)
            # Example: RL agent returns (action, confidence)
            rl_action, rl_confidence = EnhancedStockTradingEnv(symbol)

            # You can add any logic to blend or weigh these signals
            signals = {
                'regression_prediction': reg_prediction,
                'regression_confidence': reg_confidence,
                'rl_action': rl_action,
                'rl_confidence': rl_confidence,
            }
            return signals
        except Exception as e:
            logger.error(f"Error getting combined signals for {symbol}: {e}")
            return {}

    def determine_trade_action(self, signals: Dict[str, float]) -> Optional[OptionStrategy]:
        """
        Decide which option strategy to use based on the combined signals.
        
        :param signals: Combined signals dict with model predictions and confidence.
        :return: One of the OptionStrategy enum values or None if no action.
        """
        if not signals:
            return None

        # Basic example logic: if regression is bullish & RL agent is bullish, go long call
        if signals['regression_confidence'] > 0.6 and signals['rl_confidence'] > 0.6:
            # e.g., both suggest bullish
            return OptionStrategy.LONG_CALL
        elif signals['regression_confidence'] < 0.4 and signals['rl_confidence'] < 0.4:
            # e.g., both suggest bearish
            return OptionStrategy.LONG_PUT

        return None  # If signals are mixed or not strong enough, do nothing

    def calculate_position_size(self, symbol: str, signals: Dict[str, float]) -> float:
        """
        Calculate position size dynamically based on model signals, volatility, and Kelly criterion.
        
        :param symbol: Symbol to size a position for.
        :param signals: Combined signals containing regression and RL confidences.
        :return: Recommended position size in monetary value.
        """
        try:
            equity = float(self.trading_client.get_account().equity)
            base_size = equity * self.max_daily_risk

            # ATR-based scaling
            atr = self._calculate_atr(symbol)
            if atr:
                volatility_factor = 1 / atr
                base_size *= volatility_factor

            # Kelly Criterion example (for demonstration):
            # If average confidence is > 0.5, we assume it's a positive expectancy
            avg_conf = (signals.get('regression_confidence', 0) + signals.get('rl_confidence', 0)) / 2
            # Example assumption: payoff ratio = 1.5
            kelly_fraction = avg_conf - (1 - avg_conf) / 1.5
            kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limit Kelly to 50%
            base_size *= kelly_fraction

            # Ensure we do not exceed system's max_position_size
            position_size = min(base_size, self.max_position_size)
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate ATR (Average True Range) for the specified period, with basic caching.
        
        :param symbol: Symbol to calculate ATR for.
        :param period: Number of days for ATR calculation.
        :return: ATR value or None on error.
        """
        if "ATR" in self.cached_data[symbol]:
            return self.cached_data[symbol]["ATR"]
        try:
            data = self.get_historical_data(symbol, period + 1)
            if len(data) < period:
                return None

            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values

            tr = np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(close[:-1] - low[1:])
            ])
            atr = np.mean(tr[-period:])
            self.cached_data[symbol]["ATR"] = atr
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None

    def run_trading_cycle(self, symbols: List[str]) -> None:
        """
        Run a trading cycle across multiple symbols in parallel.
        
        :param symbols: List of stock symbols to evaluate.
        """
        self.update_risk_parameters()

        def process_symbol(symbol: str) -> None:
            try:
                signals = self.get_combined_signals(symbol)
                if signals:
                    strategy = self.determine_trade_action(signals)
                    if strategy:
                        self.place_option_trade(symbol, strategy, signals)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            executor.map(process_symbol, symbols)

    def place_option_trade(self, symbol: str, strategy: OptionStrategy, signals: Dict[str, float]) -> None:
        """
        Place an option trade based on the selected strategy and signals.
        
        :param symbol: Symbol to trade.
        :param strategy: OptionStrategy enum indicating the chosen strategy.
        :param signals: Combined signals dict.
        """
        try:
            position_size = self.calculate_position_size(symbol, signals)
            if position_size <= 0:
                logger.info(f"No position size calculated for {symbol}, skipping.")
                return

            # Example: we just place a simple CALL or PUT market order
            # In reality, you'd pick the correct options chain, expiration, strike, etc.
            if strategy == OptionStrategy.LONG_CALL:
                order_side = OrderSide.BUY
                direction = "CALL"
            elif strategy == OptionStrategy.LONG_PUT:
                order_side = OrderSide.BUY
                direction = "PUT"
            else:
                logger.info(f"Strategy {strategy} not implemented, skipping trade.")
                return

            if self.mode in ['paper', 'live']:
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=1,  # For demonstration, consider a single contract
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                        type=OrderType.MARKET,
                        class_=AssetClass.OPTION
                    )
                )
                logger.info(f"Submitted {direction} order for {symbol}, ID: {order.id}")

            # Record position
            self.positions[symbol] = {
                'direction': direction,
                'cost': position_size,  # or track fill cost from 'order' details
                'entry_time': datetime.now(),
                'expiry': datetime.now(),  # or set an actual option expiry date
                'max_pnl': 0.0  # For trailing stop logic
            }

        except Exception as e:
            logger.error(f"Error placing option trade for {symbol}: {e}")

    def monitor_positions(self) -> None:
        """
        Monitor open positions and apply trailing stops or exit logic.
        Closes positions if conditions are met (stop loss, trailing stop, or expiry).
        """
        try:
            for symbol, position in list(self.positions.items()):
                current_price = self.get_current_price(symbol)
                if not current_price:
                    continue

                # Convert cost from total $$ to an implied "entry price" notion 
                # (you might need more precise fills from order fills)
                entry_price = position['cost'] / 100.0
                if entry_price <= 0:
                    continue

                if position['direction'] == 'CALL':
                    pnl = (current_price - entry_price) / entry_price
                else:  # 'PUT'
                    pnl = (entry_price - current_price) / entry_price

                # Update max P&L for trailing stop
                if 'max_pnl' not in position:
                    position['max_pnl'] = pnl
                position['max_pnl'] = max(position['max_pnl'], pnl)

                # Define exit conditions
                should_exit = (
                    pnl <= -0.20  # Stop loss at -20%
                    or pnl >= position['max_pnl'] - 0.10  # Trailing stop 10% below max
                    or datetime.now() >= position['expiry']
                )

                if should_exit:
                    self.close_position(symbol, position)
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    def close_position(self, symbol: str, position: Dict) -> None:
        """
        Close an open position for a symbol.
        
        :param symbol: Symbol with an open position.
        :param position: Dictionary with position details.
        """
        try:
            if self.mode in ['paper', 'live']:
                self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=1,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                        type=OrderType.MARKET,
                        class_=AssetClass.OPTION
                    )
                )
                logger.info(f"Closed position for {symbol}")

            if symbol in self.positions:
                del self.positions[symbol]

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
