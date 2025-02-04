import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
import time

# Import the function to fetch trending stock symbols
from data.news_scraper import get_trending_stocks # Ensure this function returns a list of stock symbols

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('swing_trading.log'),
        logging.StreamHandler()
    ]
)

class SwingTrader:
    def __init__(self, stop_loss_pct=0.25, min_profit_pct=0.75, profit_pullback_pct=0.10):
        """
        Initialize SwingTrader with trading parameters.
        Args:
            stop_loss_pct (float): Stop loss percentage (default: 25%).
            min_profit_pct (float): Minimum profit target percentage (default: 75%).
            profit_pullback_pct (float): Pullback percentage to trigger sale at peak (default: 10%).
        """
        self.stop_loss_pct = stop_loss_pct
        self.min_profit_pct = min_profit_pct
        self.profit_pullback_pct = profit_pullback_pct
        self.positions = {}
        self.trade_history = []

    def identify_swing_trade_opportunities(self, symbols, lookback_period=20):
        """
        Identify potential swing trade opportunities based on technical analysis.
        Args:
            symbols (list): List of stock symbols to analyze.
            lookback_period (int): Number of days to look back for analysis.
        Returns:
            list: List of dictionaries containing trade opportunities.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_period)
        
        try:
            # Batch download to reduce API calls
            data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
        except Exception as e:
            logging.error(f"Error downloading data for symbols: {symbols}. {str(e)}")
            return []

        opportunities = []
        for symbol in symbols:
            try:
                stock_data = data[symbol]
                if len(stock_data) < lookback_period:
                    continue

                # Calculate technical indicators
                stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
                stock_data['RSI'] = self._calculate_rsi(stock_data['Close'])

                # Swing trade criteria
                price_above_sma = stock_data['Close'].iloc[-1] > stock_data['SMA_20'].iloc[-1]
                rsi_oversold = stock_data['RSI'].iloc[-1] < 30
                volume_spike = stock_data['Volume'].iloc[-1] > stock_data['Volume'].rolling(window=10).mean().iloc[-1] * 1.5

                if price_above_sma and rsi_oversold and volume_spike:
                    opportunities.append({
                        'symbol': symbol,
                        'current_price': stock_data['Close'].iloc[-1],
                        'volume': stock_data['Volume'].iloc[-1],
                        'rsi': stock_data['RSI'].iloc[-1],
                        'timestamp': datetime.now()
                    })
            except KeyError:
                logging.warning(f"No data for symbol: {symbol}. Skipping.")
            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {str(e)}")
        
        return opportunities

    def execute_swing_trades(self, opportunities, max_positions=5):
        """
        Execute trades based on identified opportunities.
        Args:
            opportunities (list): List of trade opportunities.
            max_positions (int): Maximum number of concurrent positions.
        """
        available_positions = max_positions - len(self.positions)
        if available_positions <= 0:
            logging.info("Maximum positions reached, no new trades executed.")
            return

        for opportunity in opportunities[:available_positions]:
            symbol = opportunity['symbol']
            entry_price = opportunity['current_price']
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            target_price = entry_price * (1 + self.min_profit_pct)

            self.positions[symbol] = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'peak_price': entry_price,
                'quantity': self._calculate_position_size(entry_price),
                'entry_date': datetime.now()
            }

            logging.info(f"Executed swing trade for {symbol} at {entry_price}.")

    def monitor_swing_positions(self):
        """
        Monitor existing positions and handle exits based on strategy rules.
        Returns:
            list: List of closed positions.
        """
        closed_positions = []

        for symbol, position in list(self.positions.items()):
            try:
                current_price = self._get_current_price(symbol)
                position['peak_price'] = max(position['peak_price'], current_price)

                if current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = current_price
                elif current_price >= position['target_price']:
                    pullback = (position['peak_price'] - current_price) / position['peak_price']
                    if pullback >= self.profit_pullback_pct:
                        exit_reason = 'target_with_pullback'
                        exit_price = current_price
                    else:
                        continue
                else:
                    continue

                profit_loss = (exit_price - position['entry_price']) / position['entry_price']
                closed_position = {
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'exit_reason': exit_reason,
                    'entry_date': position['entry_date'],
                    'exit_date': datetime.now(),
                    'holding_period': (datetime.now() - position['entry_date']).days
                }

                self.trade_history.append(closed_position)
                closed_positions.append(closed_position)
                del self.positions[symbol]

                logging.info(f"Closed position for {symbol} at {exit_price} ({exit_reason}).")
            except Exception as e:
                logging.error(f"Error monitoring {symbol}: {str(e)}")

        return closed_positions

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_position_size(self, price, risk_per_trade=0.02, account_size=100000):
        """Calculate position size based on risk management rules."""
        risk_amount = account_size * risk_per_trade
        return int(risk_amount / (price * self.stop_loss_pct))

    def _get_current_price(self, symbol):
        """Get the current price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('regularMarketPrice', None)
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {str(e)}")
            raise

    def _has_options_chain(self, symbol):
        """Check if a stock has options chains available."""
        try:
            ticker = yf.Ticker(symbol)
            return bool(ticker.options)
        except Exception as e:
            logging.error(f"Error checking options chain for {symbol}: {str(e)}")
            return False

# Main script
if __name__ == "__main__":
    trader = SwingTrader()

    while True:
        try:
            # Fetch dynamic symbols
            all_symbols = get_trending_stocks()

            # Filter symbols without options chains
            swing_trade_symbols = [symbol for symbol in all_symbols if not trader._has_options_chain(symbol)]

            # Identify opportunities
            opportunities = trader.identify_swing_trade_opportunities(swing_trade_symbols)
            
            # Execute trades
            trader.execute_swing_trades(opportunities)
            
            # Monitor positions
            closed_positions = trader.monitor_swing_positions()

            if closed_positions:
                metrics = trader.get_performance_metrics()
                logging.info(f"Performance metrics: {metrics}")

            time.sleep(300)  # 5-minute delay
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            time.sleep(60)