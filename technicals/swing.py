import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict

class SwingTrader:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """Initialize the swing trader with Alpaca credentials."""
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.logger = self._setup_logger()
        self.account = self.api.get_account()
        self.trading_cash = float(self.account.cash) * 0.7  # Using 70% of account cash
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SwingTrader')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_historical_data(self, symbol: str, timeframe: str = '1Day', limit: int = 200) -> pd.DataFrame:
        """Fetch historical data from Alpaca."""
        try:
            bars = self.api.get_barset(symbol, timeframe, limit=limit).df[symbol]
            return bars
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        # SMAs
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        
        # Volume MA
        df['volume_MA20'] = df['volume'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # ADX
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Directional Indicators
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        df['plus_di'] = 100 * pd.Series(plus_dm).rolling(window=14).mean() / df['ATR']
        df['minus_di'] = 100 * pd.Series(minus_dm).rolling(window=14).mean() / df['ATR']
        
        return df

    def check_trend_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if trend confirmation indicators align."""
        latest = df.iloc[-1]
        return (
            latest['SMA50'] > latest['SMA200'] and  # Bullish bias
            latest['close'] > latest['SMA50'] and    # Price above moving average
            latest['volume'] > latest['volume_MA20']  # Strong volume
        )

    def check_momentum_volatility(self, df: pd.DataFrame) -> bool:
        """Check momentum and volatility indicators."""
        latest = df.iloc[-1]
        return (
            latest['plus_di'] > latest['minus_di'] and  # Positive directional movement
            30 <= latest['RSI'] <= 70                   # RSI in favorable zone
        )

    def check_risk_reward(self, df: pd.DataFrame, risk_ratio: float = 2.0) -> bool:
        """Evaluate risk/reward ratio using Bollinger Bands."""
        latest = df.iloc[-1]
        potential_reward = latest['BB_upper'] - latest['close']
        potential_risk = latest['close'] - latest['BB_lower']
        return potential_reward >= (risk_ratio * potential_risk)

    def generate_signals(self, symbols: List[str], indicator_choice: str) -> Dict[str, bool]:
        """Generate trading signals based on selected indicators."""
        signals = {}
        
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if df.empty:
                continue
                
            df = self.calculate_indicators(df)
            
            if indicator_choice == '1':
                signals[symbol] = self.check_trend_confirmation(df)
            elif indicator_choice == '2':
                signals[symbol] = self.check_momentum_volatility(df)
            elif indicator_choice == '3':
                signals[symbol] = self.check_risk_reward(df)
            else:  # All indicators
                signals[symbol] = (
                    self.check_trend_confirmation(df) and
                    self.check_momentum_volatility(df) and
                    self.check_risk_reward(df)
                )
        
        return signals

    def execute_trades(self, signals: Dict[str, bool]):
        """Execute trades based on generated signals."""
        for symbol, should_buy in signals.items():
            try:
                position = self.api.get_position(symbol)
                # If we have a position but signal is sell, close it
                if not should_buy:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    self.logger.info(f"Closed position in {symbol}")
            except Exception:  # No position exists
                if should_buy:
                    # Calculate position size based on available cash
                    price = float(self.api.get_last_trade(symbol).price)
                    shares = int(self.trading_cash / (len(signals) * price))
                    
                    if shares > 0:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=shares,
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        self.logger.info(f"Opened position in {symbol}: {shares} shares")

def main():
    """Main function to run the swing trading system."""
    # Replace with your Alpaca API credentials
    API_KEY = "YOUR_API_KEY"
    API_SECRET = "YOUR_API_SECRET"
    BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading for testing
    
    trader = SwingTrader(API_KEY, API_SECRET, BASE_URL)
    
    # Get user input
    symbols = input("Enter stock symbols (comma-separated): ").split(',')
    symbols = [s.strip().upper() for s in symbols]
    
    print("\nChoose indicator combination:")
    print("1: Trend Confirmation only")
    print("2: Momentum/Volatility only")
    print("3: Risk/Reward only")
    print("4: All indicators")
    
    choice = input("Enter your choice (1-4): ")
    
    while True:
        try:
            signals = trader.generate_signals(symbols, choice)
            trader.execute_trades(signals)
            trader.logger.info("Waiting for next iteration...")
            time.sleep(300)  # Wait 5 minutes before next iteration
            
        except KeyboardInterrupt:
            trader.logger.info("Stopping the trading system...")
            break
        except Exception as e:
            trader.logger.error(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()