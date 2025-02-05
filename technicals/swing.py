import os
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict, Set
import alpaca_trade_api as tradeapi

load_dotenv()

class SwingTrader:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the swing trader with Alpaca credentials."""
        # Initialize legacy REST client
        self.api = tradeapi.REST(
            api_key,
            api_secret,
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        self.logger = self._setup_logger()
        self.account = self.api.get_account()
        self.trading_cash = float(self.account.cash) * 0.7
        self.verified_symbols = set()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SwingTrader')
        logger.setLevel(logging.INFO)
        if logger.handlers:
            logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def verify_symbols(self, symbols: List[str]) -> Set[str]:
        """Verify which symbols are available for trading using legacy REST client."""
        verified_symbols = set()
        
        end = datetime.now()
        start = end - timedelta(days=5)  # Check last 5 days
        
        for symbol in symbols:
            try:
                bars = self.api.get_bars(
                    symbol,
                    '1D',  # Changed from 'Day' to '1D'
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    limit=5,
                    feed='iex'
                ).df
                
                if len(bars) > 0:
                    verified_symbols.add(symbol)
                    self.logger.info(f"Successfully verified {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                self.logger.warning(f"Could not verify {symbol}: {str(e)}")
        
        return verified_symbols

    def get_historical_data(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Fetch historical data using legacy REST client."""
        try:
            end = datetime.now()
            start = end - timedelta(days=limit)
            
            bars = self.api.get_bars(
                symbol,
                '1D',  # Changed from 'Day' to '1D'
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=limit
            ).df
            
            if len(bars) > 0:
                return bars
                
            self.logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_last_quote(self, symbol: str) -> float:
        """Get the latest quote for a symbol using the free data API."""
        try:
            # For free API, we'll use the latest trade instead of quote
            trade = self.data_api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return 0.0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        if df.empty:
            return df
            
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return pd.DataFrame()

    def check_trend_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if trend confirmation indicators align."""
        try:
            if df.empty or df.iloc[-1].isnull().any():
                return False
                
            latest = df.iloc[-1]
            return (
                latest['SMA50'] > latest['SMA200'] and  # Bullish bias
                latest['close'] > latest['SMA50'] and    # Price above moving average
                latest['volume'] > latest['volume_MA20']  # Strong volume
            )
        except Exception as e:
            self.logger.error(f"Error in trend confirmation: {str(e)}")
            return False

    def check_momentum_volatility(self, df: pd.DataFrame) -> bool:
        """Check momentum and volatility indicators."""
        try:
            if df.empty or df.iloc[-1].isnull().any():
                return False
                
            latest = df.iloc[-1]
            return (
                latest['plus_di'] > latest['minus_di'] and  # Positive directional movement
                30 <= latest['RSI'] <= 70                   # RSI in favorable zone
            )
        except Exception as e:
            self.logger.error(f"Error in momentum check: {str(e)}")
            return False

    def check_risk_reward(self, df: pd.DataFrame, risk_ratio: float = 2.0) -> bool:
        """Evaluate risk/reward ratio using Bollinger Bands."""
        try:
            if df.empty or df.iloc[-1].isnull().any():
                return False
                
            latest = df.iloc[-1]
            potential_reward = latest['BB_upper'] - latest['close']
            potential_risk = latest['close'] - latest['BB_lower']
            
            if potential_risk == 0:
                return False
                
            return potential_reward >= (risk_ratio * potential_risk)
        except Exception as e:
            self.logger.error(f"Error in risk/reward check: {str(e)}")
            return False

    def generate_signals(self, symbols: List[str], indicator_choice: str) -> Dict[str, bool]:
        """Generate trading signals based on selected indicators."""
        signals = {}
        
        # Verify symbols if not already verified
        if not self.verified_symbols:
            self.verified_symbols = self.verify_symbols(symbols)
        
        if not self.verified_symbols:
            self.logger.error("No valid symbols to trade!")
            return signals
            
        for symbol in self.verified_symbols:
            df = self.get_historical_data(symbol)
            if df.empty:
                continue
                
            df = self.calculate_indicators(df)
            if df.empty:
                continue

            try:
                if indicator_choice == '1':
                    signal = self.check_trend_confirmation(df)
                elif indicator_choice == '2':
                    signal = self.check_momentum_volatility(df)
                elif indicator_choice == '3':
                    signal = self.check_risk_reward(df)
                else:  # All indicators
                    trend = self.check_trend_confirmation(df)
                    momentum = self.check_momentum_volatility(df)
                    risk = self.check_risk_reward(df)
                    signal = trend and momentum and risk
                
                signals[symbol] = signal
                self.logger.info(f"Generated signal for {symbol}: {signal}")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
                continue
        
        return signals

    def execute_trades(self, signals: Dict[str, bool]):
        """Execute trades based on generated signals."""
        if not signals:
            self.logger.warning("No valid signals to execute trades")
            return

        for symbol, should_buy in signals.items():
            try:
                # Check current position
                try:
                    position = self.api.get_position(symbol)
                    has_position = True
                except:
                    has_position = False

                if has_position and not should_buy:
                    # Sell position
                    self.api.submit_order(
                        symbol=symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc',
                        feed='iex'
                    )
                    self.logger.info(f"Closed position in {symbol}")
                
                elif should_buy and not has_position:
                    # Get current price
                    last_trade = self.api.get_latest_trade(symbol)
                    price = float(last_trade.price)
                    
                    if price > 0:
                        # Calculate position size
                        shares = int(self.trading_cash / (len(signals) * price))
                        
                        if shares > 0:
                            self.api.submit_order(
                                symbol=symbol,
                                qty=shares,
                                side='buy',
                                type='market',
                                time_in_force='gtc',
                                feed='iex'
                            )
                            self.logger.info(f"Opened position in {symbol}: {shares} shares")
                
            except Exception as e:
                self.logger.error(f"Error executing trade for {symbol}: {str(e)}")

def main():
    """Main function to run the swing trading system."""
    API_KEY = os.getenv('SWING_API_KEY')
    API_SECRET = os.getenv('SWING_SECRET_KEY')
    
    if not API_KEY or not API_SECRET:
        print("Error: API credentials not found in environment variables")
        return
    
    trader = SwingTrader(API_KEY, API_SECRET)
    
    # Get user input
    symbols = input("\nEnter stock symbols (comma-separated): ").split(',')
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    
    if not symbols:
        print("Error: No symbols entered")
        return
        
    print("\nVerifying symbols...")
    verified = trader.verify_symbols(symbols)
    
    if not verified:
        print("Error: None of the entered symbols could be verified")
        return
        
    print(f"\nVerified symbols: {', '.join(verified)}")
    
    print("\nChoose indicator combination:")
    print("1: Trend Confirmation only")
    print("2: Momentum/Volatility only")
    print("3: Risk/Reward only")
    print("4: All indicators")
    
    choice = input("Enter your choice (1-4): ")
    if choice not in ['1', '2', '3', '4']:
        print("Error: Invalid choice")
        return
    
    print("\nStarting trading system...")
    
    while True:
        try:
            signals = trader.generate_signals(symbols, choice)
            if signals:
                trader.execute_trades(signals)
            trader.logger.info("Waiting for next iteration...")
            time.sleep(300)
            
        except KeyboardInterrupt:
            trader.logger.info("Stopping the trading system...")
            break
        except Exception as e:
            trader.logger.error(f"An error occurred: {str(e)}")
            break

if __name__ == "__main__":
    main()