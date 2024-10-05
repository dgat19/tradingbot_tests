import alpaca_trade_api as tradeapi
import requests
import numpy as np

class SwingTrader:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST('PKV1PSBFZJSVP0SVHZ7U', 'vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il', 'https://paper-api.alpaca.markets/v2')
        
    def execute_strategy(self):
        trades = []
        try:
            # Fetch active assets
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            for asset in assets:
                symbol = asset.symbol
                market_data = self.api.get_barset(symbol, 'day', limit=20)[symbol]
                
                if len(market_data) < 20:
                    continue
                
                # Calculate moving averages for swing trading
                close_prices = [bar.c for bar in market_data]
                short_ma = np.mean(close_prices[-5:])
                long_ma = np.mean(close_prices[-20:])

                # Check for breakout pattern
                if short_ma > long_ma * 1.05:  # Example breakout condition
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data[-1].c,
                        'volatility': np.std(close_prices),
                        'volume': 100,
                        'news_sentiment': self.get_news_sentiment(symbol),
                        'unrealized_gain': 0.0,
                        'return': 0.0
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error fetching data or executing trade: {e}")
        
        return trades

    def backtest(self, start_date, end_date):
        # Backtest the strategy using historical data
        trades = []
        try:
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            for asset in assets:
                symbol = asset.symbol
                market_data = self.api.get_barset(symbol, 'day', start=start_date.isoformat(), end=end_date.isoformat(), limit=1000)[symbol]
                
                if len(market_data) < 20:
                    continue
                
                # Calculate moving averages for swing trading
                close_prices = [bar.c for bar in market_data]
                short_ma = np.mean(close_prices[-5:])
                long_ma = np.mean(close_prices[-20:])

                # Check for breakout pattern
                if short_ma > long_ma * 1.05:
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data[-1].c,
                        'volatility': np.std(close_prices),
                        'volume': 100,
                        'return': np.random.uniform(-0.1, 0.3)  # Simulated return for backtesting
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error during backtesting: {e}")
        
        return trades

    def get_news_sentiment(self, symbol):
        # Mock function to get news sentiment (you can use actual sentiment analysis API)
        return np.random.uniform(0, 1)