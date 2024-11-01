import os
import alpaca_trade_api as tradeapi
import numpy as np
import yfinance as yf
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD
from dotenv import load_dotenv

load_dotenv()

class SwingTrader:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_SECRET_KEY'),
            'https://paper-api.alpaca.markets/v2'
        )
        
    def execute_strategy(self):
        trades = []
        try:
            # Fetch active assets
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            for asset in assets:
                symbol = asset.symbol

                # Fetch historical market data
                market_data = yf.download(symbol, period='1mo')

                if market_data is None or market_data.empty or len(market_data) < 20:
                    continue

                # Calculate indicators
                market_data['RSI'] = RSIIndicator(market_data['Close']).rsi()
                macd_indicator = MACD(market_data['Close'])
                market_data['MACD'] = macd_indicator.macd()
                market_data['Signal'] = macd_indicator.macd_signal()

                latest_data = market_data.iloc[-1]

                # Check for bullish signals
                if (latest_data['RSI'] < 30 and
                    latest_data['MACD'] > latest_data['Signal']):

                    trade = {
                        'symbol': symbol,
                        'entry_price': latest_data['Close'],
                        'volatility': np.std(market_data['Close'].pct_change().dropna()) * np.sqrt(252),
                        'volume': 100,
                        'news_sentiment': self.get_news_sentiment(symbol),
                        'unrealized_gain': 0.0,
                        'return': 0.0
                    }
                    trades.append(trade)
            return trades
        except Exception as e:
            print(f"Error fetching data or executing trade: {e}")
            return trades

    def backtest(self, start_date, end_date):
        trades = []
        try:
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            for asset in assets:
                symbol = asset.symbol
                market_data = yf.download(symbol, start=start_date, end=end_date)

                if market_data is None or market_data.empty or len(market_data) < 20:
                    continue

                # Calculate indicators
                market_data['RSI'] = RSIIndicator(market_data['Close']).rsi()
                macd_indicator = MACD(market_data['Close'])
                market_data['MACD'] = macd_indicator.macd()
                market_data['Signal'] = macd_indicator.macd_signal()

                latest_data = market_data.iloc[-1]

                # Check for bullish signals
                if (latest_data['RSI'] < 30 and
                    latest_data['MACD'] > latest_data['Signal']):

                    trade = {
                        'symbol': symbol,
                        'entry_price': latest_data['Close'],
                        'volatility': np.std(market_data['Close'].pct_change().dropna()) * np.sqrt(252),
                        'volume': 100,
                        'return': self.simulate_trade_return(symbol, market_data)
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error during backtesting: {e}")
        
        return trades

    def get_news_sentiment(self, symbol):
        # Using Finnhub API for news sentiment
        try:
            api_key = os.getenv('FINNHUB_API_KEY')
            url = f'https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={api_key}'
            response = requests.get(url)
            data = response.json()
            sentiment_score = data.get('sentiment', {}).get('score', 0)
            return sentiment_score
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {e}")
            return 0.0

    def simulate_trade_return(self, symbol, market_data):
        try:
            entry_price = market_data['Close'].iloc[-1]
            future_prices = market_data['Close'].iloc[-5:]
            exit_price = future_prices.mean()
            trade_return = (exit_price - entry_price) / entry_price
            return trade_return
        except Exception as e:
            print(f"Error simulating trade return for {symbol}: {e}")
            return 0.0
