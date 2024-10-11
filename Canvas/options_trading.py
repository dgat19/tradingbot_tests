import os
import alpaca_trade_api as tradeapi
import requests
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class OptionsTrader:
    def __init__(self):
        # Initialize Alpaca API with secure API key management
        self.api = tradeapi.REST(
            os.getenv('PKV1PSBFZJSVP0SVHZ7U'),
            os.getenv('vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il'),
            'https://paper-api.alpaca.markets'
        )
        self.vix_symbol = '^VIX'  # VIX symbol for Yahoo Finance

    def execute_strategy(self):
        trades = []
        try:
            # Fetch market volatility data (VIX)
            vix_data = yf.download(self.vix_symbol, period='5d')
            market_volatility = vix_data['Close'].iloc[-1]

            # Fetch options data from Alpaca (Note: Alpaca may not support options data directly)
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            for asset in assets:
                symbol = asset.symbol

                # Fetch historical market data
                market_data = self.get_market_data(symbol, '5d')

                if market_data is None or len(market_data) < 5:
                    continue

                # Calculate volatility
                price_changes = market_data['Close'].pct_change().dropna()
                avg_volatility = np.std(price_changes) * np.sqrt(252)  # Annualized volatility

                # Get news sentiment
                news_sentiment = self.get_news_sentiment(symbol)

                # Get options Greeks
                option_greeks = self.get_option_greeks(symbol)

                # Adjusted thresholds based on statistical analysis
                if (avg_volatility > market_volatility * 1.2 and
                    news_sentiment > 0.2 and
                    option_greeks['delta'] > 0.5):

                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data['Close'].iloc[-1],
                        'volatility': avg_volatility,
                        'volume': 100,
                        'news_sentiment': news_sentiment,
                        'delta': option_greeks['delta'],
                        'theta': option_greeks['theta'],
                        'gamma': option_greeks['gamma'],
                        'unrealized_gain': 0.0,
                        'return': 0.0
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error fetching data or executing trade: {e}")
        
        return trades

    def backtest(self, start_date, end_date):
        trades = []
        try:
            # Fetch historical data for backtesting period
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            for asset in assets:
                symbol = asset.symbol

                # Fetch historical market data
                market_data = self.get_market_data(symbol, start_date, end_date)

                if market_data is None or len(market_data) < 5:
                    continue

                # Calculate volatility
                price_changes = market_data['Close'].pct_change().dropna()
                avg_volatility = np.std(price_changes) * np.sqrt(252)

                # Get news sentiment
                news_sentiment = self.get_news_sentiment(symbol)

                # Get options Greeks
                option_greeks = self.get_option_greeks(symbol)

                if (avg_volatility > 0.3 and  # Adjusted threshold
                    news_sentiment > 0.2 and
                    option_greeks['delta'] > 0.5):

                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data['Close'].iloc[-1],
                        'volatility': avg_volatility,
                        'volume': 100,
                        'news_sentiment': news_sentiment,
                        'delta': option_greeks['delta'],
                        'theta': option_greeks['theta'],
                        'gamma': option_greeks['gamma'],
                        'return': self.simulate_trade_return(symbol, market_data)
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error during backtesting: {e}")
        
        return trades

    def get_market_data(self, symbol, start_date=None, end_date=None):
        try:
            if start_date and end_date:
                data = yf.download(symbol, start=start_date, end=end_date)
            else:
                data = yf.download(symbol, period='5d')
            return data
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None

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

    def get_option_greeks(self, symbol):
        # Using Options API (e.g., from Tradier)
        try:
            api_key = os.getenv('TRADIER_API_KEY')
            url = f'https://api.tradier.com/v1/markets/options/chains?symbol={symbol}&expiration={self.get_next_expiration()}'
            headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
            response = requests.get(url, headers=headers)
            options_data = response.json()

            # Select an at-the-money option
            options = options_data.get('options', {}).get('option', [])
            atm_option = min(options, key=lambda x: abs(x['strike'] - x['last']))

            # Return the Greeks
            greeks = atm_option.get('greeks', {})
            return {
                'delta': greeks.get('delta', 0),
                'theta': greeks.get('theta', 0),
                'gamma': greeks.get('gamma', 0)
            }
        except Exception as e:
            print(f"Error fetching option greeks for {symbol}: {e}")
            return {'delta': 0, 'theta': 0, 'gamma': 0}

    def get_next_expiration(self):
        # Get the next monthly options expiration date
        today = datetime.now().date()
        third_friday = today + timedelta((4 - today.weekday() + 7) % 7 + 14)
        return third_friday.isoformat()

    def simulate_trade_return(self, symbol, market_data):
        # Simulate trade return based on historical data
        try:
            entry_price = market_data['Close'].iloc[-1]
            future_prices = market_data['Close'].iloc[-5:]
            exit_price = future_prices.mean()
            trade_return = (exit_price - entry_price) / entry_price
            return trade_return
        except Exception as e:
            print(f"Error simulating trade return for {symbol}: {e}")
            return 0.0
