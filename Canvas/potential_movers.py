import os
import alpaca_trade_api as tradeapi
import requests
import yfinance as yf
from bs4 import BeautifulSoup

class PotentialMovers:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('PKV1PSBFZJSVP0SVHZ7U'),
            os.getenv('vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il'),
            'https://paper-api.alpaca.markets'
        )
        
    def get_top_movers(self):
        movers = []
        try:
            # Fetch top movers using Yahoo Finance API
            response = requests.get('https://finance.yahoo.com/most-active')
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'W(100%)'})
            rows = table.find_all('tr')[1:11]  # Get top 10 movers

            for row in rows:
                cols = row.find_all('td')
                symbol = cols[0].text.strip()
                movers.append(symbol)
        except Exception as e:
            print(f"Error fetching top movers: {e}")
        
        return movers

    def execute_strategy(self):
        trades = []
        try:
            top_movers = self.get_top_movers()
            for symbol in top_movers:
                # Fetch pre-market data
                market_data = yf.download(symbol, period='2d', interval='1m', prepost=True)

                if market_data.empty:
                    continue

                # Check for significant pre-market volume spike
                pre_market_data = market_data.between_time('04:00', '09:30')
                if pre_market_data['Volume'].sum() > market_data['Volume'].sum() * 0.5:
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data['Close'].iloc[-1],
                        'volume': 100,
                        'unrealized_gain': 0.0,
                        'return': 0.0
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error executing potential movers strategy: {e}")
        
        return trades

    def backtest(self, start_date, end_date):
        trades = []
        try:
            top_movers = self.get_top_movers()
            for symbol in top_movers:
                market_data = yf.download(symbol, start=start_date, end=end_date)

                if market_data is None or market_data.empty:
                    continue

                # Check for significant price movement
                if market_data['Close'].iloc[-1] > market_data['Open'].iloc[-1] * 1.05:
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data['Close'].iloc[-1],
                        'volume': 100,
                        'return': self.simulate_trade_return(symbol, market_data)
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error during backtesting: {e}")
        
        return trades

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
