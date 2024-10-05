import alpaca_trade_api as tradeapi
import requests
import numpy as np
from bs4 import BeautifulSoup

class PotentialMovers:
    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST('PKV1PSBFZJSVP0SVHZ7U', 'vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il', 'https://paper-api.alpaca.markets')
        
    def get_top_movers(self):
        movers = []
        try:
            # Fetch top movers from Yahoo Finance or another reliable source
            response = requests.get('https://finance.yahoo.com/markets/stocks/gainers/')
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'W(100%)'})
            rows = table.find_all('tr')[1:]
            
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
                # Fetch market data for each symbol
                market_data = self.api.get_barset(symbol, 'day', limit=5)[symbol]
                
                if len(market_data) < 5:
                    continue
                
                # Example criteria: Breakout based on previous day close and volume
                if market_data[-1].c > market_data[-1].o * 1.05:
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data[-1].c,
                        'volume': 100,
                        'unrealized_gain': 0.0,
                        'return': 0.0
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error executing potential movers strategy: {e}")
        
        return trades

    def backtest(self, start_date, end_date):
        # Backtest the strategy using historical data
        trades = []
        try:
            top_movers = self.get_top_movers()
            for symbol in top_movers:
                market_data = self.api.get_barset(symbol, 'day', start=start_date.isoformat(), end=end_date.isoformat(), limit=1000)[symbol]
                
                if len(market_data) < 5:
                    continue
                
                # Example criteria: Breakout based on previous day close and volume
                if market_data[-1].c > market_data[-1].o * 1.05:
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data[-1].c,
                        'volume': 100,
                        'return': np.random.uniform(-0.1, 0.3)  # Simulated return for backtesting
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error during backtesting: {e}")
        
        return trades