# potential_movers.py
import os
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from alpaca_trade_api import REST
from dotenv import load_dotenv

load_dotenv()

class PotentialMovers:
    def __init__(self):
        # Initialize Alpaca API
        self.api = REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            'https://paper-api.alpaca.markets/v2'
        )

    def get_top_movers(self):
        movers = []
        try:
            response = requests.get('https://finance.yahoo.com/most-active')
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'class': 'W(100%)'})
                
                # Check if the table was found
                if table:
                    rows = table.find_all('tr')[1:11]  # Get top 10 movers
                    for row in rows:
                        cols = row.find_all('td')
                        symbol = cols[0].text.strip()
                        movers.append(symbol)
                else:
                    print("Warning: Top movers table not found.")
            else:
                print(f"Error: Unable to fetch top movers. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching top movers: {e}")
        
        return movers

    def execute_strategy(self):
        trades = []
        try:
            top_movers = self.get_top_movers()
            for symbol in top_movers:
                # Fetch recent market data
                market_data = yf.download(symbol, period='2d', interval='1m', prepost=True)
                
                if market_data.empty:
                    continue

                # Example condition: Enter trade if a volume spike is detected
                pre_market_data = market_data.between_time('04:00', '09:30')
                if pre_market_data['Volume'].sum() > market_data['Volume'].sum() * 0.5:
                    trade = {
                        'symbol': symbol,
                        'entry_price': market_data['Close'].iloc[-1],
                        'volume': 100,
                        'return': 0.0  # Placeholder, actual return can be calculated after exit
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"Error executing potential movers strategy: {e}")
        
        return trades