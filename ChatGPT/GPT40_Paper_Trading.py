import yfinance as yf
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from alpaca_trade_api.rest import REST
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, OrderStatus
import time
import requests
import sys
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PK9RIB7H3DVU9FMHROR7"
ALPACA_API_SECRET = "dvwSlk4p1ZKBqPsLGJbehu1dAcd82MSwLJ5BgHVh"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url=ALPACA_BASE_URL)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
print("Starting the script...")

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")

# 1. Scrape News Articles
def scrape_news(stock_symbol):
    url = f"https://www.google.com/search?q={stock_symbol}+stock+news"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract news headlines
    headlines = [h.get_text() for h in soup.find_all('h3')]
    return headlines

# 2. Sentiment Analysis on Headlines
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    
    # Average sentiment score (positive if above 0.05, negative if below -0.05)
    avg_sentiment = np.mean(sentiments)
    return avg_sentiment

# 3. Get Stock Volatility from Yahoo Finance based on Historical Monthly Data
def get_stock_monthly_volatility(symbol):
    stock_data = yf.download(symbol, period='5y', interval='1d')
    stock_data['Month'] = stock_data.index.month
    
    # Calculate volatility (standard deviation) for each month
    monthly_volatility = stock_data.groupby('Month')['Close'].std()
    return monthly_volatility

# 4. Get Stock Volatility from Yahoo Finance for Current Period
def get_stock_volatility_yahoo(symbol):
    stock_data = yf.download(symbol, period='1mo', interval='1d')
    close_prices = stock_data['Close'].values
    
    # Calculate volatility as the standard deviation of daily closing prices
    volatility = np.std(close_prices)
    return volatility

# 5. Get Options Chain
def get_options_chain(symbol, current_price):
    try:
        stock = yf.Ticker(symbol)
        options_dates = stock.options  # Get available expiration dates
        current_date = datetime.today().date()

        # Find the closest expiration date within the next month
        nearest_expiration = None
        for date_str in options_dates:
            expiration_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if expiration_date <= current_date + timedelta(days=30):
                nearest_expiration = date_str
                break

        if not nearest_expiration:
            print(f"No options expiring within a month for {symbol}")
            return None

        # Get the options chain for that expiration date
        options_chain = stock.option_chain(nearest_expiration)

        # Filter options that are within 75% of the current stock price
        lower_bound = current_price * 0.75
        upper_bound = current_price * 1.25

        calls_within_range = options_chain.calls[
            (options_chain.calls['strike'] >= lower_bound) & (options_chain.calls['strike'] <= upper_bound)
        ]
        puts_within_range = options_chain.puts[
            (options_chain.puts['strike'] >= lower_bound) & (options_chain.puts['strike'] <= upper_bound)
        ]
        
        return calls_within_range, puts_within_range
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {e}")
        return None, None

# 6. Place Option Trade
def place_option_trade(symbol, contract_symbol, qty, option_type='call'):
    try:
        side = OrderSide.BUY if option_type == 'call' else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE
        )
        
        order = trading_client.submit_order(order_data=market_order_data)
        
        if order.status == OrderStatus.FILLED:
            print(f"Option order for {contract_symbol} ({side}) has been filled.")
        else:
            print(f"Option order for {contract_symbol} ({side}) is {order.status}.")
    except Exception as e:
        print(f"Error placing options trade for {symbol}: {e}")

# 7. Fetch Live Stock Data (this resolves the `fetch_live_data` error)
def fetch_live_data(symbol):
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period="1d", interval="1m")
    return stock_data.iloc[-1]

# 8. Main Automated Trading Loop
def automated_trading(stock_symbol, qty=1):
    headlines = scrape_news(stock_symbol)
    sentiment_score = analyze_sentiment(headlines)
    
    current_volatility_yahoo = get_stock_volatility_yahoo(stock_symbol)
    current_volatility = current_volatility_yahoo

    live_data = fetch_live_data(stock_symbol)
    current_price = live_data['Close']

    calls, puts = get_options_chain(stock_symbol, current_price)
    if calls is None or puts is None:
        return

    if sentiment_score > 0.05 and current_volatility > 1:
        contract_symbol = calls.iloc[0]['contractSymbol']
        place_option_trade(stock_symbol, contract_symbol, qty=qty, option_type='call')
    elif sentiment_score < -0.05 and current_volatility > 1:
        contract_symbol = puts.iloc[0]['contractSymbol']
        place_option_trade(stock_symbol, contract_symbol, qty=qty, option_type='put')
    else:
        print(f"No significant action for {stock_symbol} - sentiment: {sentiment_score}, volatility: {current_volatility}")

# 9. Continuous Trading
def continuous_trading(stock_list, qty=1, interval=180):
    while True:
        for stock_symbol in stock_list:
            try:
                automated_trading(stock_symbol, qty)
            except Exception as e:
                print(f"Error trading {stock_symbol}: {e}")
        
        for remaining in range(interval, 0, -1):
            sys.stdout.write(f"\rNext refresh in {remaining} seconds...   ")
            sys.stdout.flush()
            time.sleep(1)
        
        sys.stdout.write("\rNext refresh in 0 seconds...    \n")
        sys.stdout.flush()

# 9. Execute the Trading Strategy
if __name__ == "__main__":
    # List of stock symbols to monitor
    stock_list = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NVDA", "AVGO", "INTC", "ASTS", "LUNR"]

    # Start monitoring and trading options
    continuous_trading(stock_list, qty=1, interval=180)
