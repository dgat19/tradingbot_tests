import yfinance as yf
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from alpaca_trade_api.rest import REST #, TimeFrame
#from alpaca_trade_api.entity import Order
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, OrderStatus
import time
import requests
from bs4 import BeautifulSoup

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

# 3. Get Stock Volatility from Yahoo Finance
def get_stock_volatility_yahoo(symbol):
    stock_data = yf.download(symbol, period='1mo', interval='1d')
    close_prices = stock_data['Close'].values
    
    # Calculate volatility as the standard deviation of daily closing prices
    volatility = np.std(close_prices)
    return volatility

# 4. Get Options Chain
def get_options_chain(symbol):
    try:
        stock = yf.Ticker(symbol)
        options_dates = stock.options  # Get available expiration dates
        latest_date = options_dates[-1]  # Choose the latest expiration date
        
        options_chain = stock.option_chain(latest_date)
        return options_chain
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {e}")
        return None

# 5. Place Option Trade
def place_option_trade(symbol, contract_symbol, qty, option_type='call'):
    try:
        side = OrderSide.BUY if option_type == 'call' else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=contract_symbol,  # This is the options contract symbol
            qty=qty,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.SIMPLE
        )
        
        # Submit the order and capture the response
        order = trading_client.submit_order(order_data=market_order_data)
        
        # Check the order status
        if order.status == OrderStatus.FILLED:
            print(f"Option order for {contract_symbol} ({side}) has been filled.")
        else:
            print(f"Option order for {contract_symbol} ({side}) is {order.status}.")
    except Exception as e:
        print(f"Error placing options trade for {symbol}: {e}")

# 6. Fetch live data for stock symbols
def fetch_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d", interval="1m")
        latest_data = data.iloc[-1]  # Get the latest minute data
        print(f"Live data for {symbol} - Close: {latest_data['Close']}, High: {latest_data['High']}, Low: {latest_data['Low']}, Open: {latest_data['Open']}, Volume: {latest_data['Volume']}")
        return latest_data
    except Exception as e:
        print(f"Error fetching live data for {symbol}: {e}")
        return None

# 7. Main Automated Trading Loop
def automated_trading(stock_symbol, qty=1):
    # Step 1: Scrape news and perform sentiment analysis
    headlines = scrape_news(stock_symbol)
    sentiment_score = analyze_sentiment(headlines)
    
    # Step 2: Get stock data and calculate volatility
    current_volatility_yahoo = get_stock_volatility_yahoo(stock_symbol)
    current_volatility = current_volatility_yahoo  # Use Yahoo volatility for simplicity

    live_data = fetch_live_data(stock_symbol)

    # Step 3: Get the options chain for the stock
    options_chain = get_options_chain(stock_symbol)
    if options_chain is None:
        return

    # Choose a strike price for the option (for simplicity, choosing the first option)
    option_contract = options_chain.calls.iloc[0] if sentiment_score > 0.05 else options_chain.puts.iloc[0]
    contract_symbol = option_contract['contractSymbol']

    # Step 4: Decision logic based on sentiment and volatility
    if sentiment_score > 0.04 and current_volatility > 1:
        place_option_trade(stock_symbol, contract_symbol, qty=qty, option_type='call')
    elif sentiment_score < -0.04 and current_volatility > 1:
        place_option_trade(stock_symbol, contract_symbol, qty=qty, option_type='put')
    else:
        print(f"No significant action for {stock_symbol} - sentiment: {sentiment_score}, volatility: {current_volatility}, live data: {live_data}")

# 8. Continuous Trading
def continuous_trading(stock_list, qty=1, interval=300):
    while True:
        for stock_symbol in stock_list:
            try:
                # Check for trading opportunities
                automated_trading(stock_symbol, qty)
            except Exception as e:
                print(f"Error trading {stock_symbol}: {e}")
        
        # Countdown timer
        for remaining in range(interval, 0, -1):
            print(f"Next refresh in {remaining} seconds...", end='\r')
            time.sleep(1)

        print() 

# 9. Execute the Trading Strategy
if __name__ == "__main__":
    # List of stock symbols to monitor
    stock_list = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT"]

    # Start monitoring and trading options
    continuous_trading(stock_list, qty=1, interval=300)
