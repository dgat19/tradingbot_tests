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

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKLSUU1T53HAXFDKFQMY"
ALPACA_API_SECRET = "M46BGIZBuunwXIgDU1ttxnNj0nURPZfxt1IjLkdr"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url=ALPACA_BASE_URL)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

print("Starting the script...")

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
    print(f"Account cash withdrawal: {account.options_buying_power}")
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
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE
        )
        
        # Submit the order and capture the response
        order = trading_client.submit_order(order_data=market_order_data)
        
        # Check the order status
        if order.status == OrderStatus.FILLED:
            print(f"Option order for {contract_symbol} ({side}) has been filled.")
            return order  # Return the order for further processing
        else:
            print(f"Option order for {contract_symbol} ({side}) is {order.status}.")
            return None  # No valid order
    except Exception as e:
        print(f"Error placing options trade for {symbol}: {e}")
        return None

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

# 7. Check and Close Trade at 75% Gain
def check_and_close_trade(entry_price, contract_symbol, qty):
    try:
        # Fetch current price of the option
        option_data = yf.Ticker(contract_symbol).history(period="1d", interval="1m")
        current_price = option_data['Close'].iloc[-1]  # Get the last close price
        
        # Calculate gain percentage
        gain_percentage = ((current_price - entry_price) / entry_price) * 100
        
        if gain_percentage >= 75:
            print(f"Closing trade for {contract_symbol} at a gain of {gain_percentage:.2f}%")
            place_option_trade(contract_symbol, contract_symbol, qty, option_type='sell')  # Place sell order
        else:
            print(f"Current gain for {contract_symbol} is {gain_percentage:.2f}%, waiting to close.")
    except Exception as e:
        print(f"Error checking trade for {contract_symbol}: {e}")

# 8. Main Automated Trading Loop
def automated_trading(stock_symbol, qty=1):
    # Step 1: Scrape news and perform sentiment analysis
    headlines = scrape_news(stock_symbol)
    sentiment_score = analyze_sentiment(headlines)
    
    # Step 2: Get stock data and calculate volatility
    current_volatility = get_stock_volatility_yahoo(stock_symbol)

    # Step 3: Get the options chain for the stock
    options_chain = get_options_chain(stock_symbol)
    if options_chain is None:
        return

    # Choose a strike price for the option (for simplicity, choosing the first option)
    option_contract = options_chain.calls.iloc[0] if sentiment_score > 0.05 else options_chain.puts.iloc[0]
    contract_symbol = option_contract['contractSymbol']

    # Step 4: Decision logic based on sentiment and volatility
    order = None
    if sentiment_score > 0.05 and current_volatility > 1:
        order = place_option_trade(stock_symbol, contract_symbol, qty=qty, option_type='call')
    elif sentiment_score < -0.05 and current_volatility > 1:
        order = place_option_trade(stock_symbol, contract_symbol, qty=qty, option_type='put')
    else:
        print(f"No significant action for {stock_symbol} - sentiment: {sentiment_score}, volatility: {current_volatility}")
    
    # Step 5: Track the entry price and check for closure
    if order:
        entry_price = option_contract['lastPrice']  # Assuming lastPrice from the option contract
        check_and_close_trade(entry_price, contract_symbol, qty)

# 9. Continuous Trading
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
            sys.stdout.write(f"\rNext refresh in {remaining} seconds...   ")
            sys.stdout.flush()
            time.sleep(1)
        
        sys.stdout.write("\rNext refresh in 0 seconds...    \n")
        sys.stdout.flush()

# 10. Execute the Trading Strategy
if __name__ == "__main__":
    # List of stock symbols to monitor
    stock_list = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NVDA", "AVGO", "INTC", "ASTS", "LUNR"]

    # Start monitoring and trading options
    continuous_trading(stock_list, qty=1, interval=300)
