import yfinance as yf
from newsapi import NewsApiClient
import alpaca_trade_api as tradeapi
import datetime as dt
import talib as ta


# Define your stock symbol and options expiration date
stock_symbol = input('Input stock symbol: ').upper()

# Initialize Alpaca API (Paper trading)
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
APCA_API_KEY_ID = 'PA3NRFGUO5AU'
APCA_API_SECRET_KEY = 'PKLLPAIZVPAFBTCF72XM'

api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Initialize News API
newsapi = NewsApiClient(api_key = 'bcec8fa2304344b5892af472fab2a6b0')


def get_stock_data(symbol):
    # Fetch historical data
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    return hist

def apply_technical_indicators(data):
    # Calculate moving averages
    data['SMA_50'] = ta.SMA(data['Close'], timeperiod=50)
    data['SMA_200'] = ta.SMA(data['Close'], timeperiod=200)
    
    # RSI
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    
    # MACD
    data['MACD'], data['MACD_signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    return data

def analyze_data(data):
    # Simple strategy based on SMA crossover
    data['Signal'] = 0
    data['Signal'][data['SMA_50'] > data['SMA_200']] = 1
    data['Position'] = data['Signal'].diff()
    
    return data

def fetch_latest_news(symbol):
    # Fetch the latest news related to the stock symbol
    today = dt.datetime.now().strftime('%Y-%m-%d')
    all_articles = newsapi.get_everything(q = symbol,
                                          from_param = today,
                                          to = today,
                                          language = 'en',
                                          sort_by = 'relevancy')
    return all_articles['articles']

def place_order(symbol, order_type, qty):
    try:
        # Check if the market is open
        clock = api.get_clock()
        if clock.is_open:
            # Place a market order
            order = api.submit_order(
                symbol = symbol,
                qty = qty,
                side = order_type,
                type = 'market',
                time_in_force = 'gtc'
            )
            print(f"Order placed: {order_type} {qty} shares of {symbol}")
        else:
            print("Market is closed. Cannot place order.")
    except Exception as e:
        print(f"Error placing order: {str(e)}")

def strategy(data, symbol, exp_date):
    # Placeholder for a trading strategy. Buy calls if bullish and puts if bearish
    if data['Position'].iloc[-1] == 1:
        print(f"Buy Call Option for {symbol} expiring on {exp_date}")
        place_order(symbol, 'buy', 1)
    elif data['Position'].iloc[-1] == -1:
        print(f"Buy Put Option for {symbol} expiring on {exp_date}")
        place_order(symbol, 'sell', 1)
    else:
        print("No trading signal at the moment.")

def get_future_fridays(n=10):
    # Get a list of the next n Fridays
    today = dt.date.today()
    fridays = []
    while len(fridays) < n:
        today += dt.timedelta(days=1)
        if today.weekday() == 4:  # 4 corresponds to Friday
            fridays.append(today.strftime('%Y-%m-%d'))
    return fridays

def main():
    # Get future Fridays and prompt user to select one
    future_fridays = get_future_fridays()
    print("Select an expiration date from the following list of Fridays:")
    for i, date in enumerate(future_fridays):
        print(f"{i + 1}: {date}")
    
    choice = int(input("Enter the number corresponding to your chosen date: ")) - 1
    expiration_date = future_fridays[choice]

    # Fetch stock data
    stock_data = get_stock_data(stock_symbol)
    
    # Apply technical indicators
    stock_data = apply_technical_indicators(stock_data)
    
    # Analyze data for trading signals
    stock_data = analyze_data(stock_data)
    
    # Execute strategy based on the analysis
    strategy(stock_data, stock_symbol, expiration_date)
    
    # Fetch and display the latest news
    news = fetch_latest_news(stock_symbol)
    for article in news:
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"Published At: {article['publishedAt']}")
        print(f"URL: {article['url']}\n")
    
if __name__ == "__main__":
    main()
