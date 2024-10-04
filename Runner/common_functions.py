import yfinance as yf
from news_scraper import get_news_sentiment

# Function to fetch stock information from yfinance
def get_stock_info(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    stock_data = stock.history(period="1mo")
    
    if not stock_data.empty:
        price = stock_data['Close'].iloc[-1]
        open_price = stock_data['Open'].iloc[-1]
        if open_price == 0:
            day_change = 0
        else:
            day_change = ((price - open_price) / open_price) * 100
        volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock_data['Volume'].mean()
    else:
        price, day_change, volume, avg_volume = 0, 0, 0, 0

    return {
        'price': price,
        'day_change': day_change,
        'volume': volume,
        'avg_volume': avg_volume
    }



def calculate_market_sentiment(stock_symbol):
    """
    Function to calculate market sentiment based on news analysis.
    Replace this with your actual sentiment calculation function.
    """
    sentiment = get_news_sentiment(stock_symbol)  # Fetch sentiment from news_scraper
    return sentiment