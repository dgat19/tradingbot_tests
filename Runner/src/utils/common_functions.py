import yfinance as yf
import pandas as pd
from data.news_scraper import get_news_sentiment

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

# Fetch current stock data from Yahoo Finance
def get_current_stock_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="1d", interval="1d")

        if stock_data.empty:
            print(f"No data available for {stock_symbol}")
            return None

        current_price = stock_data['Close'].iloc[-1]
        open_price = stock_data['Open'].iloc[-1]
        current_volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock.history(period="1mo", interval="1d")['Volume'].mean()

        return {
            'current_price': current_price,
            'open_price': open_price,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }

    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        return None

# Utility function to fetch stock volatility
def get_stock_volatility(stock_symbol):
    stock_data = yf.download(stock_symbol, period="1mo", interval="1d")
    if stock_data.empty:
        return 0
    return stock_data['Close'].std()

# Function to calculate market sentiment based on news analysis
def calculate_market_sentiment(stock_symbol):
    sentiment = get_news_sentiment(stock_symbol)  # Fetch sentiment from news_scraper
    return sentiment