import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(symbol, period="1mo"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data

def get_stock_volatility(symbol, period='1mo', interval='1d'):
    stock_data = yf.download(symbol, period=period, interval=interval)
    close_prices = stock_data['Close'].values
    volatility = np.std(close_prices)
    return volatility

def get_historical_monthly_performance(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if len(data) < 30:  # If we have less than a month of data
        return 0  # Return neutral performance
    
    monthly_data = data['Close'].resample('ME').last()
    monthly_returns = monthly_data.pct_change().dropna()
    
    if len(monthly_returns) == 0:
        return 0  # Return neutral performance if we can't calculate returns
    
    current_month = datetime.now().month
    current_month_performance = monthly_returns.groupby(monthly_returns.index.month).mean().get(current_month, 0)
    
    return current_month_performance

#def get_volume_indicator(symbol, period='1mo', interval='1d'):
    #stock_data = yf.download(symbol, period=period, interval=interval)
    #if len(stock_data) < 2:
        #return False  # Not enough data to determine high volume
    #average_volume = np.mean(stock_data['Volume'].values)
    #latest_volume = stock_data['Volume'].values[-1]
    
    #volume_ratio = latest_volume / average_volume
    #return volume_ratio > 1.5  # Return True if volume is 50% above average
def get_volume_indicator(stock_symbol):
    stock_data = yf.Ticker(stock_symbol).history(period="1mo")
    avg_volume = stock_data['Volume'].mean()
    current_volume = stock_data['Volume'].iloc[-1]
    
    return current_volume > avg_volume

def simple_linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

#def get_trend_indicator(symbol, period='6mo', interval='1d'):
    #stock_data = yf.download(symbol, period=period, interval=interval)
    
    #if len(stock_data) < 2:
       # return False  # Not enough data to determine trend
    
    #x = np.arange(len(stock_data))
    #y = stock_data['Close'].values
    
    #slope, _ = simple_linear_regression(x, y)
    #return slope > 0  # Return True if trend is positive
def get_trend_indicator(stock_symbol):
    stock_data = yf.Ticker(stock_symbol).history(period="6mo")
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    
    # If 50-SMA > 200-SMA, upward trend, else downward
    return 1 if stock_data['SMA_50'].iloc[-1] > stock_data['SMA_200'].iloc[-1] else -1

def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return macd, signal

def get_bollinger_bands(stock_symbol, window=20):
    stock_data = yf.Ticker(stock_symbol).history(period="6mo")
    stock_data['SMA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['stddev'] = stock_data['Close'].rolling(window=window).std()
    
    stock_data['Upper Band'] = stock_data['SMA'] + (stock_data['stddev'] * 2)
    stock_data['Lower Band'] = stock_data['SMA'] - (stock_data['stddev'] * 2)
    
    latest_data = stock_data.iloc[-1]
    return latest_data['Upper Band'], latest_data['Lower Band']

def analyze_indicators(symbol):
    data = get_stock_data(symbol, period="1mo")
    
    # Calculate Indicators
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)
    
    # Example Indicator Analysis
    rsi = data['RSI'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_Signal'].iloc[-1]
    
    positive_trend = macd > macd_signal
    high_volume = data['Volume'].iloc[-1] > data['Volume'].mean()
    
    return {
        "monthly_performance": data['Close'].pct_change(periods=20).iloc[-1],  # Monthly performance
        "positive_trend": positive_trend,
        "high_volume": high_volume,
        "rsi": rsi
    }