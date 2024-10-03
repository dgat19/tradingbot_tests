import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

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

def get_volume_indicator(symbol, period='1mo', interval='1d'):
    stock_data = yf.download(symbol, period=period, interval=interval)
    if len(stock_data) < 2:
        return False  # Not enough data to determine high volume
    average_volume = np.mean(stock_data['Volume'].values)
    latest_volume = stock_data['Volume'].values[-1]
    
    volume_ratio = latest_volume / average_volume
    return volume_ratio > 1.5  # Return True if volume is 50% above average

def simple_linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

def get_trend_indicator(symbol, period='6mo', interval='1d'):
    stock_data = yf.download(symbol, period=period, interval=interval)
    
    if len(stock_data) < 2:
        return False  # Not enough data to determine trend
    
    x = np.arange(len(stock_data))
    y = stock_data['Close'].values
    
    slope, _ = simple_linear_regression(x, y)
    return slope > 0  # Return True if trend is positive

def analyze_indicators(symbol):
    try:
        volatility = get_stock_volatility(symbol)
        monthly_performance = get_historical_monthly_performance(symbol)
        volume_indicator = get_volume_indicator(symbol)
        trend_indicator = get_trend_indicator(symbol)
        
        return {
            'volatility': volatility,
            'monthly_performance': monthly_performance,
            'high_volume': volume_indicator,
            'positive_trend': trend_indicator
        }
    except Exception as e:
        print(f"Error analyzing indicators for {symbol}: {str(e)}")
        return {
            'volatility': 0,
            'monthly_performance': 0,
            'high_volume': False,
            'positive_trend': False
        }