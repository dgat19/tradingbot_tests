import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
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
    
    # Use numpy for efficient calculations
    monthly_returns = np.diff(data['Close'].resample('ME').last().values) / data['Close'].resample('ME').last().values[:-1]
    monthly_performance = np.zeros(12)
    for month in range(1, 13):
        month_mask = data.index.month == month
        if np.any(month_mask[:-1]):
            monthly_performance[month-1] = np.mean(monthly_returns[month_mask[:-1]])
    
    current_month = datetime.now().month
    current_month_performance = monthly_performance[current_month-1]
    
    return current_month_performance

def get_volume_indicator(symbol, period='1mo', interval='1d'):
    stock_data = yf.download(symbol, period=period, interval=interval)
    average_volume = np.mean(stock_data['Volume'].values)
    latest_volume = stock_data['Volume'].values[-1]
    
    volume_ratio = latest_volume / average_volume
    return volume_ratio > 1.5  # Return True if volume is 50% above average

def get_trend_indicator(symbol, period='6mo', interval='1d'):
    stock_data = yf.download(symbol, period=period, interval=interval)

    
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y = stock_data['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    trend_slope = model.coef_[0]
    return trend_slope > 0  # Return True if trend is positive

def analyze_indicators(symbol):
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