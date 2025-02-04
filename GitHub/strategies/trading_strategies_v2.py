from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from pymongo import MongoClient
import pywt
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

#V2 is now available so please use those methods instead.
def get_historical_data(ticker, client, days=100):
    """
    Fetch historical bar data for a given stock ticker.
    
    :param ticker: The stock ticker symbol.
    :param client: An instance of StockHistoricalDataClient.
    :param days: Number of days of historical data to fetch.
    :return: DataFrame with historical stock bar data.
    """
    start_time = datetime.now() - timedelta(days=days)  # Get data for the past 'days' days
    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_time
    )
    
    bars = client.get_stock_bars(request_params)
    data = bars.df  # Returns a pandas DataFrame
    return data

def rsi_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    RSI strategy: Buy when RSI is oversold, sell when overbought.
    """
    window = 14
    max_investment = total_portfolio_value * 0.10

    # Calculate RSI
    delta = historical_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]

    # Buy signal: RSI below 30 (oversold)
    if current_rsi < 30 and account_cash > 0:
        quantity_to_buy = min(int(max_investment // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal: RSI above 70 (overbought)
    elif current_rsi > 70 and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    # No action triggered
    return ('hold', portfolio_qty, ticker)

def bollinger_bands_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Bollinger Bands strategy: Buy when price touches lower band, sell when it touches upper band.
    """
    window = 20
    num_std = 2
    max_investment = total_portfolio_value * 0.10

    historical_data['MA'] = historical_data['close'].rolling(window=window).mean()
    historical_data['STD'] = historical_data['close'].rolling(window=window).std()
    historical_data['Upper'] = historical_data['MA'] + (num_std * historical_data['STD'])
    historical_data['Lower'] = historical_data['MA'] - (num_std * historical_data['STD'])

    upper_band = historical_data['Upper'].iloc[-1]
    lower_band = historical_data['Lower'].iloc[-1]

    # Buy signal: Price at or below lower band
    if current_price <= lower_band and account_cash > 0:
        quantity_to_buy = min(int(max_investment // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal: Price at or above upper band
    elif current_price >= upper_band and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    # No action triggered
    return ('hold', portfolio_qty, ticker)

def macd_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    MACD strategy: Buy when MACD line crosses above signal line, sell when it crosses below.
    """
    max_investment = total_portfolio_value * 0.10

    # Calculate MACD
    exp1 = historical_data['close'].ewm(span=12, adjust=False).mean()
    exp2 = historical_data['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # Get the last two MACD and signal values
    macd_current, macd_prev = macd.iloc[-1], macd.iloc[-2]
    signal_current, signal_prev = signal.iloc[-1], signal.iloc[-2]

    # Buy signal: MACD line crosses above signal line
    if macd_prev <= signal_prev and macd_current > signal_current and account_cash > 0:
        quantity_to_buy = min(int(max_investment // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal: MACD line crosses below signal line
    elif macd_prev >= signal_prev and macd_current < signal_current and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    # No action triggered
    return ('hold', portfolio_qty, ticker)

def momentum_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Momentum strategy logic to determine buy or sell signals based on short and long moving averages.
    Limits the amount to invest to less than 10% of the total portfolio.
    """
    # Maximum percentage of portfolio to invest per trade
    max_investment_percentage = 0.10  # 10% of total portfolio value
    max_investment = total_portfolio_value * max_investment_percentage

    # Momentum Logic
    short_window = 10
    long_window = 50
    
    short_ma = historical_data['close'].rolling(short_window).mean().iloc[-1]
    long_ma = historical_data['close'].rolling(long_window).mean().iloc[-1]

    # Buy signal (short MA crosses above long MA)
    if short_ma > long_ma and account_cash > 0:
        # Calculate amount to invest based on available cash and max investment
        amount_to_invest = min(account_cash, max_investment)
        quantity_to_buy = int(amount_to_invest // current_price)  # Calculate quantity to buy

        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal (short MA crosses below long MA)
    elif short_ma < long_ma and portfolio_qty > 0:
        # Sell 50% of the current holding, at least 1 share
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    # No action triggered
    return ('hold', portfolio_qty, ticker)

def mean_reversion_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Mean reversion strategy: Buy if the stock price is below the moving average, sell if above.

    :param ticker: The stock ticker symbol.
    :param current_price: The current price of the stock.
    :param historical_data: Historical stock data for the ticker.
    :param account_cash: Available cash in the account.
    :param portfolio_qty: Quantity of stock held in the portfolio.
    :param total_portfolio_value: Total value of the portfolio.
    :return: Tuple (action, quantity, ticker).
    """
    
    # Calculate moving average
    window = 20  # Use a 20-day moving average
    historical_data['MA20'] = historical_data['close'].rolling(window=window).mean()
    
    # Drop NaN values after creating the moving average
    historical_data.dropna(inplace=True)
    
    # Define max investment (10% of total portfolio value)
    max_investment = total_portfolio_value * 0.10

    # Buy signal: if current price is below the moving average by more than 2%
    if current_price < historical_data['MA20'].iloc[-1] * 0.98 and account_cash > 0:
        quantity_to_buy = min(int(max_investment // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)
    
    # Sell signal: if current price is above the moving average by more than 2%
    elif current_price > historical_data['MA20'].iloc[-1] * 1.02 and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))  # Sell 50% of portfolio, or at least 1
        return ('sell', quantity_to_sell, ticker)
    
    # No action triggered
    return ('hold', portfolio_qty, ticker)

# Keltner Channel Strategy
def keltner_channel_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Keltner Channel Strategy for buy/sell signals.
    """
    atr_window = 14
    ema_window = 20
    multiplier = 2

    historical_data['ATR'] = historical_data['close'].rolling(window=atr_window).apply(lambda x: max(x) - min(x))
    historical_data['EMA'] = historical_data['close'].ewm(span=ema_window, adjust=False).mean()
    historical_data['Upper'] = historical_data['EMA'] + (multiplier * historical_data['ATR'])
    historical_data['Lower'] = historical_data['EMA'] - (multiplier * historical_data['ATR'])

    # Buy signal
    if current_price < historical_data['Lower'].iloc[-1] and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal
    elif current_price > historical_data['Upper'].iloc[-1] and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Dual Thrust Strategy
def dual_thrust_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Dual Thrust Strategy based on breakout of previous day high and low.
    """
    k = 0.5
    historical_data['Range'] = historical_data['high'] - historical_data['low']
    range_value = historical_data['Range'].iloc[-2]  # Use previous day range

    historical_data['Upper'] = historical_data['close'].iloc[-2] + k * range_value
    historical_data['Lower'] = historical_data['close'].iloc[-2] - k * range_value

    # Buy signal
    if current_price > historical_data['Upper'].iloc[-1] and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal
    elif current_price < historical_data['Lower'].iloc[-1] and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Adaptive Momentum Strategy
def adaptive_momentum_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Adaptive Momentum Strategy using market volatility.
    """
    volatility = historical_data['close'].pct_change().std()
    adaptive_window = int(max(10, min(50, 20 / volatility)))
    
    historical_data['Momentum'] = historical_data['close'].pct_change(periods=adaptive_window)
    current_momentum = historical_data['Momentum'].iloc[-1]

    # Buy if momentum is significantly positive
    if current_momentum > 0.02 and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell if momentum is significantly negative
    elif current_momentum < -0.02 and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Hull Moving Average Strategy
def hull_moving_average_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Hull Moving Average Strategy to identify buy/sell signals.
    """
    short_window = 9
    long_window = 21

    hma_short = 2 * historical_data['close'].rolling(window=short_window).mean() - historical_data['close'].rolling(window=short_window // 2).mean()
    hma_long = 2 * historical_data['close'].rolling(window=long_window).mean() - historical_data['close'].rolling(window=long_window // 2).mean()

    if hma_short.iloc[-1] > hma_long.iloc[-1] and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    elif hma_short.iloc[-1] < hma_long.iloc[-1] and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

def triple_moving_average_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Triple Moving Average Strategy to determine buy, sell, or hold signals based on short, medium, and long-term moving averages.

    :param ticker: The stock ticker symbol.
    :param current_price: The current price of the stock.
    :param historical_data: Historical stock data for the ticker.
    :param account_cash: Available cash in the account.
    :param portfolio_qty: Quantity of stock held in the portfolio.
    :param total_portfolio_value: Total value of the portfolio.
    :return: Tuple (action, quantity, ticker).
    """
    # Define moving average windows
    short_window = 5
    medium_window = 15
    long_window = 50
    
    # Maximum percentage of portfolio to invest per trade
    max_investment_percentage = 0.10  # 10% of total portfolio value
    max_investment = total_portfolio_value * max_investment_percentage

    # Calculate the moving averages
    historical_data['SMA'] = historical_data['close'].rolling(window=short_window).mean()
    historical_data['MMA'] = historical_data['close'].rolling(window=medium_window).mean()
    historical_data['LMA'] = historical_data['close'].rolling(window=long_window).mean()

    # Drop NaN values after creating the moving averages
    historical_data.dropna(inplace=True)

    short_ma = historical_data['SMA'].iloc[-1]
    medium_ma = historical_data['MMA'].iloc[-1]
    long_ma = historical_data['LMA'].iloc[-1]

    # Buy signal: Short MA crosses above both Medium and Long MAs
    if short_ma > medium_ma and short_ma > long_ma and account_cash > 0:
        quantity_to_buy = min(int(max_investment // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal: Short MA crosses below both Medium and Long MAs
    elif short_ma < medium_ma and short_ma < long_ma and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    # No action triggered
    return ('hold', 0, ticker)

def volume_price_trend_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Volume Price Trend (VPT) Strategy for determining buy, sell, or hold signals.

    :param ticker: The stock ticker symbol.
    :param current_price: The current price of the stock.
    :param historical_data: Historical stock data for the ticker.
    :param account_cash: Available cash in the account.
    :param portfolio_qty: Quantity of stock held in the portfolio.
    :param total_portfolio_value: Total value of the portfolio.
    :return: Tuple (action, quantity, ticker).
    """
    # Calculate Volume Price Trend (VPT)
    historical_data['Price_Change'] = historical_data['close'].pct_change()
    historical_data['VPT'] = (historical_data['Price_Change'] * historical_data['volume']).cumsum()

    # Drop NaN values
    historical_data.dropna(inplace=True)

    # Define strategy thresholds
    vpt_current = historical_data['VPT'].iloc[-1]
    vpt_previous = historical_data['VPT'].iloc[-2]

    # Maximum percentage of portfolio to invest per trade
    max_investment_percentage = 0.10  # 10% of total portfolio value
    max_investment = total_portfolio_value * max_investment_percentage

    # Buy signal: VPT is increasing
    if vpt_current > vpt_previous and account_cash > 0:
        quantity_to_buy = min(int(max_investment // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal: VPT is decreasing
    elif vpt_current < vpt_previous and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    # No action triggered
    return ('hold', 0, ticker)

# Ultimate Oscillator Strategy
def ultimate_oscillator_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Ultimate Oscillator Strategy for buy/sell signals.
    """
    short_period = 7
    medium_period = 14
    long_period = 28

    # Calculate Buying Pressure (BP) and True Range (TR)
    historical_data['BP'] = historical_data['close'] - np.minimum(historical_data['low'], historical_data['close'].shift(1))
    historical_data['TR'] = np.maximum(historical_data['high'], historical_data['close'].shift(1)) - np.minimum(historical_data['low'], historical_data['close'].shift(1))

    # Calculate Average BP and TR for different periods
    historical_data['Avg_BP_Short'] = historical_data['BP'].rolling(window=short_period).mean()
    historical_data['Avg_TR_Short'] = historical_data['TR'].rolling(window=short_period).mean()

    historical_data['Avg_BP_Medium'] = historical_data['BP'].rolling(window=medium_period).mean()
    historical_data['Avg_TR_Medium'] = historical_data['TR'].rolling(window=medium_period).mean()

    historical_data['Avg_BP_Long'] = historical_data['BP'].rolling(window=long_period).mean()
    historical_data['Avg_TR_Long'] = historical_data['TR'].rolling(window=long_period).mean()

    # Calculate Ultimate Oscillator
    buying_pressure = (
        4 * (historical_data['Avg_BP_Short'] / historical_data['Avg_TR_Short']) +
        2 * (historical_data['Avg_BP_Medium'] / historical_data['Avg_TR_Medium']) +
        1 * (historical_data['Avg_BP_Long'] / historical_data['Avg_TR_Long'])
    )
    ultimate_oscillator = 100 * buying_pressure / (4 + 2 + 1)

    # Buy/Sell Logic
    current_uo = ultimate_oscillator.iloc[-1]

    # Buy signal
    if current_uo > 70 and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal
    elif current_uo < 30 and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Stochastic Momentum Strategy
def stochastic_momentum_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Stochastic Momentum Strategy for buy/sell signals.
    """
    lookback_period = 14

    historical_data['Lowest_Low'] = historical_data['low'].rolling(window=lookback_period).min()
    historical_data['Highest_High'] = historical_data['high'].rolling(window=lookback_period).max()

    historical_data['Stochastic'] = 100 * (historical_data['close'] - historical_data['Lowest_Low']) / (historical_data['Highest_High'] - historical_data['Lowest_Low'])

    stochastic_value = historical_data['Stochastic'].iloc[-1]

    # Buy signal
    if stochastic_value < 20 and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal
    elif stochastic_value > 80 and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Volume Weighted MACD Strategy
def volume_weighted_macd_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Volume Weighted Moving Average Convergence Divergence (MACD) Strategy.
    """
    short_window = 12
    long_window = 26
    signal_window = 9

    historical_data['VWAP'] = (historical_data['close'] * historical_data['volume']).cumsum() / historical_data['volume'].cumsum()
    historical_data['EWMA12'] = historical_data['VWAP'].ewm(span=short_window, adjust=False).mean()
    historical_data['EWMA26'] = historical_data['VWAP'].ewm(span=long_window, adjust=False).mean()
    historical_data['MACD'] = historical_data['EWMA12'] - historical_data['EWMA26']
    historical_data['Signal'] = historical_data['MACD'].ewm(span=signal_window, adjust=False).mean()

    # Buy/Sell Logic
    macd_value = historical_data['MACD'].iloc[-1]
    signal_value = historical_data['Signal'].iloc[-1]

    if macd_value > signal_value and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    elif macd_value < signal_value and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Volatility Breakout Strategy
def volatility_breakout_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Volatility Breakout Strategy for buy/sell signals.
    """
    breakout_multiplier = 1.5
    atr_window = 14

    historical_data['ATR'] = historical_data['close'].rolling(window=atr_window).apply(lambda x: max(x) - min(x))
    breakout_threshold = historical_data['ATR'].iloc[-1] * breakout_multiplier

    # Buy signal
    if current_price > (historical_data['close'].iloc[-2] + breakout_threshold) and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell signal
    elif current_price < (historical_data['close'].iloc[-2] - breakout_threshold) and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Momentum Divergence Strategy
def momentum_divergence_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Momentum Divergence Strategy for buy/sell signals.
    """
    short_window = 10
    long_window = 30

    historical_data['Momentum_Short'] = historical_data['close'].diff(periods=short_window)
    historical_data['Momentum_Long'] = historical_data['close'].diff(periods=long_window)

    if historical_data['Momentum_Short'].iloc[-1] > 0 > historical_data['Momentum_Long'].iloc[-1] and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    elif historical_data['Momentum_Short'].iloc[-1] < 0 < historical_data['Momentum_Long'].iloc[-1] and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)

# Wavelet Decomposition Strategy
def wavelet_decomposition_strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
    """
    Wavelet Decomposition Strategy for buy/sell signals.
    """
    wavelet = 'db1'

    coeffs = pywt.wavedec(historical_data['close'], wavelet, level=2)
    cA2, cD2, cD1 = coeffs

    reconstructed_signal = pywt.waverec([cA2, None, None], wavelet)

    # Buy if the current price is below the reconstructed signal
    if current_price < reconstructed_signal[-1] and account_cash > 0:
        quantity_to_buy = min(int((total_portfolio_value * 0.10) // current_price), int(account_cash // current_price))
        if quantity_to_buy > 0:
            return ('buy', quantity_to_buy, ticker)

    # Sell if the current price is above the reconstructed signal
    elif current_price > reconstructed_signal[-1] and portfolio_qty > 0:
        quantity_to_sell = min(portfolio_qty, max(1, int(portfolio_qty * 0.5)))
        return ('sell', quantity_to_sell, ticker)

    return ('hold', 0, ticker)