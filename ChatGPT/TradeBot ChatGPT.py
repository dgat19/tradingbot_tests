import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import DonchianChannel

# Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Perform moving average crossover analysis
def moving_average_crossover_analysis(stock_data, short_window, long_window):
    signals = pd.DataFrame(index=stock_data.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = SMAIndicator(stock_data['Close'], window=short_window).sma_indicator()

    # Create long simple moving average
    signals['long_mavg'] = SMAIndicator(stock_data['Close'], window=long_window).sma_indicator()

    # Generate trading signals based on the crossover
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Perform RSI analysis
def rsi_analysis(stock_data, rsi_window, rsi_overbought, rsi_oversold):
    signals = pd.DataFrame(index=stock_data.index)
    signals['signal'] = 0.0

    # Calculate RSI
    rsi = RSIIndicator(stock_data['Close'], window=rsi_window).rsi()

    # Generate trading signals based on RSI levels
    signals['signal'][rsi < rsi_oversold] = 1.0
    signals['signal'][rsi > rsi_overbought] = -1.0

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Perform Donchian channel analysis
def donchian_channel_analysis(stock_data, dc_window):
    signals = pd.DataFrame(index=stock_data.index)
    signals['signal'] = 0.0

    # Calculate Donchian channels
    donchian = DonchianChannel(stock_data['High'], stock_data['Low'], stock_data['Close'], window=dc_window)
    signals['upper_band'] = donchian.donchian_channel_hband()
    signals['lower_band'] = donchian.donchian_channel_lband()

    # Generate trading signals based on price crossing Donchian channels
    signals['signal'][stock_data['Close'] > signals['upper_band']] = -1.0
    signals['signal'][stock_data['Close'] < signals['lower_band']] = 1.0

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

# Plotting the stock prices and trading signals
def plot_stock_data(stock_data, signals):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(stock_data.index, stock_data['Close'], label='Stock Price')

    # Plot additional indicators if they exist
    if 'short_mavg' in signals.columns:
        ax.plot(stock_data.index, signals['short_mavg'], label='Short MA', linestyle='--')

    if 'long_mavg' in signals.columns:
        ax.plot(stock_data.index, signals['long_mavg'], label='Long MA', linestyle='--')

    ax2 = ax.twinx()
    
    if 'rsi' in signals.columns:
        ax2.plot(stock_data.index, signals['rsi'], label='RSI', color='orange', alpha=0.5)
        ax2.axhline(70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(30, color='g', linestyle='--', alpha=0.5, label='Oversold')

    ax3 = ax.twinx()
    
    if 'upper_band' in signals.columns and 'lower_band' in signals.columns:
        ax3.plot(stock_data.index, signals['upper_band'], label='Upper Band', linestyle='-.', color='purple', alpha=0.5)
        ax3.plot(stock_data.index, signals['lower_band'], label='Lower Band', linestyle='-.', color='brown', alpha=0.5)

    # Plot buy signals
    if 'positions' in signals.columns:
        ax.plot(signals.loc[signals.positions == 1.0].index,
                stock_data['Close'][signals.positions == 1.0],
                '^', markersize=10, color='g', label='Buy Signal')

    # Plot sell signals
    if 'positions' in signals.columns:
        ax.plot(signals.loc[signals.positions == -1.0].index,
                stock_data['Close'][signals.positions == -1.0],
                'v', markersize=10, color='r', label='Sell Signal')

    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    plt.show()

# Example usage
ticker = 'AAPL'
start_date = '2024-01-01'
end_date = '2024-03-07'
short_window = 40
long_window = 100
rsi_window = 14
rsi_overbought = 70
rsi_oversold = 30
dc_window = 20

stock_data = download_stock_data(ticker, start_date, end_date)

# Moving Average Crossover Analysis
signals_ma = moving_average_crossover_analysis(stock_data, short_window, long_window)

# RSI Analysis
signals_rsi = rsi_analysis(stock_data, rsi_window, rsi_overbought, rsi_oversold)

# Donchian Channel Analysis
signals_dc = donchian_channel_analysis(stock_data, dc_window)

# Combine signals (you might want to customize this part based on your strategy)
signals_combined = signals_ma[['signal', 'positions']].add(signals_rsi[['signal', 'positions']], fill_value=0).add(signals_dc[['signal', 'positions']], fill_value=0)

plot_stock_data(stock_data, signals_combined)