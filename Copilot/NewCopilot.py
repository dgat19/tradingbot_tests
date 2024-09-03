import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

# Use yfinance while pandas_datareader is deprecated
yf.pdr_override()

# Input stock symbol
stock = input("Enter a stock symbol: ").upper()

# Get the end date as today's date
end_date = datetime.now()

# Get the start date as 6 months ago
start_date = end_date - timedelta(days=6*30)

# Get the historical data
data = pdr.get_data_yahoo(stock, start=start_date, end=end_date)

# Calculate Simple Moving Average (SMA)
data['SMA'] = data['Close'].rolling(window=20).mean()

# Calculate MACD
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
data['MACD'] = macd
data['Signal Line'] = signal

# Create a function to signal when to buy and sell an asset
def buy_sell(signal):
    buy = []
    sell = []
    flag = -1

    for i in range(len(signal)):
        if signal['MACD'][i] < signal['Signal Line'][i]:
            sell.append(np.nan)
            if flag != 1:
                buy.append(signal['Close'][i])
                flag = 1
            else:
                buy.append(np.nan)
        elif signal['MACD'][i] > signal['Signal Line'][i]:
            buy.append(np.nan)
            if flag != 0:
                sell.append(signal['Close'][i])
                flag = 0
            else:
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return (buy, sell)

# Create buy and sell column
a = buy_sell(data)
data['Buy_Signal_Price'] = a[0]
data['Sell_Signal_Price'] = a[1]

# Show the data
print(data)

# Plot the stock close price, SMA, Buy-Signal and Sell-Signal
plt.figure(figsize=(12.5, 4.5))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA'], label='Simple Moving Average', color='orange')
plt.scatter(data.index, data['Buy_Signal_Price'], color='green', label='Buy Signal', marker='^', alpha=1)
plt.scatter(data.index, data['Sell_Signal_Price'], color='red', label='Sell Signal', marker='v', alpha=1)
plt.title('Stock Close Price History & Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend(loc='upper left')
plt.show()

# Plot MACD and Signal line
plt.figure(figsize=(12.5, 4.5))
plt.plot(data.index, macd, label='MACD', color = 'red')
plt.plot(data.index, signal, label='Signal Line', color='blue')
plt.legend(loc='upper left')
plt.show()