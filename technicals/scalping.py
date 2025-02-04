import time
import threading
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt

# ================================
# Alpaca API Initialization
# ================================
ALPACA_API_KEY = 'PKWVCXLEZXPLXWFI8XZE'
ALPACA_SECRET_KEY = 'sgDQb0feimO2ZAveHlLn8upqxkejUzwXaH5jzasg'
BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# ================================
# Global Trades List
# ================================
trades = []

# ================================
# Account Info
# ================================
def display_account_info():
    """
    Retrieve and display the account's cash.
    Returns the cash as a float.
    """
    try:
        account = api.get_account()
        cash = float(account.cash)
        print(f"\nAccount Cash: ${cash:.2f}")
        return cash
    except Exception as e:
        print(f"Error fetching account info: {e}")
        return 0.0

# ================================
# Data Retrieval
# ================================
def get_historical_data(ticker, timeframe='1Min', limit=1000):
    """
    Retrieve historical bar data for the given ticker, including extended hours if available.
    """
    try:
        # Attempt to include extended hours by setting feed='iex' or feed='sip'
        # and adjustment='raw'. 
        bars = api.get_bars(
            symbol=ticker,
            timeframe=timeframe,
            limit=limit,
            adjustment='raw',    # show raw trades (splits/dividends not adjusted)
            feed='iex'           # or 'sip' if your subscription supports it
        )
        bars = list(bars)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    if not bars:
        print(f"No bars returned for {ticker}")
        return None

    data = {
        'time': [bar.t for bar in bars],
        'open': [bar.o for bar in bars],
        'high': [bar.h for bar in bars],
        'low': [bar.l for bar in bars],
        'close': [bar.c for bar in bars],
        'volume': [bar.v for bar in bars]
    }
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    return df


# ================================
# Wilder Smoothing Function
# ================================
def wilder_smoothing(series, period):
    result = series.copy()
    initial_avg = series.iloc[:period].mean()
    result.iloc[:period] = initial_avg
    for i in range(period, len(series)):
        result.iloc[i] = result.iloc[i-1] - (result.iloc[i-1] / period) + series.iloc[i]
    return result

# ================================
# ADX/DI Indicator Calculation
# ================================
def calculate_indicators(df, len_period=14):
    prev_close = df['close'].shift(1)
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    dm_plus = np.where(
        (df['high'] - prev_high) > (prev_low - df['low']),
        np.maximum(df['high'] - prev_high, 0),
        0
    )
    dm_minus = np.where(
        (prev_low - df['low']) > (df['high'] - prev_high),
        np.maximum(prev_low - df['low'], 0),
        0
    )
    
    smoothed_tr = wilder_smoothing(true_range, len_period)
    smoothed_dm_plus = wilder_smoothing(pd.Series(dm_plus, index=df.index), len_period)
    smoothed_dm_minus = wilder_smoothing(pd.Series(dm_minus, index=df.index), len_period)
    
    df['DI+'] = (smoothed_dm_plus / smoothed_tr) * 100
    df['DI-'] = (smoothed_dm_minus / smoothed_tr) * 100
    
    df['DX'] = (abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])) * 100
    df['DX'] = df['DX'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# ================================
# Display Ticker Info
# ================================
def display_ticker_info(ticker, df):
    if df is None or df.empty:
        print(f"No data available for {ticker} to display info.")
        return
    
    current_open = df['open'].iloc[-1]
    if len(df) > 1:
        previous_close = df['close'].iloc[-2]
    else:
        previous_close = current_open
    
    current_price = df['close'].iloc[-1]
    current_di_plus = df['DI+'].iloc[-1]
    current_di_minus = df['DI-'].iloc[-1]
    
    print(f"\nTicker: {ticker}")
    print(f"  Last Bar Open: {current_open:.2f}")
    print(f"  Previous Bar Close: {previous_close:.2f}")
    print(f"  Current Price: {current_price:.2f}")
    print(f"  DI+: {current_di_plus:.2f}")
    print(f"  DI-: {current_di_minus:.2f}")

# ================================
# Signal Logic
# ================================
def check_signals(df):
    if df is None or len(df) < 2:
        return None
    
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    if (prev['DI+'] < prev['DI-']) and (curr['DI+'] > curr['DI-']):
        return 'buy'
    elif (prev['DI+'] > prev['DI-']) and (curr['DI+'] < curr['DI-']):
        return 'sell'
    return None

# ================================
# Position & Order
# ================================
def get_position_qty(ticker):
    try:
        position = api.get_position(ticker)
        return float(position.qty)
    except Exception:
        return 0.0

def place_bracket_order(ticker, side, qty, entry_price, time_index):
    global trades
    
    if side == 'buy':
        take_profit_price = round(entry_price * 2, 2)
        stop_loss_price = round(entry_price * 0.8, 2)
    else:
        take_profit_price = round(entry_price * 1, 2)
        stop_loss_price = round(entry_price * 1.2, 2)
    
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day',
            extended_hours=True,  # <--- allow pre/post market
            order_class='bracket',
            take_profit={'limit_price': take_profit_price},
            stop_loss={'stop_price': stop_loss_price}
        )
        print(f"Placed {side} order (extended hours) for {ticker} | Qty: {qty} | Entry: {entry_price:.2f} "
              f"| TP: {take_profit_price} | SL: {stop_loss_price}")

        trades.append({
            'ticker': ticker,
            'side': side,
            'time': time_index,
            'price': entry_price
        })
        
    except Exception as e:
        print(f"Order error for {ticker}: {e}")

def place_market_sell(ticker, sell_qty, sell_price, time_index):
    global trades
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=sell_qty,
            side='sell',
            type='market',
            time_in_force='day',
            extended_hours=True,  # <--- allow pre/post market
        )
        print(f"{ticker}: Market sell submitted (extended hours) to close {sell_qty} shares at ~{sell_price:.2f}")
        
        trades.append({
            'ticker': ticker,
            'side': 'sell',
            'time': time_index,
            'price': sell_price
        })
    except Exception as e:
        print(f"Error closing position for {ticker}: {e}")

# ================================
# Trading Logic
# ================================
def run_trading():
    """
    Loop over each ticker, compute ADX/DI on 1-min data, display info,
    and execute trades if signals appear. No market-hour checks—assume 24/7 availability.
    """
    account_cash = display_account_info()
    tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'TSLA', 'PLTR', 'GOOG']
    
    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        try:
            intraday_df = get_historical_data(ticker, timeframe='1Min', limit=250)
            if intraday_df is None or intraday_df.empty:
                continue
            
            intraday_df = calculate_indicators(intraday_df, len_period=14)
            display_ticker_info(ticker, intraday_df)
            
            print(f"  Historical DI+ for {ticker}: {intraday_df['DI+'].tail(5).tolist()}")
            print(f"  Historical DI- for {ticker}: {intraday_df['DI-'].tail(5).tolist()}")
            
            signal = check_signals(intraday_df)
            if not signal:
                print(f"{ticker}: No signal generated.")
                continue
            
            current_price = intraday_df['close'].iloc[-1]
            current_position = get_position_qty(ticker)
            
            if signal == 'buy':
                if current_position > 0:
                    print(f"{ticker}: Already in position.")
                    continue
                
                max_allocation = account_cash * 0.5
                max_qty = int(max_allocation // current_price)
                if max_qty < 1:
                    print(f"{ticker}: Not enough buying power to purchase at least 1 share.")
                    continue
                
                place_bracket_order(
                    ticker, 'buy', max_qty, current_price, intraday_df.index[-1]
                )
            
            elif signal == 'sell':
                if current_position <= 0:
                    print(f"{ticker}: No position to sell.")
                    continue
                place_market_sell(
                    ticker, current_position, current_price, intraday_df.index[-1]
                )
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

        time.sleep(10)

# ================================
# Plotting
# ================================
def plot_intraday_changes(tickers, refresh_interval=60):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    while True:
        ax.clear()
        local_trades = list(trades)
        
        for ticker in tickers:
            df = get_historical_data(ticker, timeframe='5Min', limit=1000)
            if df is None or df.empty:
                continue
            
            open_price = df['open'].iloc[0]
            df['pct_change'] = (df['close'] - open_price) / open_price * 100
            ax.plot(df.index, df['pct_change'], label=ticker)
            
            # Overlay buy/sell markers
            trade_times = []
            trade_values = []
            trade_colors = []
            trade_markers = []
            
            for trade_info in local_trades:
                if trade_info['ticker'] == ticker:
                    trade_time = trade_info['time']
                    side = trade_info['side']
                    
                    if trade_time in df.index:
                        y_val = df.loc[trade_time, 'pct_change']
                        trade_times.append(trade_time)
                        trade_values.append(y_val)
                        
                        if side == 'buy':
                            trade_colors.append('green')
                            trade_markers.append('^')
                        else:
                            trade_colors.append('red')
                            trade_markers.append('v')
            
            for i in range(len(trade_times)):
                ax.scatter(
                    x=[trade_times[i]], y=[trade_values[i]],
                    color=trade_colors[i], marker=trade_markers[i],
                    s=80, zorder=5
                )
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Percentage Change (%)")
        ax.set_title("Intraday Percentage Change (with Buy/Sell Markers)")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        plt.draw()
        plt.pause(refresh_interval)

# ================================
# Trading Loop
# ================================
def trading_loop():
    while True:
        print("\n----- Starting a new trading cycle -----")
        run_trading()
        print("----- Trading cycle complete. Sleeping 60 seconds. -----\n")
        time.sleep(60)

# ================================
# Main
# ================================
if __name__ == '__main__':
    tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'TSLA', 'PLTR', 'GOOG']
    
    trading_thread = threading.Thread(target=trading_loop, daemon=True)
    trading_thread.start()
    
    plot_intraday_changes(tickers, refresh_interval=60)
