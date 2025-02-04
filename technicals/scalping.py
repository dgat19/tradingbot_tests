import sys
import os
os.environ["QT_API"] = "PySide6"
os.environ["APCA_API_DATA_URL"] = "https://data.alpaca.markets"

import time
import threading
import datetime
import pytz
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6 import QtCore, QtWidgets

# ================================
# Alpaca API Initialization
# ================================
ALPACA_API_KEY = 'PK2UB2P604D2CDEJDPE5'
ALPACA_SECRET_KEY = 'VYW9CmTKIepPEscKX43OR1OsQShWcAJb4taYAoC6'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL, api_version='v2')

# ================================
# Global Trades List
# ================================
trades = []  # Each trade is a dict: {ticker, side, time, price, (quantity)}

# ================================
# Account Info
# ================================
def display_account_info():
    try:
        account = api.get_account()
        cash = float(account.cash)
        print(f"\nAccount Cash: ${cash:.2f}")
        return cash
    except Exception as e:
        print(f"Error fetching account info: {e}")
        return 0.0

# ================================
# Market Hours Check (for order type selection)
# ================================
def is_market_open_local():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    return market_open <= now.time() <= market_close

# ================================
# Data Retrieval Functions
# ================================
def get_historical_data(ticker, timeframe='1Min', limit=1000):
    """
    Modified to handle market hours correctly
    """
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        
        # Set market open time for today
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # If before market open, use previous day
        if now.time() < datetime.time(9, 30):
            market_open = market_open - datetime.timedelta(days=1)
            market_close = market_close - datetime.timedelta(days=1)
        
        # If after market close, use today's market hours
        elif now.time() > datetime.time(16, 0):
            now = market_close
        
        # Get bars using IEX feed with market hours
        bars = api.get_bars(
            symbol=ticker,
            timeframe=timeframe,
            start=market_open.isoformat(),
            end=now.isoformat(),
            adjustment='raw',
            feed='iex'
        )
        bars = list(bars)
        
        if not bars:
            print(f"No {timeframe} bars returned for {ticker}")
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
        df.sort_index(inplace=True)
        return df
        
    except Exception as e:
        print(f"Error fetching {timeframe} data for {ticker}: {e}")
        return None

def get_daily_data(ticker, limit=2):
    try:
        bars = api.get_bars(ticker, '1Day', limit=limit, adjustment='raw', feed='iex')
        bars = list(bars)
    except Exception as e:
        print(f"Error fetching daily data for {ticker}: {e}")
        return None
    if not bars:
        print(f"No daily bars returned for {ticker}")
        return None
    data = {
        'time':   [bar.t for bar in bars],
        'open':   [bar.o for bar in bars],
        'high':   [bar.h for bar in bars],
        'low':    [bar.l for bar in bars],
        'close':  [bar.c for bar in bars],
        'volume': [bar.v for bar in bars]
    }
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df

# ================================
# Wilder Smoothing Function
# ================================
def wilder_smoothing(series, period):
    result = series.copy()
    initial_avg = result.iloc[:period].mean()
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
    prev_low  = df['low'].shift(1)
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
    dm_plus_series  = pd.Series(dm_plus, index=df.index)
    dm_minus_series = pd.Series(dm_minus, index=df.index)
    smoothed_tr      = wilder_smoothing(true_range, len_period)
    smoothed_dm_plus = wilder_smoothing(dm_plus_series, len_period)
    smoothed_dm_minus= wilder_smoothing(dm_minus_series, len_period)
    df['DI+'] = (smoothed_dm_plus / smoothed_tr) * 100
    df['DI-'] = (smoothed_dm_minus / smoothed_tr) * 100
    df['DX'] = (abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])) * 100
    df['DX'] = df['DX'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ================================
# Display Ticker Info
# ================================
def display_ticker_info(ticker, intraday_df):
    daily_df = get_daily_data(ticker, limit=2)
    if daily_df is not None and len(daily_df) >= 2:
        yesterday_close = daily_df['close'].iloc[-2]
        today_open = daily_df['open'].iloc[-1]
    else:
        yesterday_close = float('nan')
        today_open = float('nan')
    if intraday_df is None or intraday_df.empty:
        print(f"\nTicker: {ticker}")
        print(f"  Yesterday's Close: {yesterday_close:.2f}")
        print(f"  Today's Open: {today_open:.2f}")
        print("  No intraday data available.")
        return
    current_price = intraday_df['close'].iloc[-1]
    current_di_plus = intraday_df['DI+'].iloc[-1]
    current_di_minus = intraday_df['DI-'].iloc[-1]
    print(f"\nTicker: {ticker}")
    print(f"  Yesterday's Close: {yesterday_close:.2f}")
    print(f"  Today's Open: {today_open:.2f}")
    print(f"  Current Price (Last Bar): {current_price:.2f}")
    print(f"  DI+: {current_di_plus:.2f}")
    print(f"  DI-: {current_di_minus:.2f}")

# ================================
# Signal Logic
# ================================
def check_signals(df, ticker):
    """
    Modified signal logic to prevent unfavorable trades and hold through drawdowns
    """
    if df is None or len(df) < 2:
        return None
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Calculate price movement
    price_change = (curr['close'] - prev['close']) / prev['close'] * 100
    current_price = curr['close']
    
    # Buy conditions with price confirmation
    if (prev['DI+'] < prev['DI-']) and (curr['DI+'] > curr['DI-']):
        # Only buy if price isn't moving down significantly
        if price_change >= -0.5:  # Less than 0.5% down
            return 'buy'
            
    # Get current position info
    position_qty = get_position_qty(ticker)
    if position_qty > 0:
        entry_price = get_position_entry_price(ticker)
        if entry_price:
            # Calculate current P/L
            current_pl = (current_price - entry_price) / entry_price * 100
            
            # Sell conditions with position protection
            if (prev['DI+'] > prev['DI-']) and (curr['DI+'] < curr['DI-']):
                # Only sell if we're in profit or DI- is very strong
                if current_pl > 0 or curr['DI-'] >= 60:
                    return 'sell'
                else:
                    print(f"{ticker}: Holding through drawdown. Current P/L: {current_pl:.2f}%")
                    return None
            
            # Take profit condition
            elif (curr['DI+'] >= 60) and price_change >= 1.0:
                return 'sell'
            
            # Additional protection: sell if loss exceeds threshold
            elif current_pl <= -2.0:  # 2% max loss
                print(f"{ticker}: Stop loss triggered. Current P/L: {current_pl:.2f}%")
                return 'sell'
    else:
        # If no position, use normal sell signals
        if (prev['DI+'] > prev['DI-']) and (curr['DI+'] < curr['DI-']):
            return 'sell'
        elif (curr['DI+'] >= 60) and price_change >= 1.0:
            return 'sell'
    
    return None

# ================================
# Position & Order Functions
# ================================
def get_position_qty(ticker):
    try:
        position = api.get_position(ticker)
        qty = float(position.qty)
        return qty
    except Exception as e:
        return 0.0

def get_position_entry_price(ticker):
    """
    Get the entry price for current position
    """
    try:
        position = api.get_position(ticker)
        entry_price = float(position.avg_entry_price)
        return entry_price
    except Exception as e:
        return None

def get_confidence_from_di(di_plus: float, di_minus: float) -> float:
    if di_plus <= di_minus:
        return 0.0
    diff = di_plus - di_minus
    total = di_plus + di_minus
    if total <= 0:
        return 0.0
    raw_ratio = diff / total
    confidence = max(0.0, min(1.0, raw_ratio))
    return confidence

def get_filled_price(order_id):
    """
    Get the actual filled price for an order from Alpaca
    """
    try:
        # Wait briefly for the order to fill
        time.sleep(2)
        order = api.get_order(order_id)
        
        # Wait up to 10 seconds for the order to fill
        wait_count = 0
        while order.status == 'new' or order.status == 'accepted' or order.status == 'pending_new':
            time.sleep(1)
            order = api.get_order(order_id)
            wait_count += 1
            if wait_count >= 10:
                break
        
        if order.status == 'filled':
            return float(order.filled_avg_price)
        else:
            print(f"Order {order_id} not filled after waiting. Status: {order.status}")
            return None
    except Exception as e:
        print(f"Error getting fill price: {e}")
        return None

def place_bracket_order(ticker, side, qty, entry_price, time_index):
    """
    Modified to use actual fill price instead of estimated entry price
    """
    global trades
    try:
        q = float(qty)
    except:
        q = 0.0
    if q != int(q):
        print(f"{ticker}: Fractional order detected ({qty}), using simple order instead of bracket order.")
        place_simple_order(ticker, side, qty, entry_price, time_index)
        return

    if side == 'buy':
        take_profit_price = round(entry_price * 2, 2)
        stop_loss_price   = round(entry_price * 0.8, 2)
    else:
        take_profit_price = round(entry_price * 1.0, 2)
        stop_loss_price   = round(entry_price * 1.2, 2)
        
    try:
        if is_market_open_local():
            order = api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day',
                order_class='bracket',
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_loss_price}
            )
            # Get actual fill price
            filled_price = get_filled_price(order.id)
            if filled_price:
                print(f"Placed {side} BRACKET order for {ticker} | Qty: {qty} | Filled at: {filled_price:.2f} "
                      f"| TP: {take_profit_price} | SL: {stop_loss_price}")
                trades.append({
                    'ticker': ticker,
                    'side': side,
                    'time': time_index,
                    'price': filled_price  # Use actual fill price
                })
            else:
                print(f"Warning: Could not get fill price for {ticker} order")
        else:
            order = api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day',
                extended_hours=True
            )
            filled_price = get_filled_price(order.id)
            if filled_price:
                print(f"Placed {side} EXTENDED-HOURS order for {ticker} | Qty: {qty} | Filled at: {filled_price:.2f}")
                trades.append({
                    'ticker': ticker,
                    'side': side,
                    'time': time_index,
                    'price': filled_price  # Use actual fill price
                })
            else:
                print(f"Warning: Could not get fill price for {ticker} order")
    except Exception as e:
        print(f"Order error for {ticker}: {e}")

def place_simple_order(ticker, side, qty, entry_price, time_index):
    """
    Modified to use actual fill price instead of estimated entry price
    """
    global trades
    try:
        if is_market_open_local():
            order = api.submit_order(
                symbol=ticker,
                qty=str(qty),
                side=side,
                type='market',
                time_in_force='day'
            )
            filled_price = get_filled_price(order.id)
            if filled_price:
                print(f"Placed SIMPLE {side} order for {ticker} (REGULAR hours) | Qty: {qty} | Filled at: {filled_price:.2f}")
                trades.append({
                    'ticker': ticker,
                    'side': side,
                    'time': time_index,
                    'price': filled_price  # Use actual fill price
                })
            else:
                print(f"Warning: Could not get fill price for {ticker} order")
        else:
            order = api.submit_order(
                symbol=ticker,
                qty=str(qty),
                side=side,
                type='market',
                time_in_force='day',
                extended_hours=True
            )
            filled_price = get_filled_price(order.id)
            if filled_price:
                print(f"Placed SIMPLE {side} order for {ticker} (EXTENDED hours) | Qty: {qty} | Filled at: {filled_price:.2f}")
                trades.append({
                    'ticker': ticker,
                    'side': side,
                    'time': time_index,
                    'price': filled_price  # Use actual fill price
                })
            else:
                print(f"Warning: Could not get fill price for {ticker} order")
    except Exception as e:
        print(f"Simple order error for {ticker}: {e}")

def place_market_sell(ticker, sell_qty, sell_price, time_index):
    """
    Modified to use actual fill price instead of estimated entry price
    """
    global trades
    try:
        current_qty = get_position_qty(ticker)
        if current_qty <= 0:
            print(f"{ticker}: No position available to sell.")
            return
        sell_qty = min(abs(sell_qty), current_qty)
        if sell_qty <= 0:
            print(f"{ticker}: Invalid sell quantity {sell_qty}")
            return
        sell_qty_str = str(round(sell_qty, 3))
        
        if is_market_open_local():
            order = api.submit_order(
                symbol=ticker,
                qty=sell_qty_str,
                side='sell',
                type='market',
                time_in_force='day'
            )
            filled_price = get_filled_price(order.id)
            if filled_price:
                print(f"{ticker}: Market sell filled (REGULAR hours) to close {sell_qty_str} shares at {filled_price:.2f}")
                trades.append({
                    'ticker': ticker,
                    'side': 'sell',
                    'time': time_index,
                    'price': filled_price,  # Use actual fill price
                    'quantity': sell_qty_str
                })
            else:
                print(f"Warning: Could not get fill price for {ticker} sell order")
        else:
            order = api.submit_order(
                symbol=ticker,
                qty=sell_qty_str,
                side='sell',
                type='market',
                time_in_force='day',
                extended_hours=True
            )
            filled_price = get_filled_price(order.id)
            if filled_price:
                print(f"{ticker}: Market sell filled (EXTENDED hours) to close {sell_qty_str} shares at {filled_price:.2f}")
                trades.append({
                    'ticker': ticker,
                    'side': 'sell',
                    'time': time_index,
                    'price': filled_price,  # Use actual fill price
                    'quantity': sell_qty_str
                })
            else:
                print(f"Warning: Could not get fill price for {ticker} sell order")
    except Exception as e:
        print(f"Error closing position for {ticker}: {e}")

# ================================
# Trading Logic
# ================================
def run_trading():
    """
    Use confidence-based sizing: if confidence=1, buy up to 10% of the 50% allocation; 
    if 0.4, buy 4% of the 50% allocation.
    Now, orders are fractional.
    """
    account_cash = display_account_info()
    tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'TSLA', 'PLTR', 'GOOG']
    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        try:
            intraday_df = get_historical_data(ticker, timeframe='1Min', limit=250)
            if intraday_df is None or intraday_df.empty:
                print("No intraday data.")
                continue
            intraday_df = calculate_indicators(intraday_df, len_period=14)
            display_ticker_info(ticker, intraday_df)
            print(f"  Historical DI+ for {ticker}: {intraday_df['DI+'].tail(5).tolist()}")
            print(f"  Historical DI- for {ticker}: {intraday_df['DI-'].tail(5).tolist()}")
            
            # Only change: pass ticker to check_signals
            signal = check_signals(intraday_df, ticker)
            
            if not signal:
                print(f"{ticker}: No signal generated.")
                continue
            last_bar = intraday_df.iloc[-1]
            di_plus = last_bar['DI+']
            di_minus = last_bar['DI-']
            confidence = get_confidence_from_di(di_plus, di_minus)
            print(f"{ticker} Confidence from DI: {confidence:.2f}")
            current_price = last_bar['close']
            current_position = get_position_qty(ticker)
            if signal == 'buy':
                if current_position > 0:
                    print(f"{ticker}: Already in position.")
                    continue
                max_allocation = account_cash * 0.5  # 50% of account cash
                fraction_of_allocation = 0.1 * confidence  # Scale from 0% to 10% of that allocation
                buy_dollars = max_allocation * fraction_of_allocation
                qty = buy_dollars / current_price  # Fractional shares allowed
                qty = round(qty, 3)
                if qty < 0.001:
                    print(f"{ticker}: Confidence is {confidence:.2f}, but not enough funds to buy a fractional share.")
                    continue
                # If qty is fractional (not an integer), we use a simple order
                if qty != int(qty):
                    place_simple_order(ticker, 'buy', qty, current_price, intraday_df.index[-1])
                else:
                    place_bracket_order(ticker, 'buy', qty, current_price, intraday_df.index[-1])
            elif signal == 'sell':
                if current_position <= 0:
                    print(f"{ticker}: No position to sell.")
                    continue
                place_market_sell(ticker, current_position, current_price, intraday_df.index[-1])
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        time.sleep(10)

# ================================
# Plotting GUI with Tabs and Auto-Refresh
# ================================
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Two subplots: ax1 for price, ax2 for DI lines
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

def plot_di_for_ticker(ticker, canvas):
    """
    Modified to show correct market hours
    """
    df = get_historical_data(ticker, timeframe='1Min')
    if df is None or df.empty:
        canvas.ax1.clear()
        canvas.ax2.clear()
        canvas.ax1.text(0.5, 0.5, f"No data available for {ticker}", transform=canvas.ax1.transAxes,
                        ha="center", va="center", color="black")
        canvas.draw()
        return
        
    df = calculate_indicators(df, len_period=14)
    
    # Top subplot: Price with y-axis label
    canvas.ax1.clear()
    canvas.ax1.plot(df.index, df['close'], color="black", label="Close Price")
    
    # Add current time and price annotation
    current_price = df['close'].iloc[-1]
    last_update = df.index[-1].strftime('%H:%M')
    canvas.ax1.text(0.17, 0.7, 
                   f"Last Update: {last_update}\nPrice: {current_price:.2f}",
                   transform=canvas.ax1.transAxes,
                   ha="right", va="center",
                   color="blue", fontsize=10,
                   bbox=dict(facecolor="white", alpha=0.7))
    
    canvas.ax1.set_ylabel("Price", color="black")
    canvas.ax1.legend(loc="upper left")
    
    # Bottom subplot: DI lines with y-axis label
    canvas.ax2.clear()
    canvas.ax2.plot(df.index, df["DI+"], color="green", label="DI+")
    canvas.ax2.plot(df.index, df["DI-"], color="red", label="DI-")
    canvas.ax2.set_ylabel("DI (%)", color="black")
    canvas.ax2.legend(loc="upper left")
    
    # Format x-axis to show market hours correctly
    for ax in [canvas.ax1, canvas.ax2]:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        # Set major ticks for every hour
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(9, 17)))
        # Add minor ticks every 15 minutes
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax.tick_params(rotation=45)
        
        # Set x-axis limits to market hours
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.time() > datetime.time(16, 0):
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            market_close = now
            
        ax.set_xlim(market_open, market_close)
    
    # Overlay buy/sell markers
    for trade_info in trades:
        if trade_info['ticker'] == ticker:
            trade_time = trade_info['time']
            try:
                diffs = np.abs((df.index - trade_time).total_seconds())
                idx = diffs.argmin()
                closest_time = df.index[idx]
            except Exception as e:
                closest_time = trade_time
                
            if closest_time in df.index:
                # Get price value for the trade
                price_val = df.loc[closest_time, "close"]
                
                # Configure marker properties based on trade side
                if trade_info['side'] == 'buy':
                    marker = '^'
                    color = 'green'
                    offset = 2
                    annotation = f"BUY\n${price_val:.2f}"
                else:
                    marker = 'v'
                    color = 'red'
                    offset = -2
                    annotation = f"SELL\n${price_val:.2f}"
                
                # Plot marker and annotation on price chart
                canvas.ax1.scatter(closest_time, price_val, 
                                 color=color, 
                                 marker=marker, 
                                 s=100,
                                 zorder=5)
                
                # Add price annotation with background
                canvas.ax1.annotate(annotation,
                                  xy=(closest_time, price_val),
                                  xytext=(8, offset * 8),
                                  textcoords='offset points',
                                  ha='left',
                                  va='center',
                                  color=color,
                                  weight='bold',
                                  bbox=dict(
                                      facecolor='white',
                                      edgecolor=color,
                                      alpha=0.8,
                                      pad=1,
                                      boxstyle='round,pad=0.5'
                                  ),
                                  arrowprops=dict(
                                      arrowstyle='->',
                                      color=color,
                                      connectionstyle='arc3,rad=0.2'
                                  ))
                
                # Plot marker on DI subplot
                di_val = df.loc[closest_time, "DI+"] if trade_info['side'] == 'buy' else df.loc[closest_time, "DI-"]
                canvas.ax2.scatter(closest_time, di_val, 
                                 color=color, 
                                 marker=marker, 
                                 s=80, 
                                 zorder=5)
    
    # Set theme and layout
    for ax in [canvas.ax1, canvas.ax2]:
        ax.set_facecolor("#eeeeee")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            
    canvas.ax1.set_title(f"{ticker} Intraday Price & DI", color="black")
    canvas.fig.tight_layout()
    canvas.draw()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Intraday Price & DI Plot - Real-time Updates")
        self.setGeometry(200, 100, 1000, 600)
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self.tab_widget.setDocumentMode(True)
        self.setStyleSheet("""
            QTabWidget::pane { background: #eeeeee; }
            QTabBar::tab { background: #cccccc; color: black; padding: 10px; }
            QTabBar::tab:selected { background: #bbbbbb; color: black; }
            QWidget { background-color: #eeeeee; color: black; }
        """)
        self.canvases = {}
        self.create_tabs()
        self.setCentralWidget(self.tab_widget)
        
        # Update plots more frequently (every 15 seconds)
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.refresh_all_plots)
        self.plot_timer.start(15000)  # 15 seconds instead of 60

    def create_tabs(self):
        tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'TSLA', 'PLTR', 'GOOG', 'SPY']
        self.tab_widget.clear()
        for ticker in tickers:
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()
            canvas = MplCanvas(self, width=8, height=5, dpi=100)
            layout.addWidget(canvas)
            widget.setLayout(layout)
            self.tab_widget.addTab(widget, ticker)
            self.canvases[ticker] = canvas
            plot_di_for_ticker(ticker, canvas)

    def refresh_all_plots(self):
        """
        Modified to handle refresh more efficiently
        """
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            current_ticker = self.tab_widget.tabText(self.tab_widget.currentIndex())
            # Update current tab immediately
            if current_ticker in self.canvases:
                plot_di_for_ticker(current_ticker, self.canvases[current_ticker])
        
        # Update other tabs in the background
        for ticker, canvas in self.canvases.items():
            if ticker != current_ticker:
                plot_di_for_ticker(ticker, canvas)

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
# Main Execution
# ================================
if __name__ == '__main__':
    trading_thread = threading.Thread(target=trading_loop, daemon=True)
    trading_thread.start()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
