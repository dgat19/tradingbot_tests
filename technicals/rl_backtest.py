import sys
import os
from dotenv import load_dotenv
os.environ["QT_API"] = "PySide6"
from datetime import datetime, timedelta
import pytz
import pandas as pd
from polygon import BaseClient
import alpaca_trade_api as tradeapi
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

# Import functions from local libraries
from scalping import (
    calculate_indicators,
    check_signals,
    get_confidence_from_di,
)
from reinforcement import (
    EnhancedTradingAgent,
    EnhancedTradingEnvironment,
    check_signals_with_rl
)

class MarketDataHandler:
    def __init__(self, api_key):
        """Initialize with Polygon.io API key"""
        self.client = BaseClient(api_key)
        self.eastern_tz = pytz.timezone('US/Eastern')
    
    def get_historical_bars(self, ticker, date, timespan='minute', multiplier=1):
        """
        Get historical bar data from Polygon.io
        
        Parameters:
        - ticker: Stock symbol
        - date: Date to fetch data for
        - timespan: Time interval ('minute', 'hour', 'day', etc.)
        - multiplier: Number of timespans (e.g., 1 for 1-minute bars)
        """
        try:
            # Convert date to eastern timezone
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d').date()
            elif isinstance(date, datetime):
                date = date.date()
            
            # Set time range with padding for indicator calculation
            start_date = date - timedelta(days=1)  # Extra day for calculations
            end_date = date + timedelta(days=1)    # Extra day to ensure we get all data
            
            # Construct path for API request
            path = f'/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{start_date.strftime("%Y-%m-%d")}/{end_date.strftime("%Y-%m-%d")}'
            
            # Make the request
            response = self.client._get_response(path)
            
            # Check if response has results
            if not response or 'results' not in response:
                print(f"No data returned for {ticker}")
                return None
            
            bars = response['results']
            
            if not bars:
                print(f"No data returned for {ticker}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame([{
                'time': pd.Timestamp(bar['t'], unit='ms', tz=self.eastern_tz),
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'vwap': bar.get('vw')  # Using get() to handle potential missing vwap
            } for bar in bars])
            
            # Set index and sort
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter to market hours
            market_open = datetime.combine(date, datetime.time(9, 30))
            market_open = self.eastern_tz.localize(market_open)
            market_close = datetime.combine(date, datetime.time(16, 0))
            market_close = self.eastern_tz.localize(market_close)
            
            df = df[market_open:market_close]
            
            print(f"Retrieved {len(df)} bars for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
            
    def get_real_time_quote(self, ticker):
        """Get real-time quote data"""
        try:
            quote = self.client.get_last_trade(ticker)
            return {
                'price': quote.price,
                'size': quote.size,
                'timestamp': pd.Timestamp(quote.timestamp, unit='ns', tz=self.eastern_tz),
                'conditions': quote.conditions
            }
        except Exception as e:
            print(f"Error fetching real-time quote for {ticker}: {e}")
            return None
            
    def get_live_bars(self, ticker, limit=1000):
        """Get most recent intraday bars"""
        try:
            now = datetime.now(self.eastern_tz)
            today = now.strftime('%Y-%m-%d')
            
            # Construct path for API request
            path = f'/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{today}/{today}'
            
            # Make the request
            response = self.client._get_response(path)
            
            # Check if response has results
            if not response or 'results' not in response:
                print(f"No data returned for {ticker}")
                return None
            
            bars = response['results']
            
            if not bars:
                return None
                
            df = pd.DataFrame([{
                'time': pd.Timestamp(bar['t'], unit='ms', tz=self.eastern_tz),
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'vwap': bar.get('vw')  # Using get() to handle potential missing vwap
            } for bar in bars])
            
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching live bars for {ticker}: {e}")
            return None
            
    def get_previous_close(self, ticker):
        """Get previous day's closing price"""
        try:
            yesterday = (datetime.now(self.eastern_tz).date() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Construct path for API request
            path = f'/v2/aggs/ticker/{ticker.upper()}/range/1/day/{yesterday}/{yesterday}'
            
            # Make the request
            response = self.client._get_response(path)
            
            # Check if response has results
            if not response or 'results' not in response:
                print(f"No data returned for {ticker}")
                return None
            
            bars = response['results']
            
            if bars:
                return bars[0]['c']
            return None
            
        except Exception as e:
            print(f"Error fetching previous close for {ticker}: {e}")
            return None

def initialize_data_handler():
    """Initialize the market data handler with configuration"""
    load_dotenv()
    api_key = os.getenv('POLYGON_API_KEY')
    return MarketDataHandler(api_key)

class DatePickerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout()
        
        self.date_edit = QtWidgets.QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QtCore.QDate.currentDate())
        self.date_edit.setMaximumDate(QtCore.QDate.currentDate())
        
        self.today_button = QtWidgets.QPushButton("Today")
        self.today_button.clicked.connect(self.set_today)
        
        layout.addWidget(QtWidgets.QLabel("Select Date:"))
        layout.addWidget(self.date_edit)
        layout.addWidget(self.today_button)
        self.setLayout(layout)
        
    def set_today(self):
        self.date_edit.setDate(QtCore.QDate.currentDate())
        
    def get_selected_date(self):
        return self.date_edit.date().toPython()

class BacktestCanvas(QtWidgets.QWidget):
    """Canvas widget for backtest visualization"""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        super().__init__(parent)
        
        # Create the figure and canvas
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(311)  # Price chart
        self.ax2 = self.fig.add_subplot(312, sharex=self.ax1)  # DI lines
        self.ax3 = self.fig.add_subplot(313, sharex=self.ax1)  # RL confidence
        
        # Create layout and add canvas
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.fig.tight_layout()

    def draw(self):
        """Forward draw call to canvas"""
        self.canvas.draw()

    def clear(self):
        """Clear all axes"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

class BacktestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Strategy Backtesting")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)
        
        # Add controls layout
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Ticker selection
        self.ticker_combo = QtWidgets.QComboBox()
        self.ticker_combo.addItems(['NVDA', 'AAPL', 'MSFT', 'META', 'TSLA', 'PLTR', 'GOOG'])
        controls_layout.addWidget(QtWidgets.QLabel("Symbol:"))
        controls_layout.addWidget(self.ticker_combo)
        
        # Date picker
        self.date_picker = DatePickerWidget()
        controls_layout.addWidget(self.date_picker)
        
        # Strategy selection
        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems(['RL Strategy', 'Base DI Strategy', 'Both'])
        controls_layout.addWidget(QtWidgets.QLabel("Strategy:"))
        controls_layout.addWidget(self.strategy_combo)
        
        # Run backtest button
        self.run_button = QtWidgets.QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        controls_layout.addWidget(self.run_button)
        
        layout.addLayout(controls_layout)
        
        # Create and add canvas
        self.canvas = BacktestCanvas(self, width=10, height=6)
        layout.addWidget(self.canvas)
        
        # Add metrics text
        self.metrics_text = QtWidgets.QTextEdit()
        self.metrics_text.setMaximumHeight(100)
        self.metrics_text.setReadOnly(True)
        layout.addWidget(self.metrics_text)
        
        # Initialize components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize trading components and API"""
        self.trading_env = EnhancedTradingEnvironment()
        self.trading_agent = EnhancedTradingAgent()
        
        try:
            self.trading_agent.load('trading_qtable.pkl')
            print("Loaded existing Q-table")
        except:
            print("Starting with new Q-table")
        
        load_dotenv()
        self.api = tradeapi.REST(
            key_id=os.getenv('APCA_PAPER_API_KEY'),
            secret_key=os.getenv('APCA_PAPER_SECRET_KEY'),
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )

    def get_historical_data(self, ticker, date):
        """Get historical data using Polygon.io"""
        if not hasattr(self, 'data_handler'):
            self.data_handler = initialize_data_handler()
            
        df = self.data_handler.get_historical_bars(ticker, date)
        if df is not None:
            df = calculate_indicators(df)  # Calculate technical indicators
        return df

    def simulate_base_strategy(self, df):
        """Simulate trades using base DI strategy"""
        trades = []
        for i in range(1, len(df)):
            slice_df = df.iloc[:i+1]
            signal = check_signals(slice_df, self.ticker_combo.currentText())
            
            if signal:
                trades.append({
                    'time': df.index[i],
                    'price': df['close'].iloc[i],
                    'type': signal,
                    'confidence': get_confidence_from_di(
                        df['DI+'].iloc[i],
                        df['DI-'].iloc[i]
                    )
                })
        
        return trades

    def simulate_rl_strategy(self, df):
        """Simulate trades using RL agent"""
        if df is None or df.empty:
            return []
        
        trades = []
        self.trading_env.reset()
        
        for i in range(1, len(df)):
            slice_df = df.iloc[:i+1]
            state = self.trading_env.update_state(slice_df)
            signal = check_signals_with_rl(slice_df, self.ticker_combo.currentText(), 
                                         self.trading_agent, self.trading_env)
            
            if signal:
                trades.append({
                    'time': df.index[i],
                    'price': df['close'].iloc[i],
                    'type': signal,
                    'confidence': state.position.get('size', 0)
                })
        
        return trades

    def simulate_trades(self, df, strategy='RL'):
        """Simulate trades based on selected strategy"""
        if df is None or df.empty:
            return []
        
        if strategy == 'Base DI Strategy':
            return self.simulate_base_strategy(df)
        elif strategy == 'RL Strategy':
            return self.simulate_rl_strategy(df)
        else:  # Both strategies
            rl_trades = self.simulate_rl_strategy(df)
            base_trades = self.simulate_base_strategy(df)
            for trade in rl_trades:
                trade['strategy'] = 'RL'
            for trade in base_trades:
                trade['strategy'] = 'Base'
            return rl_trades + base_trades

    def plot_backtest(self, df, trades):
        """Plot backtest results with market hours formatting"""
        self.canvas.clear()
        
        # Set market hours limits
        eastern = pytz.timezone('US/Eastern')
        market_open = df.index[0].replace(hour=9, minute=30)
        market_close = df.index[0].replace(hour=16, minute=0)
        
        # Plot price and trades
        self.canvas.ax1.plot(df.index, df['close'], color='black', label='Price')
        
        # Plot trades with legends for each type
        rl_buys = []
        rl_sells = []
        base_buys = []
        base_sells = []
        
        for trade in trades:
            is_rl = 'strategy' not in trade or trade['strategy'] == 'RL'
            is_buy = trade['type'] == 'buy'
            
            if is_rl:
                if is_buy:
                    rl_buys.append(trade)
                else:
                    rl_sells.append(trade)
            else:
                if is_buy:
                    base_buys.append(trade)
                else:
                    base_sells.append(trade)
        
        # Plot each trade type with a single legend entry
        if rl_buys:
            times = [t['time'] for t in rl_buys]
            prices = [t['price'] for t in rl_buys]
            self.canvas.ax1.scatter(times, prices, color='green', marker='^', s=100, label='RL Buy')
        
        if rl_sells:
            times = [t['time'] for t in rl_sells]
            prices = [t['price'] for t in rl_sells]
            self.canvas.ax1.scatter(times, prices, color='red', marker='v', s=100, label='RL Sell')
        
        if base_buys:
            times = [t['time'] for t in base_buys]
            prices = [t['price'] for t in base_buys]
            self.canvas.ax1.scatter(times, prices, color='blue', marker='s', s=100, label='Base Buy')
        
        if base_sells:
            times = [t['time'] for t in base_sells]
            prices = [t['price'] for t in base_sells]
            self.canvas.ax1.scatter(times, prices, color='orange', marker='D', s=100, label='Base Sell')
        
        # Plot indicators
        self.canvas.ax2.plot(df.index, df['DI+'], color='green', label='DI+', linewidth=1.5)
        self.canvas.ax2.plot(df.index, df['DI-'], color='red', label='DI-', linewidth=1.5)
        
        # Plot confidence levels for RL trades
        rl_trades = [t for t in trades if 'strategy' not in t or t['strategy'] == 'RL']
        if rl_trades:
            confidences = [t['confidence'] for t in rl_trades]
            times = [t['time'] for t in rl_trades]
            colors = ['green' if t['type']=='buy' else 'red' for t in rl_trades]
            self.canvas.ax3.bar(times, confidences, color=colors, width=0.001)  # Adjusted width
            self.canvas.ax3.set_ylim([0, max(confidences) * 1.1])  # Add some padding
        
        # Format axes
        for ax in [self.canvas.ax1, self.canvas.ax2, self.canvas.ax3]:
            ax.set_xlim(market_open, market_close)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(9, 17)))
            ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Set titles and labels
        self.canvas.ax1.set_title(f'{self.ticker_combo.currentText()} - Price and Trades', pad=15)
        self.canvas.ax2.set_title('DI Indicators', pad=15)
        self.canvas.ax3.set_title('RL Agent Confidence', pad=15)
        
        self.canvas.ax1.legend(loc='upper left')
        self.canvas.ax2.legend(loc='upper left')
        
        # Add price range label
        price_range = f"Range: ${min(df['close']):.2f} - ${max(df['close']):.2f}"
        self.canvas.ax1.text(0.02, 0.98, price_range, 
                            transform=self.canvas.ax1.transAxes,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.8))
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def calculate_metrics(self, df, trades):
        """Calculate and format performance metrics"""
        if not trades:
            return "No trades generated"
        
        if any('strategy' in t for t in trades):
            rl_trades = [t for t in trades if t['strategy'] == 'RL']
            base_trades = [t for t in trades if t['strategy'] == 'Base']
            
            metrics = "RL Strategy:\n"
            metrics += self._calculate_strategy_metrics(rl_trades)
            metrics += "\n\nBase DI Strategy:\n"
            metrics += self._calculate_strategy_metrics(base_trades)
        else:
            metrics = self._calculate_strategy_metrics(trades)
        
        return metrics

    def _calculate_strategy_metrics(self, trades):
        """Calculate metrics for a single strategy with dollar values"""
        if not trades:
            return "No trades generated"
        
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        total_pl = 0.0
        total_pl_dollars = 0.0
        position = None
        shares = 0
        trades_closed = 0
        starting_cash = 100000.0  # Start with fixed simulation amount
        current_cash = starting_cash
        
        for trade in trades:
            if trade['type'] == 'buy' and position is None:
                position = trade['price']
                shares = current_cash * 0.1 / position  # Use 10% of cash per trade
                current_cash -= position * shares
            elif trade['type'] == 'sell' and position is not None:
                pl_pct = (trade['price'] - position) / position * 100
                pl_dollars = shares * (trade['price'] - position)
                total_pl += pl_pct
                total_pl_dollars += pl_dollars
                current_cash += trade['price'] * shares
                position = None
                trades_closed += 1
        
        avg_pl = total_pl / trades_closed if trades_closed > 0 else 0
        
        return (f"Total Signals: {len(trades)}\n"
                f"Buy Signals: {len(buy_trades)}\n"
                f"Sell Signals: {len(sell_trades)}\n"
                f"Completed Trades: {trades_closed}\n"
                f"Total P/L: {total_pl:.2f}% (${total_pl_dollars:.2f})\n"
                f"Average P/L per Trade: {avg_pl:.2f}%\n"
                f"Starting Cash: ${starting_cash:.2f}\n"
                f"Ending Cash: ${current_cash:.2f}\n"
                f"Net Change: ${(current_cash - starting_cash):.2f}")

    def run_backtest(self):
        """Execute backtest with selected parameters"""
        ticker = self.ticker_combo.currentText()
        date = self.date_picker.get_selected_date()
        strategy = self.strategy_combo.currentText()
        
        df = self.get_historical_data(ticker, date)
        if df is not None:
            df = calculate_indicators(df)
            trades = self.simulate_trades(df, strategy)
            
            self.plot_backtest(df, trades)
            self.metrics_text.setText(self.calculate_metrics(df, trades))
        else:
            self.metrics_text.setText("No data available for selected date")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = BacktestWindow()
    window.show()
    sys.exit(app.exec())