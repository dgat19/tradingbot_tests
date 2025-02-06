import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets, QtCore
import numpy as np
import pytz
from datetime import datetime, timedelta
from indicators import SignalGenerator
from reinforcement import check_signals_with_rl

def round_up_to_next_half_hour(dt):
        """
        Round up the given datetime dt to the next half-hour mark.
        For example, 10:07 becomes 10:30, and 10:30 remains 10:30.
        """
        # Remove seconds and microseconds for rounding purposes.
        dt = dt.replace(second=0, microsecond=0)
        remainder = dt.minute % 30
        if remainder == 0:
            return dt
        else:
            return dt + timedelta(minutes=(30 - remainder))

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=8, dpi=100):  # Increased height for third subplot
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Create three subplots: price, DI lines, and MACD
        self.ax1 = self.fig.add_subplot(311)  # Price chart
        self.ax2 = self.fig.add_subplot(312, sharex=self.ax1)  # DI indicators
        self.ax3 = self.fig.add_subplot(313, sharex=self.ax1)  # MACD
        super().__init__(self.fig)
        self.fig.tight_layout()
        # Connect mouse motion event to our handler
        self.mpl_connect("motion_notify_event", self.on_mouse_move)

    def on_mouse_move(self, event):
        # If mouse is not over any axis, do nothing.
        if event.inaxes is None:
            return
        x = event.xdata
        # For each axis in the canvas, draw a vertical dashed line at x.
        for ax in [self.ax1, self.ax2, self.ax3]:
            # Remove previous vertical line if present
            if hasattr(ax, '_vline'):
                try:
                    ax._vline.remove()
                except Exception:
                    pass
            ax._vline = ax.axvline(x=x, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        self.draw()

class TradingPlotter:
    def __init__(self, logger, trading_env=None, trading_agent=None):
        self.logger = logger
        self.trades = []
        self.central_tz = pytz.timezone('US/Central')
        self.trading_env = trading_env
        self.trading_agent = trading_agent
        
        # Setup auto-refresh timer
        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.timeout.connect(self.refresh_plots)
        self.refresh_timer.start(15000)  # Refresh every 15 seconds
        
        self.active_canvases = {}

    def plot_di_for_ticker(self, ticker, df, canvas):
        """Plot DI indicators, price, and MACD for a ticker"""
        try:
            if df is None or df.empty:
                for ax in [canvas.ax1, canvas.ax2, canvas.ax3]:
                    ax.clear()
                canvas.ax1.text(0.5, 0.5, f"No data available for {ticker}", 
                            transform=canvas.ax1.transAxes,
                            ha="center", va="center", color="black")
                canvas.draw()
                return

            # Calculate MACD if not present
            if 'MACD' not in df.columns or 'Signal_Line' not in df.columns or 'MACD_Hist' not in df.columns:
                ema_fast = df['close'].ewm(span=12, adjust=False).mean()
                ema_slow = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema_fast - ema_slow
                df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

            # Store canvas reference
            self.active_canvases[ticker] = canvas

            # Convert timestamps to central time
            df.index = df.index.tz_convert(self.central_tz)

            # Plot price chart
            canvas.ax1.clear()
            canvas.ax1.plot(df.index, df['close'], color="black", label="Close Price")
            
            # Add current time and price annotation
            current_price = df['close'].iloc[-1]
            last_update = df.index[-1].strftime('%H:%M CT')
            current_time = datetime.now(self.central_tz).strftime('%H:%M:%S CT')
            
            canvas.ax1.text(0.95, 0.85, 
                        f"Current Time: {current_time}\n"
                        f"Last Update: {last_update}\n"
                        f"Price: {current_price:.2f}",
                        transform=canvas.ax1.transAxes,
                        ha="right", va="center",
                        color="blue", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.7))
            
            canvas.ax1.set_ylabel("Price", color="black")
            canvas.ax1.legend(loc="upper left")
            
            # Plot DI lines
            canvas.ax2.clear()
            canvas.ax2.plot(df.index, df["DI+"], color="green", label="DI+")
            canvas.ax2.plot(df.index, df["DI-"], color="red", label="DI-")
            if 'ADX' in df.columns:
                canvas.ax2.plot(df.index, df["ADX"], color="blue", label="ADX", linestyle='--')
            canvas.ax2.set_ylabel("DI/ADX (%)", color="black")
            canvas.ax2.legend(loc="upper left")

            # Plot historical signals
            self._plot_historical_signals(ticker, df, canvas)
            
            # Plot MACD
            canvas.ax3.clear()
            canvas.ax3.bar(df.index, df['MACD_Hist'], 
                        color=['red' if x < 0 else 'green' for x in df['MACD_Hist']],
                        alpha=0.3, label='Histogram')
            canvas.ax3.plot(df.index, df['MACD'], color='blue', label='MACD')
            canvas.ax3.plot(df.index, df['Signal_Line'], color='black', label='Signal',
                        linestyle='--')
            canvas.ax3.set_ylabel("MACD", color="black")
            canvas.ax3.legend(loc="upper left")
            
            # Add zero line for MACD
            canvas.ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            # Set x-axis limits and format
            x_left = df.index.min()
            current_dt = datetime.now(self.central_tz)
            x_right = round_up_to_next_half_hour(current_dt)

            for ax in [canvas.ax1, canvas.ax2, canvas.ax3]:
                ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(matplotlib.dates.HourLocator())
                ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
                ax.tick_params(rotation=45)
                ax.set_xlim(x_left, x_right)
                ax.set_facecolor("#eeeeee")
                ax.tick_params(colors="black")
                for spine in ax.spines.values():
                    spine.set_color("black")
            
            # Add trade markers
            self._plot_trade_markers(ticker, df, canvas)
            
            # Update title
            macd_value = df['MACD'].iloc[-1]
            signal_value = df['Signal_Line'].iloc[-1]
            hist_value = df['MACD_Hist'].iloc[-1]
            
            title = (f"{ticker} - {current_time}\n"
                    f"MACD: {macd_value:.3f} | Signal: {signal_value:.3f} | "
                    f"Hist: {hist_value:.3f}")
            canvas.ax1.set_title(title, color="black", pad=10)
            
            # Add padding between subplots
            canvas.fig.subplots_adjust(hspace=0.3)
            canvas.draw()

        except Exception as e:
            self.logger.log_error(f"Error plotting data for {ticker}: {str(e)}")
            import traceback
            self.logger.log_error(traceback.format_exc())

    def _plot_historical_signals(self, ticker, df, canvas):
        """Plot historical signals throughout the day"""
        try:
            # Create signal generator instance
            signal_gen = SignalGenerator(self.logger)
            
            # Track RL agent position state
            rl_position = False
            entry_price = None
            
            # Analyze signals for each point in time
            for i in range(30, len(df)):  # Start after enough bars for indicators
                historical_df = df.iloc[:i+1].copy()  # Historical data up to this point
                
                # Get base signal for this point in time
                base_signal = signal_gen.check_signals(historical_df, ticker)
                
                # Plot base signals
                if base_signal:
                    price = historical_df['close'].iloc[-1]
                    signal_time = historical_df.index[-1]
                    
                    if base_signal == 'buy':
                        canvas.ax1.scatter(signal_time, price, color='lightgreen', 
                                        marker='^', s=100, alpha=0.5, zorder=5,
                                        label='Base Buy Signal')
                    elif base_signal == 'sell':
                        canvas.ax1.scatter(signal_time, price, color='pink', 
                                        marker='v', s=100, alpha=0.5, zorder=5,
                                        label='Base Sell Signal')
                
                # Add RL signals if environment and agent are available
                if self.trading_env and self.trading_agent:
                    current_state = self.trading_env.update_state(historical_df)
                    if current_state is not None:
                        rl_signal = check_signals_with_rl(historical_df, ticker, 
                                                        self.trading_agent, self.trading_env)
                        
                        if rl_signal:
                            price = historical_df['close'].iloc[-1]
                            signal_time = historical_df.index[-1]
                            
                            if rl_signal == 'buy' and not rl_position:
                                canvas.ax1.scatter(signal_time, price, color='purple', 
                                                marker='D', s=120, alpha=0.7, zorder=6,
                                                label='RL Buy Signal')
                                rl_position = True
                                entry_price = price
                                
                                # Add marker on DI plot
                                canvas.ax2.scatter(signal_time, historical_df['DI+'].iloc[-1], 
                                                color='purple', marker='D', s=100, alpha=0.7, zorder=6)
                                
                            elif rl_signal == 'sell' and rl_position:
                                canvas.ax1.scatter(signal_time, price, color='orange', 
                                                marker='X', s=120, alpha=0.7, zorder=6,
                                                label='RL Sell Signal')
                                rl_position = False
                                
                                # Add profit/loss annotation
                                if entry_price:
                                    pl_pct = ((price - entry_price) / entry_price) * 100
                                    canvas.ax1.annotate(f'{pl_pct:.1f}%', 
                                                    xy=(signal_time, price),
                                                    xytext=(10, 10),
                                                    textcoords='offset points',
                                                    color='orange',
                                                    alpha=0.7)
                                
                                # Add marker on DI plot
                                canvas.ax2.scatter(signal_time, historical_df['DI-'].iloc[-1], 
                                                color='orange', marker='X', s=100, alpha=0.7, zorder=6)
            
            # Remove duplicate labels
            handles, labels = canvas.ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            canvas.ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
                
        except Exception as e:
            self.logger.log_error(f"Error plotting historical signals for {ticker}: {str(e)}")

    def refresh_plots(self):
        """Refresh all active plots"""
        for ticker, canvas in self.active_canvases.items():
            if not canvas.isVisible():
                continue
            self.logger.log_info(f"Refreshing plot for {ticker}")
            # Emit signal to main thread to update plot
            QtCore.QMetaObject.invokeMethod(canvas, 
                                          "update",
                                          QtCore.Qt.ConnectionType.QueuedConnection)

    def _plot_trade_markers(self, ticker, df, canvas):
        """Add trade markers to the plot"""
        for trade_info in self.trades:
            if trade_info['ticker'] == ticker:
                trade_time = trade_info['time']
                if not trade_time.tzinfo:
                    trade_time = self.central_tz.localize(trade_time)
                
                try:
                    diffs = np.abs((df.index - trade_time).total_seconds())
                    idx = diffs.argmin()
                    closest_time = df.index[idx]
                except Exception as e:
                    closest_time = trade_time
                
                if closest_time in df.index:
                    price_val = df.loc[closest_time, "close"]
                    signal_strength = trade_info.get('signal_strength', None)
                    
                    # Configure marker properties:
                    # If this is a virtual (RL) trade, use different markers/colors.
                    if trade_info.get('virtual', True):
                        if trade_info['side'] == 'buy':
                            marker = 'D'  # Diamond for virtual buy
                            color = 'purple'
                        else:
                            marker = 'X'  # X marker for virtual sell
                            color = 'orange'
                        offset = 2 if trade_info['side'] == 'buy' else -2
                        annotation = f"RL {trade_info['side'].upper()}\n${price_val:.2f}"
                        if signal_strength:
                            annotation += f"\nStr: {signal_strength:.1%}"
                    else:
                        # Actual executed trade markers.
                        if trade_info['side'] == 'buy':
                            marker = '^'
                            color = 'green'
                            offset = 2
                            annotation = f"BUY\n${price_val:.2f}"
                            if signal_strength:
                                annotation += f"\nStr: {signal_strength:.1%}"
                        else:
                            marker = 'v'
                            color = 'red'
                            offset = -2
                            annotation = f"SELL\n${price_val:.2f}"
                            if signal_strength:
                                annotation += f"\nStr: {signal_strength:.1%}"
                    
                    # Plot markers and annotations on the price chart.
                    self._add_trade_marker(canvas.ax1, closest_time, price_val, 
                                            color, marker, annotation, offset)
                    
                    # Also plot a marker on the DI subplot.
                    di_val = (df.loc[closest_time, "DI+"] if trade_info['side'] == 'buy'
                            else df.loc[closest_time, "DI-"])
                    canvas.ax2.scatter(closest_time, di_val, 
                                    color=color, marker=marker, s=80, zorder=5)

    def _add_trade_marker(self, ax, time, price, color, marker, annotation, offset):
        """Add trade marker and annotation to plot"""
        ax.scatter(time, price, color=color, marker=marker, s=100, zorder=5)
        ax.annotate(annotation,
                   xy=(time, price),
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

    def add_trade(self, trade_info):
        """Add trade to the trades list"""
        if not trade_info['time'].tzinfo:
            trade_info['time'] = self.central_tz.localize(trade_info['time'])
        self.trades.append(trade_info)


class PerformancePlotter(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trade Performance Distribution")
        self.setGeometry(300, 200, 800, 600)
        
        # Setup UI
        self._setup_ui()
        self.profits = []
        
        # Auto-refresh timer
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_plot)
        self.refresh_timer.start(60000)  # Refresh every minute
        
    def _setup_ui(self):
        """Setup the UI components"""
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)
        
        # Add timestamp label
        self.timestamp_label = QtWidgets.QLabel()
        layout.addWidget(self.timestamp_label)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_plot)
        layout.addWidget(self.refresh_button)
        
    def update_data(self, profits):
        """Update performance data"""
        self.profits = profits.copy()
        self.refresh_plot()
        
    def refresh_plot(self):
        """Refresh the performance plot"""
        if not self.profits:
            return
            
        # Update timestamp
        current_time = datetime.now(pytz.timezone('US/central'))
        self.timestamp_label.setText(f"Last Updated: {current_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
            
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Plot histogram
        ax.hist(self.profits, bins=30, color='blue', alpha=0.7)
        
        # Calculate and plot statistics
        mean_profit = np.mean(self.profits)
        std_profit = np.std(self.profits)
        median_profit = np.median(self.profits)
        
        ax.axvline(mean_profit, color='green', linestyle='--', 
                  label=f'Mean: {mean_profit:.2f}%')
        ax.axvline(mean_profit + std_profit, color='red', linestyle=':', 
                  label=f'+1 Std: {(mean_profit + std_profit):.2f}%')
        ax.axvline(mean_profit - std_profit, color='red', linestyle=':', 
                  label=f'-1 Std: {(mean_profit - std_profit):.2f}%')
        ax.axvline(median_profit, color='yellow', linestyle='-', 
                  label=f'Median: {median_profit:.2f}%')
        
        # Enhanced statistics
        win_rate = len([p for p in self.profits if p > 0]) / len(self.profits)
        max_drawdown = min(self.profits)
        profit_factor = abs(sum([p for p in self.profits if p > 0]) / 
                          sum([p for p in self.profits if p < 0])) if any(p < 0 for p in self.profits) else float('inf')
        
        stats_text = (
            f'Total Trades: {len(self.profits)}\n'
            f'Win Rate: {win_rate:.1%}\n'
            f'Mean: {mean_profit:.2f}%\n'
            f'Median: {median_profit:.2f}%\n'
            f'Std Dev: {std_profit:.2f}%\n'
            f'Max Drawdown: {max_drawdown:.2f}%\n'
            f'Profit Factor: {profit_factor:.2f}\n'
            f'Min: {min(self.profits):.2f}%\n'
            f'Max: {max(self.profits):.2f}%'
        )
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_title('Trade Profit Distribution', pad=20)
        ax.set_xlabel('Profit %')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()