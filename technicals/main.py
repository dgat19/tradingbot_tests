import sys
import threading
import time
from PySide6 import QtWidgets, QtCore

from alpaca_handler import AlpacaHandler
from market_hours import MarketHours
from indicators import TechnicalIndicators, SignalGenerator
from logs.trading_logger import TradingLogger
from display_handler import DisplayHandler
from plotting import TradingPlotter, PerformancePlotter, MplCanvas
from reinforcement import EnhancedTradingAgent, EnhancedTradingEnvironment, integrate_rl_agent, check_signals_with_rl

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DI Trading System")
        self.setGeometry(200, 100, 1000, 600)
        
        # Initialize components
        self.setup_components()
        self.setup_ui()
        
        # Initialize performance plotter (for trade profit distribution)
        self.performance_plotter = PerformancePlotter()
        
        # Initialize RL metrics history (for example, tracking epsilon)
        self.rl_epsilon_history = []
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
        
        # Update performance plot periodically
        self.perf_timer = QtCore.QTimer(self)
        self.perf_timer.timeout.connect(self.update_performance_plot)
        self.perf_timer.start(60000)  # Update every minute

        # Update RL plot periodically (every minute)
        self.rl_timer = QtCore.QTimer(self)
        self.rl_timer.timeout.connect(self.update_rl_plot)
        self.rl_timer.start(60000)

    def setup_components(self):
        """Initialize all trading components"""
        # Setup logger
        self.logger = TradingLogger()
        
        # Initialize components
        self.alpaca = AlpacaHandler()
        self.market_hours = MarketHours()
        self.tech_indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        
        # Initialize RL components first
        self.trading_env = EnhancedTradingEnvironment(logger=self.logger)
        self.trading_agent = EnhancedTradingAgent(
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0
        )
        
        # Initialize display and plotting with RL components
        self.display_handler = DisplayHandler(self.logger)
        self.trading_plotter = TradingPlotter(
            self.logger,
            trading_env=self.trading_env,
            trading_agent=self.trading_agent
        )
        
        # Try to load existing Q-table
        try:
            self.trading_agent.load('trading_qtable.pkl')
            self.logger.log_info("Loaded existing Q-table")
        except Exception as e:
            self.logger.log_info("Starting with new Q-table")
        
        # Trading parameters
        self.tickers = ['AAPL', 'GOOG', 'MSFT', 'META', 'NVDA', 'PLTR','TSLA',  'SPY']

    def setup_ui(self):
        """Setup the user interface"""
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
        
        # Update plots frequently (for ticker charts)
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.refresh_all_plots)
        self.plot_timer.start(15000)  # 15 seconds

    def create_tabs(self):
        """Create tabs for each ticker plus an extra tab for RL Metrics"""
        self.tab_widget.clear()
        for ticker in self.tickers:
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()
            
            try:
                # Create canvas for this ticker
                canvas = MplCanvas(self, width=8, height=5, dpi=100)  # Use MplCanvas directly
                layout.addWidget(canvas)
                self.canvases[ticker] = canvas
            except Exception as e:
                self.logger.log_error(f"Error creating canvas for {ticker}: {str(e)}")
                continue
            
            widget.setLayout(layout)
            self.tab_widget.addTab(widget, ticker)
            self.canvases[ticker] = canvas
            
            # Initial plot
            self.plot_ticker(ticker)
        
        # Create an extra tab for RL metrics
        rl_widget = QtWidgets.QWidget()
        rl_layout = QtWidgets.QVBoxLayout()
        self.rl_canvas = MplCanvas(self, width=8, height=5, dpi=100)
        rl_layout.addWidget(self.rl_canvas)
        rl_widget.setLayout(rl_layout)
        self.tab_widget.addTab(rl_widget, "RL Metrics")

    def plot_ticker(self, ticker):
        """Plot data for a specific ticker"""
        if ticker in self.canvases:
            df = self.alpaca.get_historical_data(ticker)
            if df is not None:
                df = self.tech_indicators.calculate_adx_system(df)  # Use calculate_adx_system instead
                self.trading_plotter.plot_di_for_ticker(ticker, df, self.canvases[ticker])

    def refresh_all_plots(self):
        """Refresh all ticker plots"""
        current_tab = self.tab_widget.currentWidget()
        if current_tab:
            current_ticker = self.tab_widget.tabText(self.tab_widget.currentIndex())
            # Update current tab immediately
            if current_ticker in self.canvases:
                self.plot_ticker(current_ticker)
        
        # Update other tabs in background
        for ticker in self.tickers:
            if ticker != current_tab:
                self.plot_ticker(ticker)

    def process_ticker(self, ticker):
        """Process trading logic for a ticker including RL integration"""
        try:
            intraday_df = self.alpaca.get_historical_data(ticker)
            if intraday_df is None or intraday_df.empty:
                self.logger.log_warning(f"No intraday data for {ticker}")
                return
            
            intraday_df = self.tech_indicators.calculate_adx_system(intraday_df)
            intraday_df = self.tech_indicators.calculate_macd(intraday_df)
            
            self.display_handler.display_ticker_info(ticker, intraday_df)
            
            # Update RL state and integrate new training data.
            # Pass self.signal_generator so the RL module uses the same signals.
            self.trading_agent = integrate_rl_agent(intraday_df, ticker, self.trading_agent, self.trading_env, self.signal_generator, self.trading_plotter)
            
            # Get the base signal using the common signal function.
            signal = self.signal_generator.check_signals(intraday_df, ticker)
            signal_strength = self.signal_generator.get_signal_strength(intraday_df)
            
            # Execute trade if real conditions are met.
            if signal:
                self.execute_trade(ticker, signal, intraday_df, signal_strength)
            
            # Additionally, even if no real trade is executed, record the RL trade signals as virtual trades.
            if signal:
                virtual_trade = {
                    'ticker': ticker,
                    'side': signal,
                    'time': intraday_df.index[-1],
                    'price': intraday_df['close'].iloc[-1],
                    'quantity': 0,  # Virtual signal
                    'signal_strength': signal_strength,
                    'virtual': True
                }
                self.trading_plotter.add_trade(virtual_trade)
            
        except Exception as e:
            self.logger.log_error(f"Error processing {ticker}: {str(e)}")

    def execute_trade(self, ticker, signal, df, signal_strength):
        """Execute trading orders with position sizing based on signal strength.
        Supports both long and short trades.
        """
        try:
            current_price = df['close'].iloc[-1]
            account_cash = self.alpaca.get_account_info()

            # --- Long Trade Logic ---
            if signal == 'buy':
                # Base position size (maximum 10% of account or $100k)
                max_position = min(account_cash * 0.1, 100000)
                # Adjust position size based on signal strength
                position_size = max_position * signal_strength
                # Ensure minimum position size (20% of max)
                min_position = max_position * 0.2
                position_size = max(min_position, position_size)
                qty = position_size / current_price

                self.logger.log_info(
                    f"{ticker} Buy Order Details:\n"
                    f"Signal Strength: {signal_strength:.2f}\n"
                    f"Position Size: ${position_size:,.2f}\n"
                    f"Quantity: {qty:.2f} shares"
                )
                
                filled_price = self.alpaca.place_bracket_order(ticker, 'buy', qty, current_price)
                if filled_price:
                    trade_info = {
                        'ticker': ticker,
                        'side': 'buy',
                        'time': df.index[-1],
                        'price': filled_price,
                        'quantity': qty,
                        'signal_strength': signal_strength
                    }
                    self.trading_plotter.add_trade(trade_info)
                    self.trading_env.record_trade(trade_info)

            elif signal == 'sell':
                current_qty = self.alpaca.get_position_qty(ticker)
                if current_qty > 0:
                    filled_price = self.alpaca.place_market_sell(ticker, current_qty, current_price)
                    if filled_price:
                        trade_info = {
                            'ticker': ticker,
                            'side': 'sell',
                            'time': df.index[-1],
                            'price': filled_price,
                            'quantity': current_qty,
                            'signal_strength': signal_strength
                        }
                        self.trading_plotter.add_trade(trade_info)
                        self.trading_env.record_trade(trade_info)

            # --- Short Trade Logic ---
            elif signal == 'sell_short':
                # Base short position size (maximum 10% of account or $100k)
                max_position = min(account_cash * 0.1, 100000)
                position_size = max_position * signal_strength
                min_position = max_position * 0.2
                position_size = max(min_position, position_size)
                qty = position_size / current_price

                self.logger.log_info(
                    f"{ticker} Short Sell Order Details:\n"
                    f"Signal Strength: {signal_strength:.2f}\n"
                    f"Position Size: ${position_size:,.2f}\n"
                    f"Quantity: {qty:.2f} shares"
                )
                # Place a short order using your broker's API (this method must be implemented)
                filled_price = self.alpaca.place_short_order(ticker, qty, current_price)
                if filled_price:
                    trade_info = {
                        'ticker': ticker,
                        'side': 'sell_short',
                        'time': df.index[-1],
                        'price': filled_price,
                        'quantity': qty,
                        'signal_strength': signal_strength
                    }
                    self.trading_plotter.add_trade(trade_info)
                    self.trading_env.record_trade(trade_info)

            elif signal in ['cover_profit', 'cover_stop', 'cover_signal']:
                # Cover short: buy shares to close the short position.
                current_qty = self.alpaca.get_short_position_qty(ticker)
                if current_qty > 0:
                    filled_price = self.alpaca.place_cover_order(ticker, current_qty, current_price)
                    if filled_price:
                        trade_info = {
                            'ticker': ticker,
                            'side': 'cover_short',
                            'time': df.index[-1],
                            'price': filled_price,
                            'quantity': current_qty,
                            'signal_strength': signal_strength
                        }
                        self.trading_plotter.add_trade(trade_info)
                        self.trading_env.record_trade(trade_info)

            elif signal == 'add_short':
                # Add to existing short position.
                max_position = min(account_cash * 0.1, 100000)
                position_size = max_position * signal_strength
                min_position = max_position * 0.2
                position_size = max(min_position, position_size)
                qty = position_size / current_price

                self.logger.log_info(
                    f"{ticker} Additional Short Sell Order Details:\n"
                    f"Signal Strength: {signal_strength:.2f}\n"
                    f"Position Size: ${position_size:,.2f}\n"
                    f"Quantity: {qty:.2f} shares"
                )
                filled_price = self.alpaca.place_short_order(ticker, qty, current_price)
                if filled_price:
                    trade_info = {
                        'ticker': ticker,
                        'side': 'add_short',
                        'time': df.index[-1],
                        'price': filled_price,
                        'quantity': qty,
                        'signal_strength': signal_strength
                    }
                    self.trading_plotter.add_trade(trade_info)
                    self.trading_env.record_trade(trade_info)

        except Exception as e:
            self.logger.log_error(f"Error executing trade for {ticker}: {str(e)}")

        
    def update_performance_plot(self):
        """Update the performance plot with latest trading results"""
        try:
            # Get trading history from environment
            trades = self.trading_env.trade_history
            if trades:
                # Calculate profit/loss percentages
                profits = [trade.get('pl_pct', 0) for trade in trades]
                
                # Show performance plot
                if not self.performance_plotter.isVisible():
                    self.performance_plotter.show()
                    
                # Update plot data
                self.performance_plotter.update_data(profits)
                
        except Exception as e:
            self.logger.log_error(f"Error updating performance plot: {str(e)}")
    
    def update_rl_plot(self):
        """Update the RL metrics plot (currently plotting agent epsilon over time)"""
        if hasattr(self, 'rl_canvas') and self.rl_epsilon_history:
            ax = self.rl_canvas.ax1
            ax.clear()
            ax.plot(self.rl_epsilon_history, marker='o', linestyle='-', color='purple', label="Epsilon")
            ax.set_title("RL Agent Epsilon Decay")
            ax.set_xlabel("Training Cycle")
            ax.set_ylabel("Epsilon")
            ax.legend()
            self.rl_canvas.draw()

    def trading_loop(self):
        """Main trading loop"""
        while True:
            try:
                self.logger.log_info("\n----- Starting new trading cycle -----")
                account_cash = self.alpaca.get_account_info()
                self.display_handler.display_account_info(account_cash)
                
                for ticker in self.tickers:
                    self.process_ticker(ticker)
                    time.sleep(2)  # Delay between tickers
                    
                # Update target network periodically
                self.trading_agent.update_target_network()
                
                # Save agent state
                self.trading_agent.save('trading_qtable.pkl')
                
                self.logger.log_info("----- Trading cycle complete -----")
                time.sleep(60)  # Wait before next cycle
                
            except Exception as e:
                self.logger.log_error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
            
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())