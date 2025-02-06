import logging
from datetime import datetime
import os

class TradingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create log filename with date
        log_filename = os.path.join(
            self.log_dir,
            f'trading_log_{datetime.now().strftime("%Y%m%d")}.log'
        )

        # Configure logging
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Create console handler for error messages
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        logging.getLogger('').addHandler(console)

    def log_trade(self, trade_record):
        """Log trade details"""
        logging.info("\nTrade executed:")
        logging.info(f"Ticker: {trade_record.get('ticker', 'Unknown')}")
        logging.info(f"Entry Price: {trade_record.get('entry_price', 0):.2f}")
        logging.info(f"Exit Price: {trade_record.get('exit_price', 0):.2f}")
        logging.info(f"P/L %: {trade_record.get('pl_pct', 0):.2f}%")
        logging.info(f"Position Size: {trade_record.get('size', 0)}")
        logging.info(f"Holding Time: {trade_record.get('holding_time', 0)} bars")

    def log_signal(self, ticker, signal_type, price, di_plus, di_minus):
        """Log trading signals"""
        logging.info(f"\nSignal detected for {ticker}")
        logging.info(f"Type: {signal_type}")
        logging.info(f"Price: {price:.2f}")
        logging.info(f"DI+: {di_plus:.2f}")
        logging.info(f"DI-: {di_minus:.2f}")

    def log_error(self, message, error=None):
        """Log error messages"""
        if error:
            logging.error(f"{message}: {str(error)}")
        else:
            logging.error(message)

    def log_warning(self, message):
        """Log warning messages"""
        logging.warning(message)

    def log_info(self, message):
        """Log informational messages"""
        logging.info(message)

    def log_market_data(self, ticker, data_info):
        """Log market data information"""
        logging.info(f"\nMarket Data Update for {ticker}")
        for key, value in data_info.items():
            logging.info(f"{key}: {value}")

    def log_performance(self, performance_metrics):
        """Log performance metrics"""
        logging.info("\nPerformance Update:")
        for metric, value in performance_metrics.items():
            logging.info(f"{metric}: {value}")

    # --- Add this alias method ---
    def info(self, message):
        self.log_info(message)
