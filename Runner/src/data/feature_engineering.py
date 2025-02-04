# feature_engineering.py

import pandas as pd
from src.indicators import add_technical_indicators
from sklearn.preprocessing import StandardScaler
from yfinance import Ticker
import os

class FeatureEngineer:
    def __init__(self, mode='live', data_dir='data/raw'):
        """
        Initialize the feature pipeline.
        Args:
            mode (str): 'live' for live trading or 'backtest' for historical data.
            data_dir (str): Directory for historical data (used in backtest mode).
        """
        self.mode = mode
        self.data_dir = data_dir
        self.scaler = StandardScaler() if mode == 'backtest' else None

    def load_data(self, ticker):
        """
        Load data based on mode.
        Args:
            ticker (str): Stock ticker symbol.
        Returns:
            pd.DataFrame: Loaded data.
        """
        if self.mode == 'live':
            # Fetch real-time data
            data = Ticker(ticker).history(period='1d', interval='1m')
            return data
        elif self.mode == 'backtest':
            # Load historical data from CSV
            file_path = os.path.join(self.data_dir, f"{ticker}_historical.csv")
            if os.path.exists(file_path):
                return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            else:
                raise FileNotFoundError(f"Historical data not found for {ticker} at {file_path}")

    def engineer_features(self, data):
        """
        Add features to the data.
        Args:
            data (pd.DataFrame): Input price data.
        Returns:
            pd.DataFrame: Data with added features.
        """
        # Add technical indicators
        data = add_technical_indicators(data)

        if self.mode == 'backtest':
            # Add lagged features for backtesting
            data['return_1d'] = data['Close'].pct_change(1)
            data['return_5d'] = data['Close'].pct_change(5)
            data.dropna(inplace=True)

            # Scale features
            feature_cols = ['RSI', 'MACD', 'return_1d', 'return_5d']
            data[feature_cols] = self.scaler.fit_transform(data[feature_cols])

        return data

    def process(self, ticker):
        """
        Process the data pipeline for a given ticker.
        Args:
            ticker (str): Stock ticker symbol.
        Returns:
            pd.DataFrame: Processed data.
        """
        data = self.load_data(ticker)
        return self.engineer_features(data)


# Example usage
if __name__ == "__main__":
    # For live trading
    pipeline = FeatureEngineer(mode='live')
    live_data = pipeline.process('AAPL')
    print("Live Data Processed:", live_data.head())

    # For backtesting
    pipeline = FeatureEngineer(mode='backtest')
    historical_data = pipeline.process('AAPL')
    print("Backtest Data Processed:", historical_data.head())