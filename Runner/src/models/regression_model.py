import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional, Union
import warnings


class RegressionModelHandler:
    """Handles loading the regression model and performing predictions with expanded features."""

    def __init__(self, model_dir: str = 'data/models/'):
        """
        Initialize the RegressionModelHandler.

        Args:
            model_dir (str): Directory containing saved models and scalers.
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None

        # Expanded feature columns to include macro + additional technicals
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower',
            'sma_50', 'sma_200', 'atr', 'volatility',
            # Additional technical indicators
            'stoch_k', 'stoch_d',  # Example of Stochastic oscillator
            # Macroeconomic indicators
            'vix_close',           # Example: VIX as a proxy for market volatility
            'treasury_10y',        # Example: 10-year Treasury yield
            # Add more (CPI, unemployment rate, etc.) as needed
        ]

        self._load_model_and_scaler()

    def _load_model_and_scaler(self) -> None:
        """Load the regression model and scaler from disk."""
        model_path = os.path.join(self.model_dir, 'regression_model.joblib')
        scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            warnings.warn("Model or scaler files not found. Predictions will be unavailable.")
            return

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            warnings.warn(f"Error loading model or scaler: {e}")

    @staticmethod
    def _fetch_historical_data(symbol: str, lookback_days: int = 60) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data using Yahoo Finance.

        Args:
            symbol (str): Stock ticker symbol.
            lookback_days (int): Number of days of historical data to fetch.

        Returns:
            Optional[pd.DataFrame]: Historical stock data or None if fetch fails.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            return data if not data.empty else None
        except Exception as e:
            warnings.warn(f"Error fetching data for {symbol}: {e}")
            return None

    @staticmethod
    def _fetch_macro_data(lookback_days: int = 60) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from external sources (e.g., Yahoo Finance).

        Returns:
            pd.DataFrame: A DataFrame containing macro data for the given date range.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Example: VIX data (market volatility index)
        # Ticker for VIX on Yahoo Finance is "^VIX"
        vix_df = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        # Example: 10-year Treasury yield
        # Ticker for 10yr T-bond on Yahoo Finance is "^TNX"
        tnx_df = yf.download("^TNX", start=start_date, end=end_date, progress=False)

        # Combine the macro data into a single DataFrame
        # We'll align on the Date index, and just pick the 'Close' columns for simplicity.
        macro_df = pd.DataFrame(index=vix_df.index)
        macro_df['vix_close'] = vix_df['Close']
        macro_df['treasury_10y'] = tnx_df['Close']

        # You can add more indicators here (CPI, etc.) from other sources
        return macro_df

    @staticmethod
    def _calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate core technical indicators required for the model.

        Args:
            data (pd.DataFrame): OHLCV data.

        Returns:
            pd.DataFrame: Data with additional technical indicator columns.
        """
        df = data.copy()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)  # add small epsilon to avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = (
            df['Close'].ewm(span=12, adjust=False).mean()
            - df['Close'].ewm(span=26, adjust=False).mean()
        )

        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20)
        rolling_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean.mean() + (rolling_std * 2)
        df['bb_lower'] = rolling_mean.mean() - (rolling_std * 2)

        # Moving Averages
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()

        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # Volatility (std dev of returns)
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()

        return df

    @staticmethod
    def _calculate_additional_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional technical features (e.g., Stochastic Oscillator).

        Args:
            df (pd.DataFrame): Data with at least OHLC columns.

        Returns:
            pd.DataFrame: Data with new technical features appended.
        """
        # Stochastic Oscillator
        # Using default 14-day lookback for High/Low. 
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))  # + epsilon
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # You can add more advanced features as needed
        return df

    def _merge_data(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge stock DataFrame with macroeconomic DataFrame on Date index.

        Args:
            stock_df (pd.DataFrame): Stock data with technical features.
            macro_df (pd.DataFrame): Macroeconomic data.

        Returns:
            pd.DataFrame: Merged DataFrame aligned on date index.
        """
        merged_df = stock_df.merge(macro_df, left_index=True, right_index=True, how='outer')
        # Forward fill or handle missing macro data as needed
        merged_df.fillna(method='ffill', inplace=True)
        return merged_df

    def prepare_features(self, symbol: str) -> Optional[np.ndarray]:
        """
        Prepare scaled features for the given stock symbol by merging
        both technical and macroeconomic features.

        Args:
            symbol (str): Stock ticker symbol.

        Returns:
            Optional[np.ndarray]: Scaled features or None if preparation fails.
        """
        if not self.scaler:
            warnings.warn("Scaler not loaded. Cannot prepare features.")
            return None

        # Fetch historical stock data
        stock_data = self._fetch_historical_data(symbol)
        if stock_data is None:
            warnings.warn(f"No data available for {symbol}.")
            return None

        # Fetch macro data
        macro_data = self._fetch_macro_data()

        # Calculate technical indicators
        stock_data = self._calculate_technical_indicators(stock_data)
        stock_data = self._calculate_additional_technical_features(stock_data)

        # Drop rows with NaN from calculations
        stock_data.dropna(inplace=True)

        if stock_data.empty:
            warnings.warn(f"Insufficient data after technical indicator calculations for {symbol}.")
            return None

        # Merge with macro data
        features = self._merge_data(stock_data, macro_data)

        # We only keep rows with all required features
        if not all(col in features.columns for col in self.feature_columns):
            warnings.warn(f"Not all required features are available for {symbol}.")
            return None

        # Drop rows with missing final feature columns
        features.dropna(subset=self.feature_columns, inplace=True)
        if features.empty:
            warnings.warn(f"No valid feature rows available after merging data for {symbol}.")
            return None

        # Take the most recent row for inference
        latest_features = features[self.feature_columns].iloc[-1].values.reshape(1, -1)

        # Scale features
        scaled_features = self.scaler.transform(latest_features)
        return scaled_features

    def predict(self, symbol: str) -> Dict[str, Union[float, str, None]]:
        """
        Generate a prediction for the given stock symbol.

        Args:
            symbol (str): Stock ticker symbol.

        Returns:
            Dict[str, Union[float, str, None]]: Prediction results.
        """
        if not self.model:
            return {"prediction": None, "confidence": None, "error": "Model not loaded"}

        features = self.prepare_features(symbol)
        if features is None:
            return {"prediction": None, "confidence": None, "error": "Feature preparation failed"}

        try:
            prediction = self.model.predict(features)[0]
            # If model supports predict_proba (e.g., ensemble methods), get confidence
            confidence = (
                max(self.model.predict_proba(features)[0]) if hasattr(self.model, 'predict_proba') else None
            )
            return {"prediction": float(prediction), "confidence": confidence, "error": None}
        except Exception as e:
            return {"prediction": None, "confidence": None, "error": f"Prediction error: {e}"}

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if supported by the model.

        Returns:
            Optional[Dict[str, float]]: Feature importance scores or None.
        """
        if not self.model or not hasattr(self.model, 'feature_importances_'):
            warnings.warn("Model does not support feature importance.")
            return None

        # Pair feature columns with importance scores
        return dict(
            sorted(
                zip(self.feature_columns, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )


def get_prediction(symbol: str) -> Dict[str, Union[float, str, None]]:
    """
    Convenience function to fetch predictions for a stock symbol.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        Dict[str, Union[float, str, None]]: Prediction results.
    """
    return RegressionModelHandler().predict(symbol)
