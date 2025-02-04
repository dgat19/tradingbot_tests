import numpy as np
import pandas as pd

class TechnicalIndicators:
    """Custom implementation of technical indicators using numpy and pandas."""

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).
        Returns the last RSI value as a float.
        """
        if len(prices) < period+1:
            return 50.0  # Not enough data; return neutral

        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0.0)
        loss = np.where(deltas < 0, -deltas, 0.0)

        # initial means
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        # rolling updates
        for i in range(period, len(deltas)):
            avg_gain = ((avg_gain * (period - 1)) + gain[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + loss[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
        """
        Calculate MACD (Moving Average Convergence Divergence) using Pandas ewm.
        Returns a dict with macd, signal, hist, prev_hist.
        """
        if len(prices) < slow_period + 1:
            return {'macd':0,'signal':0,'hist':0,'prev_hist':0}

        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        hist = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'hist': hist.iloc[-1],
            'prev_hist': hist.iloc[-2] if len(hist) > 1 else 0
        }

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
        """
        Calculate Bollinger Bands for the last data point in the series.
        Returns a dict { upper, middle, lower }.
        """
        if len(prices) < period:
            return {'upper':0,'middle':0,'lower':0}

        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()

        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)

        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1])
        }

    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> dict:
        """
        Calculate the Average Directional Index (ADX) along with DI+ and DI-.
        Returns a dictionary with keys: 'adx', 'di_plus', 'di_minus'.
        """
        try:
            if len(high) < period + 1:
                return {'adx': 0.0, 'di_plus': 0.0, 'di_minus': 0.0}

            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))

            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]

            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            tr_sm = np.zeros_like(tr)
            pos_dm_sm = np.zeros_like(pos_dm)
            neg_dm_sm = np.zeros_like(neg_dm)

            tr_sm[0] = tr[0]
            pos_dm_sm[0] = pos_dm[0]
            neg_dm_sm[0] = neg_dm[0]

            for i in range(1, len(tr)):
                tr_sm[i] = tr_sm[i-1] - (tr_sm[i-1] / period) + tr[i]
                pos_dm_sm[i] = pos_dm_sm[i-1] - (pos_dm_sm[i-1] / period) + pos_dm[i]
                neg_dm_sm[i] = neg_dm_sm[i-1] - (neg_dm_sm[i-1] / period) + neg_dm[i]

            epsilon = 1e-10
            di_plus = 100.0 * (pos_dm_sm / (tr_sm + epsilon))
            di_minus = 100.0 * (neg_dm_sm / (tr_sm + epsilon))

            dx = np.zeros_like(di_plus)
            denom = di_plus + di_minus
            valid = denom != 0
            dx[valid] = 100.0 * np.abs(di_plus[valid] - di_minus[valid]) / denom[valid]

            adx = np.mean(dx[-period:])
            return {
                'adx': float(adx),
                'di_plus': float(di_plus[-1]),
                'di_minus': float(di_minus[-1])
            }
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return {'adx': 0.0, 'di_plus': 0.0, 'di_minus': 0.0}

    @staticmethod
    def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candles from a DataFrame with OHLC data.
        Returns a new DataFrame with columns: HA_Open, HA_High, HA_Low, HA_Close.
        """
        ha_df = df.copy()
        ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_open = [df['Open'].iloc[0]]
        for i in range(1, len(df)):
            ha_open.append((ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
        ha_df['HA_Open'] = ha_open
        ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
        return ha_df
    
    @staticmethod
    def calculate_obv(close: np.ndarray, volume: np.ndarray) -> float:
        """
        Calculate On-Balance Volume (OBV). Returns the last OBV value.
        """
        if len(close) < 2:
            return 0.0
        obv = np.zeros_like(volume)
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return float(obv[-1])

    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average over the last `period` data points."""
        if len(prices) < period:
            return float(np.mean(prices))  # fallback
        return float(np.mean(prices[-period:]))

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """
        Calculate the Average True Range for volatility measurement.
        """
        if len(high) < period+1:
            return 0.0

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def calculate_vwap(df: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate Volume-Weighted Average Price (VWAP) over a given lookback window.
        :param df: DataFrame with columns [Close, Volume] at least.
        :param window: Number of periods to look back for VWAP.
        :return: The last VWAP value.
        """
        if len(df) < window:
            window = len(df)  # fallback if not enough data

        recent = df.iloc[-window:]
        # Typical price is often (High + Low + Close) / 3, but we can use 'Close' for simplicity
        typical_price = (recent['High'] + recent['Low'] + recent['Close']) / 3
        total_vp = (typical_price * recent['Volume']).sum()
        total_vol = recent['Volume'].sum()

        if total_vol == 0:
            return float(df['Close'].iloc[-1])  # fallback
        vwap_value = total_vp / total_vol
        return float(vwap_value)

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window_size: int = 5, tolerance: float = 0.005) -> dict:
        """
        A simple pivot-based support/resistance detection:
        - For each 'pivot point', check if price is higher/lower than surrounding bars.
        - Tolerance is how close subsequent pivot points must be to be considered the same S/R level.
        :return: Dictionary with lists 'support_levels' and 'resistance_levels'.
        """
        supports = []
        resistances = []
        highs = df['High'].values
        lows = df['Low'].values

        for i in range(window_size, len(df) - window_size):
            # A pivot high: local maximum
            if np.max(highs[i - window_size : i + window_size+1]) == highs[i]:
                # Check if it's close to an existing resistance
                close_to_existing = any(abs(r - highs[i]) < r * tolerance for r in resistances)
                if not close_to_existing:
                    resistances.append(highs[i])
            
            # A pivot low: local minimum
            if np.min(lows[i - window_size : i + window_size+1]) == lows[i]:
                close_to_existing = any(abs(s - lows[i]) < s * tolerance for s in supports)
                if not close_to_existing:
                    supports.append(lows[i])

        return {
            'support_levels': sorted(supports),
            'resistance_levels': sorted(resistances)
        }

    @staticmethod
    def detect_candlestick_pattern(df: pd.DataFrame) -> str:
        """
        A minimal candlestick pattern detection on the last bar, e.g. Doji, Hammer, etc.
        For demonstration, we check:
         - Hammer: body near the top, long lower shadow
         - Doji: open ~ close
        :param df: DataFrame with at least columns [Open, High, Low, Close].
        :return: A string indicating the pattern or 'None'.
        """
        if len(df) < 2:
            return "None"

        last_row = df.iloc[-1]
        open_price = last_row['Open']
        close_price = last_row['Close']
        high_price = last_row['High']
        low_price = last_row['Low']

        body_size = abs(close_price - open_price)
        candle_range = high_price - low_price
        if candle_range == 0:
            return "None"

        # Doji check: if the body size is very small relative to range
        if body_size < (0.1 * candle_range):
            return "Doji"

        # Hammer check: if the lower shadow is at least 2x the body, and close near top
        lower_shadow = min(open_price, close_price) - low_price
        if (lower_shadow >= 2 * body_size) and ((high_price - max(open_price, close_price)) < 0.2 * body_size):
            return "Hammer"

        return "None"