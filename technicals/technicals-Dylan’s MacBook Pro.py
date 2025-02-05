import numpy as np
import pandas as pd
import logging

class TechnicalIndicators:
    """
    Custom implementation of technical indicators using numpy and pandas.
    Includes VWAP, support/resistance, and candlestick pattern detection.
    """

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate the RSI for the last value of a price series."""
        logger = logging.getLogger('backtest')  # Ensure the logger name matches your setup

        if len(prices) < period + 1:
            logger.debug("Insufficient data for RSI calculation. Returning neutral RSI of 50.0.")
            return 50.0  # Neutral RSI when insufficient data

        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0.0)
        loss = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        logger.debug(f"Initial avg_gain: {avg_gain}, avg_loss: {avg_loss}")

        for i in range(period, len(deltas)):
            avg_gain = ((avg_gain * (period - 1)) + gain[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + loss[i]) / period

        # Ensure avg_gain and avg_loss are scalars
        if isinstance(avg_gain, np.ndarray):
            if avg_gain.size == 1:
                avg_gain = float(avg_gain.item())
            else:
                logger.error("avg_gain is an array with size greater than 1. Cannot convert to float.")
                return 50.0  # Neutral RSI in case of unexpected data
        else:
            avg_gain = float(avg_gain)

        if isinstance(avg_loss, np.ndarray):
            if avg_loss.size == 1:
                avg_loss = float(avg_loss.item())
            else:
                logger.error("avg_loss is an array with size greater than 1. Cannot convert to float.")
                return 50.0  # Neutral RSI in case of unexpected data
        else:
            avg_loss = float(avg_loss)

        if avg_loss == 0:
            logger.debug("No losses detected. RSI is set to 100.0.")
            return 100.0  # Avoid division by zero; RSI is 100 when no losses

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        logger.debug(f"RSI calculated: {rsi}")
        return float(rsi)

    @staticmethod
    def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        Returns the most recent {'macd', 'signal', 'hist', 'prev_hist'}.
        """
        logger = logging.getLogger('backtest')  # Ensure the logger name matches your setup

        if len(prices) < slow_period + 1:
            logger.debug("Insufficient data for MACD calculation. Returning zeros.")
            return {'macd': 0.0, 'signal': 0.0, 'hist': 0.0, 'prev_hist': 0.0}

        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        hist = macd_line - signal_line

        if len(hist) < 1:
            logger.error("MACD histogram is empty. Returning zeros.")
            return {'macd': 0.0, 'signal': 0.0, 'hist': 0.0, 'prev_hist': 0.0}

        macd_val = float(macd_line.iloc[-1]) if not macd_line.empty else float('nan')
        signal_val = float(signal_line.iloc[-1]) if not signal_line.empty else float('nan')
        hist_val = float(hist.iloc[-1]) if not hist.empty else float('nan')
        prev_hist_val = float(hist.iloc[-2]) if len(hist) > 1 else 0.0

        logger.debug(f"MACD calculated: macd={macd_val}, signal={signal_val}, hist={hist_val}, prev_hist={prev_hist_val}")

        return {
            'macd': macd_val,
            'signal': signal_val,
            'hist': hist_val,
            'prev_hist': prev_hist_val
        }

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
        """
        Calculate Bollinger Bands for the last data point.
        Returns {'upper', 'middle', 'lower'} as floats.
        """
        logger = logging.getLogger('backtest')  # Ensure the logger name matches your setup

        if len(prices) < period:
            logger.debug("Insufficient data for Bollinger Bands calculation. Returning zeros.")
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}

        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()

        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)

        if upper_band.empty or sma.empty or lower_band.empty:
            logger.error("One of the Bollinger Bands components is empty. Returning zeros.")
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}

        upper_val = float(upper_band.iloc[-1]) if not upper_band.empty else float('nan')
        middle_val = float(sma.iloc[-1]) if not sma.empty else float('nan')
        lower_val = float(lower_band.iloc[-1]) if not lower_band.empty else float('nan')

        logger.debug(f"Bollinger Bands calculated: upper={upper_val}, middle={middle_val}, lower={lower_val}")

        return {
            'upper': upper_val,
            'middle': middle_val,
            'lower': lower_val
        }

    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """
        Calculate the Average Directional Index (ADX) for the last value.
        """
        if len(high) < period + 1:
            return 0.0

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
        pos_di = 100.0 * (pos_dm_sm / (tr_sm + epsilon))
        neg_di = 100.0 * (neg_dm_sm / (tr_sm + epsilon))

        dx = np.zeros_like(pos_di)
        denom = pos_di + neg_di
        valid = denom != 0
        dx[valid] = 100.0 * np.abs(pos_di[valid] - neg_di[valid]) / denom[valid]

        adx = np.mean(dx[-period:])
        return float(adx)

    @staticmethod
    def calculate_obv(close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate On-Balance Volume (OBV)."""
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
            return float(np.mean(prices))
        return float(np.mean(prices[-period:]))

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range (ATR)."""
        logger = logging.getLogger('backtest')  # Ensure the logger name matches your setup

        if len(high) < period + 1:
            logger.debug("Insufficient data for ATR calculation. Returning 0.0.")
            return 0.0

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = float(np.mean(tr[-period:])) if len(tr[-period:]) > 0 else 0.0

        logger.debug(f"ATR calculated: {atr}")
        return atr

    @staticmethod
    def calculate_vwap(df: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate Volume-Weighted Average Price (VWAP) for the last `window` bars.
        If there's not enough data, it uses whatever is available.
        """
        if len(df) < 1:
            return 0.0
        lookback = min(window, len(df))
        recent = df.iloc[-lookback:]
        typical_price = (recent['High'] + recent['Low'] + recent['Close']) / 3
        total_vp = (typical_price * recent['Volume']).sum()
        total_vol = recent['Volume'].sum()
        if total_vol == 0:
            return float(df['Close'].iloc[-1])
        return float(total_vp / total_vol)

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window_size: int = 5, tolerance: float = 0.005) -> dict:
        """
        A simple pivot-based approach:
        - For each bar, check local maxima/minima within window_size.
        - Combine points close to each other within tolerance as a single S/R.
        Returns {'support_levels', 'resistance_levels'} sorted ascending.
        """
        if len(df) < 2 * window_size + 1:
            return {'support_levels': [], 'resistance_levels': []}

        supports = []
        resistances = []
        highs = df['High'].values
        lows = df['Low'].values

        for i in range(window_size, len(df) - window_size):
            if np.max(highs[i - window_size : i + window_size + 1]) == highs[i]:
                close_to_existing = any(abs(r - highs[i]) < r * tolerance for r in resistances)
                if not close_to_existing:
                    resistances.append(highs[i])
            if np.min(lows[i - window_size : i + window_size + 1]) == lows[i]:
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
        Basic candlestick detection on the last row:
        - 'Doji' if body < 10% of range
        - 'Hammer' if long lower shadow, body near top
        Returns 'Doji', 'Hammer', or 'None'.
        """
        logger = logging.getLogger('backtest')  # Ensure the logger name matches your setup

        if len(df) < 2:
            logger.debug("Insufficient data for candlestick pattern detection. Returning 'None'.")
            return "None"

        last_row = df.iloc[-1]
        op = last_row['Open']
        cl = last_row['Close']
        hi = last_row['High']
        lo = last_row['Low']

        body_size = abs(cl - op)
        candle_range = hi - lo
        if candle_range == 0:
            logger.debug("Candle range is zero. Returning 'None'.")
            return "None"

        if body_size < 0.1 * candle_range:
            logger.debug("Doji pattern detected.")
            return "Doji"

        lower_shadow = min(op, cl) - lo
        if (lower_shadow >= 2 * body_size) and ((hi - max(op, cl)) < 0.2 * body_size):
            logger.debug("Hammer pattern detected.")
            return "Hammer"

        logger.debug("No candlestick pattern detected.")
        return "None"
