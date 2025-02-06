import numpy as np
import pandas as pd

class TechnicalIndicators:
    """Custom implementation of technical indicators using numpy and pandas."""

    # @staticmethod
    # def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    #     """
    #     Calculate Relative Strength Index (RSI).
    #     Returns the last RSI value as a float.
    #     """
    #     if len(prices) < period+1:
    #         return 50.0  # Not enough data; return neutral

    #     deltas = np.diff(prices)
    #     gain = np.where(deltas > 0, deltas, 0.0)
    #     loss = np.where(deltas < 0, -deltas, 0.0)

    #     # initial means
    #     avg_gain = np.mean(gain[:period])
    #     avg_loss = np.mean(loss[:period])

    #     # rolling updates
    #     for i in range(period, len(deltas)):
    #         avg_gain = ((avg_gain * (period - 1)) + gain[i]) / period
    #         avg_loss = ((avg_loss * (period - 1)) + loss[i]) / period

    #     if avg_loss == 0:
    #         return 100.0

    #     rs = avg_gain / avg_loss
    #     rsi = 100 - (100 / (1 + rs))
    #     return float(rsi)

    @staticmethod
    def calculate_macd(df):
        """Calculate MACD components with error handling"""
        try:
            if df is None or df.empty:
                return None

            if 'close' not in df.columns:
                raise ValueError("DataFrame must contain 'close' column")

            # Calculate MACD components
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_fast - ema_slow
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            # Handle any NaN values
            df[['MACD', 'Signal_Line', 'MACD_Hist']] = df[['MACD', 'Signal_Line', 'MACD_Hist']].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            return None

    # @staticmethod
    # def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
    #     """
    #     Calculate Bollinger Bands for the last data point in the series.
    #     Returns a dict { upper, middle, lower }.
    #     """
    #     if len(prices) < period:
    #         return {'upper':0,'middle':0,'lower':0}

    #     sma = prices.rolling(window=period).mean()
    #     rolling_std = prices.rolling(window=period).std()

    #     upper_band = sma + (rolling_std * num_std)
    #     lower_band = sma - (rolling_std * num_std)

    #     return {
    #         'upper': float(upper_band.iloc[-1]),
    #         'middle': float(sma.iloc[-1]),
    #         'lower': float(lower_band.iloc[-1])
    #     }

    @staticmethod
    def calculate_adx_system(df: pd.DataFrame, len_period: int = 14, debug: bool = False) -> pd.DataFrame:
        """
        Calculate the ADX system: DI+, DI-, and ADX based on the input DataFrame.

        Parameters:
          df         : DataFrame with required columns (e.g., 'High', 'Low', 'Close').
          len_period : Lookback period for the calculations (default 14).
          debug      : If True, prints debug information.

        Returns:
          DataFrame with new columns: 'DI+', 'DI-', and 'ADX'. Returns None on error.
        """
        try:
            # Validate input DataFrame
            if df is None or df.empty:
                if debug:
                    print("DataFrame is None or empty")
                return None

            # Standardize column names to lowercase.
            df.columns = [col.lower() for col in df.columns]

            # Check required columns (after conversion, they must be in lowercase)
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    if debug:
                        print(f"Missing required column: {col}")
                    return None

            if debug:
                print("\nInput Data Sample:")
                print(df[required_columns].head())

            # Ensure enough data points; if not, you might choose to continue (with a warning)
            if len(df) < len_period + 1:
                if debug:
                    print(f"Not enough data points. Need at least {len_period + 1}, got {len(df)}")
                return None

            # Create shifted series for previous values
            prev_close = df['close'].shift(1)
            prev_high  = df['high'].shift(1)
            prev_low   = df['low'].shift(1)

            # Calculate True Range components
            high_low   = df['high'] - df['low']
            high_close = (df['high'] - prev_close).abs()
            low_close  = (df['low'] - prev_close).abs()

            # True Range is the maximum of the three
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            if debug:
                print("\nTrue Range Sample:")
                print(true_range.head())

            # Calculate directional movements
            up_move   = df['high'] - prev_high
            down_move = prev_low - df['low']
            if debug:
                print("\nUp Movement Sample:")
                print(up_move.head())
                print("\nDown Movement Sample:")
                print(down_move.head())

            # Positive and Negative DM: use np.where for vectorization
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            if debug:
                print("\nPositive DM Sample:")
                print(pos_dm[:5])
                print("\nNegative DM Sample:")
                print(neg_dm[:5])

            # Rolling sums over len_period – use min_periods=len_period so that we start calculating once we have enough points.
            tr_sum       = pd.Series(true_range).rolling(window=len_period, min_periods=len_period).sum()
            plus_dm_sum  = pd.Series(pos_dm, index=df.index).rolling(window=len_period, min_periods=len_period).sum()
            minus_dm_sum = pd.Series(neg_dm, index=df.index).rolling(window=len_period, min_periods=len_period).sum()

            # Calculate DI+ and DI-
            df['DI+'] = (plus_dm_sum / tr_sum) * 100
            df['DI-'] = (minus_dm_sum / tr_sum) * 100

            if debug:
                print("\nInitial DI+ values:")
                print(df['DI+'].head())
                print("\nInitial DI- values:")
                print(df['DI-'].head())

            # Calculate DX – avoid division by zero using np.where
            sum_di = df['DI+'] + df['DI-']
            dx = np.where(sum_di != 0, (np.abs(df['DI+'] - df['DI-']) / sum_di) * 100, 0)
            dx = pd.Series(dx, index=df.index)
            if debug:
                print("\nDX values:")
                print(dx.head())

            # Calculate ADX as the rolling average of DX (using min_periods=len_period)
            df['ADX'] = dx.rolling(window=len_period, min_periods=len_period).mean()

            if debug:
                print("\nFinal values before cleaning:")
                print(f"DI+: {df['DI+'].iloc[-1]:.2f}")
                print(f"DI-: {df['DI-'].iloc[-1]:.2f}")
                print(f"ADX: {df['ADX'].iloc[-1]:.2f}")

            # Clean up: replace Inf and NaN with forward-fill then 0 if still missing
            for col in ['DI+', 'DI-', 'ADX']:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

            if debug:
                print("\nFinal values after cleaning:")
                print(f"DI+: {df['DI+'].iloc[-1]:.2f}")
                print(f"DI-: {df['DI-'].iloc[-1]:.2f}")
                print(f"ADX: {df['ADX'].iloc[-1]:.2f}")

            return df

        except Exception as e:
            print(f"Error calculating ADX system: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    @staticmethod
    def wilder_smoothing(series, period):
        """Calculate Wilder's Smoothing"""
        result = series.copy()
        initial_avg = result.iloc[:period].mean()
        result.iloc[:period] = initial_avg
        for i in range(period, len(series)):
            result.iloc[i] = result.iloc[i-1] - (result.iloc[i-1] / period) + series.iloc[i]
        return result

    # @staticmethod
    # def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Calculate Heikin-Ashi candles from a DataFrame with OHLC data.
    #     Returns a new DataFrame with columns: HA_Open, HA_High, HA_Low, HA_Close.
    #     """
    #     ha_df = df.copy()
    #     ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    #     ha_open = [df['Open'].iloc[0]]
    #     for i in range(1, len(df)):
    #         ha_open.append((ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
    #     ha_df['HA_Open'] = ha_open
    #     ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    #     ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
    #     return ha_df
    
    # @staticmethod
    # def calculate_obv(close: np.ndarray, volume: np.ndarray) -> float:
    #     """
    #     Calculate On-Balance Volume (OBV). Returns the last OBV value.
    #     """
    #     if len(close) < 2:
    #         return 0.0
    #     obv = np.zeros_like(volume)
    #     obv[0] = volume[0]
    #     for i in range(1, len(close)):
    #         if close[i] > close[i-1]:
    #             obv[i] = obv[i-1] + volume[i]
    #         elif close[i] < close[i-1]:
    #             obv[i] = obv[i-1] - volume[i]
    #         else:
    #             obv[i] = obv[i-1]
    #     return float(obv[-1])

    # @staticmethod
    # def calculate_sma(prices: np.ndarray, period: int) -> float:
    #     """Calculate Simple Moving Average over the last `period` data points."""
    #     if len(prices) < period:
    #         return float(np.mean(prices))  # fallback
    #     return float(np.mean(prices[-period:]))

    # @staticmethod
    # def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    #     """
    #     Calculate the Average True Range for volatility measurement.
    #     """
    #     if len(high) < period+1:
    #         return 0.0

    #     tr1 = high[1:] - low[1:]
    #     tr2 = np.abs(high[1:] - close[:-1])
    #     tr3 = np.abs(low[1:] - close[:-1])
    #     tr = np.maximum(tr1, np.maximum(tr2, tr3))
    #     return float(np.mean(tr[-period:]))

    # @staticmethod
    # def calculate_vwap(df: pd.DataFrame, window: int = 20) -> float:
    #     """
    #     Calculate Volume-Weighted Average Price (VWAP) over a given lookback window.
    #     :param df: DataFrame with columns [Close, Volume] at least.
    #     :param window: Number of periods to look back for VWAP.
    #     :return: The last VWAP value.
    #     """
    #     if len(df) < window:
    #         window = len(df)  # fallback if not enough data

    #     recent = df.iloc[-window:]
    #     # Typical price is often (High + Low + Close) / 3, but we can use 'Close' for simplicity
    #     typical_price = (recent['High'] + recent['Low'] + recent['Close']) / 3
    #     total_vp = (typical_price * recent['Volume']).sum()
    #     total_vol = recent['Volume'].sum()

    #     if total_vol == 0:
    #         return float(df['Close'].iloc[-1])  # fallback
    #     vwap_value = total_vp / total_vol
    #     return float(vwap_value)

    # @staticmethod
    # def detect_support_resistance(df: pd.DataFrame, window_size: int = 5, tolerance: float = 0.005) -> dict:
    #     """
    #     A simple pivot-based support/resistance detection:
    #     - For each 'pivot point', check if price is higher/lower than surrounding bars.
    #     - Tolerance is how close subsequent pivot points must be to be considered the same S/R level.
    #     :return: Dictionary with lists 'support_levels' and 'resistance_levels'.
    #     """
    #     supports = []
    #     resistances = []
    #     highs = df['High'].values
    #     lows = df['Low'].values

    #     for i in range(window_size, len(df) - window_size):
    #         # A pivot high: local maximum
    #         if np.max(highs[i - window_size : i + window_size+1]) == highs[i]:
    #             # Check if it's close to an existing resistance
    #             close_to_existing = any(abs(r - highs[i]) < r * tolerance for r in resistances)
    #             if not close_to_existing:
    #                 resistances.append(highs[i])
            
    #         # A pivot low: local minimum
    #         if np.min(lows[i - window_size : i + window_size+1]) == lows[i]:
    #             close_to_existing = any(abs(s - lows[i]) < s * tolerance for s in supports)
    #             if not close_to_existing:
    #                 supports.append(lows[i])

    #     return {
    #         'support_levels': sorted(supports),
    #         'resistance_levels': sorted(resistances)
    #     }

    # @staticmethod
    # def detect_candlestick_pattern(df: pd.DataFrame) -> str:
    #     """
    #     A minimal candlestick pattern detection on the last bar, e.g. Doji, Hammer, etc.
    #     For demonstration, we check:
    #      - Hammer: body near the top, long lower shadow
    #      - Doji: open ~ close
    #     :param df: DataFrame with at least columns [Open, High, Low, Close].
    #     :return: A string indicating the pattern or 'None'.
    #     """
    #     if len(df) < 2:
    #         return "None"

    #     last_row = df.iloc[-1]
    #     open_price = last_row['Open']
    #     close_price = last_row['Close']
    #     high_price = last_row['High']
    #     low_price = last_row['Low']

    #     body_size = abs(close_price - open_price)
    #     candle_range = high_price - low_price
    #     if candle_range == 0:
    #         return "None"

    #     # Doji check: if the body size is very small relative to range
    #     if body_size < (0.1 * candle_range):
    #         return "Doji"

    #     # Hammer check: if the lower shadow is at least 2x the body, and close near top
    #     lower_shadow = min(open_price, close_price) - low_price
    #     if (lower_shadow >= 2 * body_size) and ((high_price - max(open_price, close_price)) < 0.2 * body_size):
    #         return "Hammer"

    #     return "None"

class SignalGenerator:
    def __init__(self, logger=None):
        self.logger = logger
        self.position = {}  # Dictionary to track positions for each ticker
        self.adx_threshold = 25  # Minimum ADX value for strong trend
        
    def initialize_position_tracker(self, ticker):
        """Initialize or reset position tracking for a ticker"""
        self.position[ticker] = {
            'in_position': False,
            'last_buy_price': None
        }

    def check_signals(self, df, ticker):
        """
        Generate trading signals based solely on DI and MACD crossovers with an ADX filter.
        
        - Buy Signal: When not in a position, buy if:
            (1) DI+ crosses above DI– (i.e. previously DI+ < DI– and now DI+ > DI–),
            (2) MACD crosses above the Signal Line (i.e. previously MACD < Signal and now MACD > Signal),
            (3) AND the ADX filter is met: previous ADX was below DI+ and current ADX is above DI+.
            
        - Sell Signal: When in a position, sell if:
            (1) DI+ crosses below DI– (i.e. previously DI+ > DI– and now DI+ < DI–),
            (2) MACD crosses below the Signal Line (i.e. previously MACD > Signal and now MACD < Signal),
            (3) AND the ADX filter is met: previous ADX was below DI– and current ADX is above DI–.
        """
        if df is None or len(df) < 2:
            return None

        try:
            # Ensure we have a position tracker for this ticker
            if ticker not in self.position:
                self.initialize_position_tracker(ticker)
                # Example structure:
                # self.position[ticker] = {'in_position': False, 'last_buy_price': None}

            # Calculate MACD if not already present
            if 'MACD' not in df.columns or 'Signal_Line' not in df.columns:
                df = TechnicalIndicators.calculate_macd(df)

            # Use the last two rows for checking crossovers
            current_idx = len(df) - 1
            prev_idx = current_idx - 1

            # DI values:
            di_plus_current = df['DI+'].iloc[current_idx]
            di_minus_current = df['DI-'].iloc[current_idx]
            di_plus_prev = df['DI+'].iloc[prev_idx]
            di_minus_prev = df['DI-'].iloc[prev_idx]

            # ADX values:
            adx_current = df['ADX'].iloc[current_idx]
            adx_prev = df['ADX'].iloc[prev_idx]

            # MACD and Signal Line values:
            macd_current = df['MACD'].iloc[current_idx]
            signal_current = df['Signal_Line'].iloc[current_idx]
            macd_prev = df['MACD'].iloc[prev_idx]
            signal_prev = df['Signal_Line'].iloc[prev_idx]

            close_price = df['close'].iloc[current_idx]

            # --- Buy Signal Conditions ---
            # (a) DI+ crossover: previous DI+ was below DI- and now DI+ > DI-
            di_buy = di_plus_prev < di_minus_prev and di_plus_current > di_minus_current
            # (b) MACD crossover: previous MACD < previous Signal and now MACD > Signal
            macd_buy = macd_prev < signal_prev and macd_current > signal_current
            # (c) ADX filter: previous ADX < DI+ and current ADX > DI+
            adx_buy = adx_prev < di_plus_prev and adx_current > di_plus_current

            # --- Sell Signal Conditions ---
            # (a) DI+ crossover: previous DI+ was above DI- and now DI+ < DI-
            di_sell = di_plus_prev > di_minus_prev and di_plus_current < di_minus_current
            # (b) MACD crossover: previous MACD > previous Signal and now MACD < Signal
            macd_sell = macd_prev > signal_prev and macd_current < signal_current
            # (c) ADX filter: previous ADX < DI- and current ADX > DI-
            adx_sell = adx_prev < di_minus_prev and adx_current > di_minus_current

            if not self.position[ticker]['in_position']:
                if di_buy and macd_buy and adx_buy:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Buy signal - DI+ crossed above DI- (prev: {di_plus_prev:.2f} < {di_minus_prev:.2f}; now: {di_plus_current:.2f} > {di_minus_current:.2f}), "
                            f"MACD crossed above Signal (prev: {macd_prev:.3f} < {signal_prev:.3f}; now: {macd_current:.3f} > {signal_current:.3f}), "
                            f"and ADX filter met (prev: {adx_prev:.3f} < DI+ {di_plus_prev:.2f}; now: {adx_current:.3f} > DI+ {di_plus_current:.2f})."
                        )
                    self.position[ticker]['in_position'] = True
                    self.position[ticker]['last_buy_price'] = close_price
                    return 'buy'
            else:
                if di_sell and macd_sell and adx_sell:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Sell signal - DI+ crossed below DI- (prev: {di_plus_prev:.2f} > {di_minus_prev:.2f}; now: {di_plus_current:.2f} < {di_minus_current:.2f}), "
                            f"MACD crossed below Signal (prev: {macd_prev:.3f} > {signal_prev:.3f}; now: {macd_current:.3f} < {signal_current:.3f}), "
                            f"and ADX filter met (prev: {adx_prev:.3f} < DI- {di_minus_prev:.2f}; now: {adx_current:.3f} > DI- {di_minus_current:.2f})."
                        )
                    self.position[ticker]['in_position'] = False
                    self.position[ticker]['last_buy_price'] = None
                    return 'sell'

            return None

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error generating signals for {ticker}: {str(e)}")
            return None
    
    def check_short_signals(self, df, ticker):
        """
        Generate short signals based on the following conditions:
        
        Entry (sell short):
        - ADX crosses above DI– (i.e. previously ADX was below DI–, now above)
        - DI– is above DI+ in the current period
        - MACD crosses above the Signal Line (i.e. previously MACD was below the signal, now above)
        
        While in a short position:
        - If the bearish signal persists (DI– remains above DI+ and MACD > Signal), then signal to add to the position.
        - Cover (exit the short) if:
            a) The profit target of 2% is reached (i.e. the current price is at least 2% below the entry price),
            b) The stop loss of 1% is hit (i.e. the current price is 1% above the entry price), or
            c) A reversal occurs: DI+ crosses above DI– and MACD crosses below its Signal.
        """
        if df is None or len(df) < 2:
            return None

        try:
            # Ensure we have a position tracker for this ticker
            if ticker not in self.position:
                self.initialize_position_tracker(ticker)
                # Example tracker structure:
                # self.position[ticker] = {'in_short': False, 'short_entry_price': None}

            # Ensure MACD and Signal_Line are available
            if 'MACD' not in df.columns or 'Signal_Line' not in df.columns:
                df = TechnicalIndicators.calculate_macd(df)

            # Use the last two rows for detecting crossovers
            current_idx = len(df) - 1
            prev_idx = current_idx - 1

            # Retrieve current and previous indicator values:
            di_plus_current = df['DI+'].iloc[current_idx]
            di_minus_current = df['DI-'].iloc[current_idx]
            di_plus_prev = df['DI+'].iloc[prev_idx]
            di_minus_prev = df['DI-'].iloc[prev_idx]

            macd_current = df['MACD'].iloc[current_idx]
            signal_current = df['Signal_Line'].iloc[current_idx]
            macd_prev = df['MACD'].iloc[prev_idx]
            signal_prev = df['Signal_Line'].iloc[prev_idx]

            close_price = df['close'].iloc[current_idx]

            # For ADX conditions, if available:
            if 'ADX' in df.columns:
                adx_current = df['ADX'].iloc[current_idx]
                adx_prev = df['ADX'].iloc[prev_idx]
            else:
                adx_current = adx_prev = None

            # --- If NOT already short, check for entry signal ---
            if not self.position[ticker].get('in_short', False):
                # Entry conditions:
                # 1. (If ADX is used) ADX crosses above DI–: previously ADX < DI–, now ADX > DI–
                # 2. DI– is above DI+ (i.e. bearish condition)
                # 3. MACD crosses above its Signal Line: previously MACD < Signal_Line, now MACD > Signal_Line.
                entry_cond = (
                    (adx_current is None or (adx_prev is not None and adx_prev < di_minus_prev and adx_current > di_minus_current))
                    and (di_minus_current > di_plus_current)
                    and (macd_prev < signal_prev and macd_current > signal_current)
                )
                if entry_cond:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Short Entry signal triggered. "
                            f"Indicators: DI+={di_plus_current:.2f}, DI-={di_minus_current:.2f}, "
                            f"MACD={macd_current:.3f}, Signal={signal_current:.3f}"
                        )
                    self.position[ticker]['in_short'] = True
                    self.position[ticker]['short_entry_price'] = close_price
                    return 'sell_short'

            # --- If already in a short position, check for adding or exit signals ---
            else:
                entry_price = self.position[ticker]['short_entry_price']
                # For a short, profit occurs when price falls relative to entry.
                profit_pct = ((entry_price - close_price) / entry_price) * 100
                loss_pct = ((close_price - entry_price) / entry_price) * 100

                # Profit target: 2% gain
                if profit_pct >= 2:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Profit target reached ({profit_pct:.2f}%). Cover short."
                        )
                    self.position[ticker]['in_short'] = False
                    self.position[ticker]['short_entry_price'] = None
                    return 'cover_profit'

                # Stop loss: 1% loss
                if loss_pct >= 1:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Stop loss hit ({loss_pct:.2f}%). Cover short."
                        )
                    self.position[ticker]['in_short'] = False
                    self.position[ticker]['short_entry_price'] = None
                    return 'cover_stop'

                # Additional short signal: if bearish conditions persist
                add_cond = (
                    (di_minus_prev < di_plus_prev and di_minus_current > di_plus_current)
                    and (macd_current > signal_current)
                )
                if add_cond:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Additional short signal: conditions persist. Consider adding shares."
                        )
                    return 'add_short'

                # Exit (cover) signal if the conditions reverse: 
                # For example, DI+ crosses above DI- and MACD crosses below Signal.
                exit_cond = (
                    (di_plus_prev > di_minus_prev and di_plus_current < di_minus_current)
                    and (macd_current < signal_current)
                )
                if exit_cond:
                    if self.logger:
                        self.logger.log_info(
                            f"{ticker}: Exit signal: conditions reversed. Cover short."
                        )
                    self.position[ticker]['in_short'] = False
                    self.position[ticker]['short_entry_price'] = None
                    return 'cover_signal'

            return None

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error generating short signals for {ticker}: {str(e)}")
            return None

    def get_signal_strength(self, df):
        """Calculate signal strength based on DI, ADX, and MACD"""
        try:
            if df is None or len(df) < 2:
                return 0.0

            current_idx = len(df) - 1
            
            di_plus = df['DI+'].iloc[current_idx]
            di_minus = df['DI-'].iloc[current_idx]
            adx = df['ADX'].iloc[current_idx]
            macd_hist = df['MACD_Hist'].iloc[current_idx]
            
            # Calculate DI strength (0 to 1)
            di_strength = max(0, min(1, (di_plus - di_minus) / 100))
            
            # Calculate ADX strength (0 to 1)
            adx_strength = min(1.0, adx / 50.0)  # Normalized, considers ADX above 50 as maximum strength
            
            # Calculate MACD strength (0 to 1)
            macd_strength = abs(macd_hist) / (abs(macd_hist) + 1)
            
            # Combine strengths with weights
            # 40% DI, 30% ADX, 30% MACD
            total_strength = (0.4 * di_strength) + (0.3 * adx_strength) + (0.3 * macd_strength)
            
            return total_strength

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error calculating signal strength: {str(e)}")
            return 0.0

    def adjust_position_size(self, base_size, signal_strength):
        """Adjust position size based on signal strength and ADX"""
        # Scale position size between 50% and 100% of base size based on signal strength
        return base_size * (0.5 + (0.5 * signal_strength))