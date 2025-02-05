import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

from technicals import TechnicalIndicators
from strategy_optimizer import StrategyOptimizer

class OptionsBacktester:
    """
    An optimized class for backtesting options trading strategies with realistic pricing
    and proper risk management.
    """

    def __init__(self, watchlist: List[str], initial_capital: float = 100000.0):
        """Initialize backtest environment with risk parameters."""
        # Core settings
        self.watchlist = watchlist
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Technical analysis
        self.ti = TechnicalIndicators()
        self.strategy_optimizer = StrategyOptimizer()
        self.is_optimizer_trained = False
        
        # Risk management
        self.max_position_size = 0.02      # Max 2% per trade
        self.max_positions = 5             # Max concurrent positions
        self.max_symbol_exposure = 0.20    # Max 20% per symbol
        self.min_trade_size = 1000.0       # Minimum trade size
        
        # Stop loss and take profit
        self.default_stop_loss = 0.15      # 15% stop loss
        self.default_take_profit = 0.25    # 25% take profit
        self.trailing_stop = 0.10          # 10% trailing stop
        
        # Trading costs
        self.commission_rate = 0.0065      # $0.65 per contract
        self.slippage = 0.01              # 1% slippage estimate
        
        # State tracking
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.open_positions: List[Dict[str, Any]] = []
        self.technical_data_collection = {}  # For ML training
        self.trade_history: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = self._setup_logger()
        
        self.logger.info(f"Initialized OptionsBacktester with ${initial_capital:,.2f} capital")

    def _setup_logger(self) -> logging.Logger:
        """Configure logging with both file and console output."""
        logger = logging.getLogger('backtest')
        logger.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG
        
        if not logger.handlers:
            fh = logging.FileHandler('backtest.log')
            fh.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger

    def download_data(self, start_date: str, end_date: str) -> None:
        """
        Download and store historical price data for all symbols in the watchlist
        using the user-provided date range.
        """
        for symbol in self.watchlist:
            try:
                self.logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}...")
                df = yf.download(symbol, start=start_date, end=end_date)
                if df.empty:
                    self.logger.warning(f"No data found for {symbol} within {start_date} to {end_date}.")
                    continue
                self.historical_data[symbol] = df
                self.logger.info(f"Retrieved {len(df)} rows of data for {symbol}.")
            except Exception as e:
                self.logger.error(f"Error downloading data for {symbol}: {e}")

    def _collect_technical_data(self, window_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for analysis and ML training."""
        try:
            # Basic size check on the entire window
            if window_data is None or len(window_data) < 200:
                self.logger.debug("Window data is None or has insufficient length.")
                return {}
            
            # Extract arrays and ensure they are 1D using ravel()
            close = window_data['Close'].values.ravel()
            high = window_data['High'].values.ravel()
            low = window_data['Low'].values.ravel()
            volume = window_data['Volume'].values.ravel()

            # Log shapes for debugging
            self.logger.debug(f"Close shape: {close.shape}")
            self.logger.debug(f"High shape: {high.shape}")
            self.logger.debug(f"Low shape: {low.shape}")
            self.logger.debug(f"Volume shape: {volume.shape}")

            # Ensure these arrays actually have enough elements
            if close.size < 200 or high.size < 200 or low.size < 200 or volume.size < 200:
                self.logger.debug("One or more price/volume arrays have insufficient size.")
                return {}

            # Now it's safe to compute indicators
            rsi = self.ti.calculate_rsi(close)
            adx = self.ti.calculate_adx(high, low, close)
            obv = self.ti.calculate_obv(close, volume)
            
            # Extract the last value if the result is a scalar or a 1D array
            rsi_val = rsi if isinstance(rsi, float) else float(rsi) if isinstance(rsi, (np.ndarray, pd.Series)) and rsi.size > 0 else float('nan')
            adx_val = adx if isinstance(adx, float) else float(adx) if isinstance(adx, (np.ndarray, pd.Series)) and adx.size > 0 else float('nan')
            obv_val = obv if isinstance(obv, float) else float(obv) if isinstance(obv, (np.ndarray, pd.Series)) and obv.size > 0 else float('nan')

            tech_data = {
                'RSI': rsi_val,
                'ADX': adx_val,
                'OBV': obv_val,
                'volume': float(volume[-1]),
                'avg_volume': float(np.mean(volume[-200:])),
                'current_price': float(close[-1])
            }

            # MACD
            macd_data = self.ti.calculate_macd(pd.Series(close))
            if macd_data and all(k in macd_data for k in ['macd', 'signal', 'hist']):
                macd_val = {
                    'macd': float(macd_data['macd']),
                    'signal': float(macd_data['signal']),
                    'hist': float(macd_data['hist'])
                }
                tech_data['MACD'] = macd_val
                self.logger.debug(f"MACD calculated: {macd_val}")
            else:
                tech_data['MACD'] = {'macd': float('nan'), 'signal': float('nan'), 'hist': float('nan')}
                self.logger.warning("MACD calculation missing required keys or returned invalid data.")

            # Bollinger Bands
            bb_data = self.ti.calculate_bollinger_bands(pd.Series(close))
            if bb_data and all(k in bb_data for k in ['upper', 'middle', 'lower']):
                bb_val = {
                    'upper': float(bb_data['upper']),
                    'middle': float(bb_data['middle']),
                    'lower': float(bb_data['lower'])
                }
                tech_data['BB'] = bb_val
                self.logger.debug(f"Bollinger Bands calculated: {bb_val}")
            else:
                tech_data['BB'] = {'upper': float('nan'), 'middle': float('nan'), 'lower': float('nan')}
                self.logger.warning("Bollinger Bands calculation missing required keys or returned invalid data.")

            # Moving Averages
            sma_50 = self.ti.calculate_sma(close, 50)
            sma_200 = self.ti.calculate_sma(close, 200)
            sma_50_val = sma_50 if isinstance(sma_50, float) else float(sma_50) if isinstance(sma_50, (np.ndarray, pd.Series)) and sma_50.size > 0 else float('nan')
            sma_200_val = sma_200 if isinstance(sma_200, float) else float(sma_200) if isinstance(sma_200, (np.ndarray, pd.Series)) and sma_200.size > 0 else float('nan')

            tech_data['MA_DATA'] = {
                'SMA_50': sma_50_val,
                'SMA_200': sma_200_val
            }
            self.logger.debug(f"Moving Averages calculated: {tech_data['MA_DATA']}")

            return tech_data

        except Exception as e:
            self.logger.error(f"Error calculating technical data: {e}", exc_info=True)
            return {}
        
    def _generate_trading_signal(self, tech_data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Generate optimized trading signals based on technical analysis."""
        try:
            rsi = tech_data.get('RSI', 50.0)
            adx = tech_data.get('ADX', 25.0)
            macd = tech_data.get('MACD', {})
            bb = tech_data.get('BB', {})
            ma_data = tech_data.get('MA_DATA', {})
            
            signal = {'strategy': 'NO_TRADE', 'confidence': 0.0}
            
            # Ensure that rsi and adx are scalars
            if not isinstance(rsi, float) or not isinstance(adx, float):
                self.logger.warning(f"RSI or ADX is not a float. RSI type: {type(rsi)}, ADX type: {type(adx)}. Skipping signal generation.")
                return signal
            
            strong_trend = adx > 25.0
            sma_50 = ma_data.get('SMA_50', current_price)
            sma_200 = ma_data.get('SMA_200', current_price)
            
            # Ensure that sma_50 and sma_200 are scalars
            if not isinstance(sma_50, float) or not isinstance(sma_200, float):
                self.logger.warning(f"SMA_50 or SMA_200 is not a float. SMA_50 type: {type(sma_50)}, SMA_200 type: {type(sma_200)}. Skipping signal generation.")
                return signal
            
            bullish_trend = sma_50 > sma_200
            
            macd_hist = macd.get('hist', 0.0)
            if not isinstance(macd_hist, float):
                self.logger.warning(f"MACD histogram is not a float. MACD hist type: {type(macd_hist)}. Skipping signal generation.")
                return signal
            
            macd_increasing = macd_hist > 0.0
            bb_middle = bb.get('middle', current_price)
            
            # Ensure that bb_middle is a scalar
            if not isinstance(bb_middle, float):
                self.logger.warning(f"Bollinger Bands middle is not a float. BB middle type: {type(bb_middle)}. Skipping signal generation.")
                return signal
            
            # Ensure that current_price is a float
            if not isinstance(current_price, float):
                self.logger.warning(f"Current price is not a float. Current price type: {type(current_price)}. Skipping signal generation.")
                return signal
            
            # Calculate price_above_middle and enforce it as a boolean
            price_above_middle = bool(current_price > bb_middle)
            
            # Log the variables for debugging
            self.logger.debug(f"RSI: {rsi} (type: {type(rsi)}), ADX: {adx} (type: {type(adx)})")
            self.logger.debug(f"SMA_50: {sma_50} (type: {type(sma_50)}), SMA_200: {sma_200} (type: {type(sma_200)})")
            self.logger.debug(f"Strong Trend: {strong_trend} (type: {type(strong_trend)})")
            self.logger.debug(f"MACD Hist: {macd_hist} (type: {type(macd_hist)}), MACD Increasing: {macd_increasing} (type: {type(macd_increasing)})")
            self.logger.debug(f"Price: {current_price} (type: {type(current_price)}), BB Middle: {bb_middle} (type: {type(bb_middle)}), Price Above Middle: {price_above_middle} (type: {type(price_above_middle)})")
            
            # Ensure that all conditions are scalar booleans
            if not isinstance(strong_trend, bool):
                self.logger.warning(f"strong_trend is not a boolean. Type: {type(strong_trend)}. Skipping signal generation.")
                return signal
            if not isinstance(bullish_trend, bool):
                self.logger.warning(f"bullish_trend is not a boolean. Type: {type(bullish_trend)}. Skipping signal generation.")
                return signal
            if not isinstance(macd_increasing, bool):
                self.logger.warning(f"macd_increasing is not a boolean. Type: {type(macd_increasing)}. Skipping signal generation.")
                return signal
            if not isinstance(price_above_middle, bool):
                self.logger.warning(f"price_above_middle is not a boolean. Type: {type(price_above_middle)}. Skipping signal generation.")
                return signal

            if strong_trend:
                if rsi < 30.0 and bullish_trend and macd_increasing:
                    signal = {
                        'strategy': 'BULL_CALL_SPREAD',
                        'confidence': 0.8,
                        'expiration_days': 45
                    }
                elif rsi > 70.0 and not bullish_trend and not macd_increasing:
                    signal = {
                        'strategy': 'BEAR_PUT_SPREAD',
                        'confidence': 0.8,
                        'expiration_days': 45
                    }
                elif bullish_trend and price_above_middle:
                    signal = {
                        'strategy': 'LONG_CALL',
                        'confidence': 0.7,
                        'expiration_days': 30
                    }
                elif not bullish_trend and not price_above_middle:
                    signal = {
                        'strategy': 'LONG_PUT',
                        'confidence': 0.7,
                        'expiration_days': 30
                    }
            
            self.logger.debug(f"Generated Signal: {signal}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}", exc_info=True)
            return {'strategy': 'NO_TRADE', 'confidence': 0.0}

    def calculate_position_size(self, option_price: float, current_price: float) -> Tuple[int, float]:
        """Calculate optimal position size based on risk parameters."""
        try:
            max_capital = min(
                self.current_capital * self.max_position_size,
                self.current_capital * self.max_symbol_exposure
            )
            contract_cost = option_price * 100
            if contract_cost <= 0:
                return 0, 0.0
            max_contracts = int(max_capital / contract_cost)
            
            if contract_cost * max_contracts < self.min_trade_size:
                return 0, 0.0
                
            num_contracts = min(max_contracts, 10)
            total_cost = contract_cost * num_contracts
            return num_contracts, total_cost
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0, 0.0

    def _apply_trading_costs(self, premium: float) -> float:
        commission = self.commission_rate
        slippage_cost = premium * self.slippage
        return premium + commission + slippage_cost

    def _find_closest_option(self, chain: pd.DataFrame, target_strike: float) -> Optional[Dict[str, Any]]:
        """Find the option contract closest to target strike price."""
        try:
            chain['strike_diff'] = abs(chain['strike'] - target_strike)
            closest = chain.nsmallest(1, 'strike_diff').iloc[0]
            mid_price = (closest['bid'] + closest['ask']) / 2
            return {
                'strike': float(closest['strike']),
                'premium': float(mid_price),
                'bid': float(closest['bid']),
                'ask': float(closest['ask']),
                'volume': float(closest.get('volume', 0)),
                'open_interest': float(closest.get('openInterest', 0))
            }
        except Exception as e:
            self.logger.error(f"Error finding closest option: {e}")
            return None
        
    def _setup_bull_call_spread(self, chain, entry_price: float, symbol: str,
                                entry_date: pd.Timestamp, expiration_date: str,
                                strategy_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            buy_strike = entry_price * 0.98
            sell_strike = entry_price * 1.05
            buy_leg = self._find_closest_option(chain.calls, buy_strike)
            sell_leg = self._find_closest_option(chain.calls, sell_strike)
            
            if not buy_leg or not sell_leg:
                return None
                
            buy_cost = self._apply_trading_costs(buy_leg['premium'])
            sell_credit = self._apply_trading_costs(sell_leg['premium'])
            net_debit = buy_cost - sell_credit
            
            num_contracts, total_cost = self.calculate_position_size(net_debit, entry_price)
            if num_contracts == 0:
                return None
                
            return {
                'symbol': symbol,
                'strategy': 'BULL_CALL_SPREAD',
                'entry_date': entry_date,
                'expiration_date': expiration_date,
                'leg1_strike': float(buy_leg['strike']),
                'leg2_strike': float(sell_leg['strike']),
                'leg1_premium': float(buy_leg['premium']),
                'leg2_premium': float(sell_leg['premium']),
                'net_debit': float(net_debit),
                'num_contracts': num_contracts,
                'total_cost': float(total_cost),
                'confidence': strategy_info.get('confidence', 0.0),
                'entry_price': entry_price,
                'max_loss': float(total_cost),
                'max_profit': float((sell_leg['strike'] - buy_leg['strike']) * 
                                    num_contracts * 100 - total_cost)
            }
        except Exception as e:
            self.logger.error(f"Error setting up bull call spread: {e}")
            return None

    def _setup_bear_put_spread(self, chain, entry_price: float, symbol: str,
                               entry_date: pd.Timestamp, expiration_date: str,
                               strategy_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            buy_strike = entry_price * 1.02
            sell_strike = entry_price * 0.95
            buy_leg = self._find_closest_option(chain.puts, buy_strike)
            sell_leg = self._find_closest_option(chain.puts, sell_strike)
            
            if not buy_leg or not sell_leg:
                return None
                
            buy_cost = self._apply_trading_costs(buy_leg['premium'])
            sell_credit = self._apply_trading_costs(sell_leg['premium'])
            net_debit = buy_cost - sell_credit
            
            num_contracts, total_cost = self.calculate_position_size(net_debit, entry_price)
            if num_contracts == 0:
                return None

            return {
                'symbol': symbol,
                'strategy': 'BEAR_PUT_SPREAD',
                'entry_date': entry_date,
                'expiration_date': expiration_date,
                'leg1_strike': float(buy_leg['strike']),
                'leg2_strike': float(sell_leg['strike']),
                'leg1_premium': float(buy_leg['premium']),
                'leg2_premium': float(sell_leg['premium']),
                'net_debit': float(net_debit),
                'num_contracts': num_contracts,
                'total_cost': float(total_cost),
                'confidence': strategy_info.get('confidence', 0.0),
                'entry_price': entry_price,
                'max_loss': float(total_cost),
                'max_profit': float((buy_leg['strike'] - sell_leg['strike']) * 
                                    num_contracts * 100 - total_cost)
            }
        except Exception as e:
            self.logger.error(f"Error setting up bear put spread: {e}")
            return None

    def _setup_long_call(self, chain, entry_price: float, symbol: str,
                         entry_date: pd.Timestamp, expiration_date: str,
                         strategy_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            target_strike = entry_price * 1.02
            option = self._find_closest_option(chain.calls, target_strike)
            
            if not option:
                return None
                
            premium = self._apply_trading_costs(option['premium'])
            num_contracts, total_cost = self.calculate_position_size(premium, entry_price)
            if num_contracts == 0:
                return None
                
            return {
                'symbol': symbol,
                'strategy': 'LONG_CALL',
                'entry_date': entry_date,
                'expiration_date': expiration_date,
                'strike': float(option['strike']),
                'premium': float(premium),
                'num_contracts': num_contracts,
                'total_cost': float(total_cost),
                'confidence': strategy_info.get('confidence', 0.0),
                'entry_price': entry_price,
                'max_loss': float(total_cost),
                'max_profit': float('inf')
            }
        except Exception as e:
            self.logger.error(f"Error setting up long call: {e}")
            return None

    def _setup_long_put(self, chain, entry_price: float, symbol: str,
                        entry_date: pd.Timestamp, expiration_date: str,
                        strategy_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            target_strike = entry_price * 0.98
            option = self._find_closest_option(chain.puts, target_strike)
            
            if not option:
                return None
                
            premium = self._apply_trading_costs(option['premium'])
            num_contracts, total_cost = self.calculate_position_size(premium, entry_price)
            if num_contracts == 0:
                return None
                
            return {
                'symbol': symbol,
                'strategy': 'LONG_PUT',
                'entry_date': entry_date,
                'expiration_date': expiration_date,
                'strike': float(option['strike']),
                'premium': float(premium),
                'num_contracts': num_contracts,
                'total_cost': float(total_cost),
                'confidence': strategy_info.get('confidence', 0.0),
                'entry_price': entry_price,
                'max_loss': float(total_cost),
                'max_profit': float(option['strike'] * num_contracts * 100)
            }
        except Exception as e:
            self.logger.error(f"Error setting up long put: {e}")
            return None
        
    def _simulate_trade(self, symbol: str, entry_date: pd.Timestamp, 
                       strategy_info: Dict[str, Any], entry_price: float) -> Optional[Dict[str, Any]]:
        try:
            if len(self.open_positions) >= self.max_positions:
                self.logger.info(f"Maximum positions ({self.max_positions}) reached, skipping trade")
                return None

            expiration_days = strategy_info.get('expiration_days', 30)
            expiration_date = (entry_date + pd.Timedelta(days=expiration_days)).strftime('%Y-%m-%d')

            ticker = yf.Ticker(symbol)
            chain = ticker.option_chain(expiration_date)
            if not chain:
                return None

            strategy_type = strategy_info['strategy']
            trade_info = None

            if strategy_type == 'BULL_CALL_SPREAD':
                trade_info = self._setup_bull_call_spread(chain, entry_price, symbol, entry_date, expiration_date, strategy_info)
            elif strategy_type == 'BEAR_PUT_SPREAD':
                trade_info = self._setup_bear_put_spread(chain, entry_price, symbol, entry_date, expiration_date, strategy_info)
            elif strategy_type == 'LONG_CALL':
                trade_info = self._setup_long_call(chain, entry_price, symbol, entry_date, expiration_date, strategy_info)
            elif strategy_type == 'LONG_PUT':
                trade_info = self._setup_long_put(chain, entry_price, symbol, entry_date, expiration_date, strategy_info)

            if trade_info:
                self.current_capital -= trade_info['total_cost']
                self.open_positions.append(trade_info)
                return trade_info

            return None

        except Exception as e:
            self.logger.error(f"Error simulating trade: {e}", exc_info=True)
            return None
    
    def _get_option_value(self, options_df: pd.DataFrame, strike: float) -> float:
        try:
            option = options_df.iloc[(options_df['strike'] - strike).abs().argsort()[:1]]
            if not option.empty:
                mid_price = (option['bid'].iloc[0] + option['ask'].iloc[0]) / 2
                return float(mid_price)
        except Exception as e:
            self.logger.error(f"Error getting option value: {e}")
        return 0.0

    def _get_underlying_close(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        try:
            df = self.historical_data.get(symbol)
            if df is None or df.empty:
                return None
            if date in df.index:
                return float(df.loc[date, 'Close'])
            else:
                return float(df.iloc[-1]['Close'])
        except Exception as e:
            self.logger.error(f"Error getting close price: {e}")
            return None

    def _close_trade(self, open_position: Dict[str, Any], exit_date: pd.Timestamp) -> Optional[Dict[str, Any]]:
        try:
            symbol = open_position['symbol']
            expiration_str = open_position['expiration_date']
            expiry_dt = datetime.strptime(expiration_str, '%Y-%m-%d')
            
            final_value = 0.0
            final_price = self._get_underlying_close(symbol, exit_date)
            if not final_price:
                return None

            strategy_type = open_position['strategy']
            
            if exit_date >= pd.Timestamp(expiry_dt):
                # At expiration
                if strategy_type == 'BULL_CALL_SPREAD':
                    spread_width = open_position['leg2_strike'] - open_position['leg1_strike']
                    intrinsic = max(0.0, min(spread_width, final_price - open_position['leg1_strike']))
                    final_value = float(intrinsic)
                
                elif strategy_type == 'BEAR_PUT_SPREAD':
                    spread_width = open_position['leg1_strike'] - open_position['leg2_strike']
                    intrinsic = max(0.0, min(spread_width, open_position['leg1_strike'] - final_price))
                    final_value = float(intrinsic)
                
                elif strategy_type == 'LONG_CALL':
                    final_value = max(0.0, final_price - open_position['strike'])
                
                elif strategy_type == 'LONG_PUT':
                    final_value = max(0.0, open_position['strike'] - final_price)
            
            else:
                # Before expiration, attempt to get chain data
                try:
                    ticker = yf.Ticker(symbol)
                    chain = ticker.option_chain(expiration_str)
                    if strategy_type in ['BULL_CALL_SPREAD', 'LONG_CALL']:
                        options = chain.calls
                    else:
                        options = chain.puts
                    
                    if strategy_type in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
                        leg1_value = self._get_option_value(options, open_position['leg1_strike'])
                        leg2_value = self._get_option_value(options, open_position['leg2_strike'])
                        final_value = leg1_value - leg2_value
                    else:
                        final_value = self._get_option_value(options, open_position['strike'])
                
                except Exception as e:
                    self.logger.error(f"Error getting current option values: {e}")
                    return None

            num_contracts = open_position['num_contracts']
            total_value = final_value * num_contracts * 100
            initial_cost = open_position['total_cost']
            
            profit_amount = total_value - initial_cost
            profit_pct = (profit_amount / initial_cost * 100) if initial_cost > 0 else 0
            profit_pct = max(min(profit_pct, 1000.0), -100.0)
            
            return {
                'exit_date': exit_date,
                'exit_price': final_price,
                'option_value': final_value,
                'total_value': total_value,
                'profit_amount': profit_amount,
                'profit_pct': profit_pct
            }
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")
            return None
        
    def _train_optimizer(self, symbol: str, hist_data: pd.DataFrame) -> bool:
        try:
            if len(hist_data) < 100:
                return False

            window_size = 20
            training_data = {}

            for i in range(window_size, len(hist_data), window_size):
                window = hist_data.iloc[i-window_size:i]
                tech_data = self._collect_technical_data(window)
                if tech_data:
                    if i + 20 < len(hist_data):
                        future_return = (hist_data.iloc[i+20]['Close'] -
                                         hist_data.iloc[i]['Close']) / hist_data.iloc[i]['Close'] * 100
                    else:
                        future_return = 0
                        
                    tech_data['profit_pct'] = future_return
                    training_data[f"{symbol}_{i}"] = tech_data

            self.technical_data_collection.update(training_data)
            success = self.strategy_optimizer.train_model(self.technical_data_collection)
            if success:
                self.is_optimizer_trained = True
                self.logger.info(f"Strategy optimizer trained with {len(training_data)} samples")
            return success

        except Exception as e:
            self.logger.error(f"Error training optimizer: {e}")
            return False

    def _backtest_symbol(self, hist_data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            trades = []
            current_trade = None
            window_size = 200  # Ensure this matches the largest technical indicator window

            self._train_optimizer(symbol, hist_data)

            for i in range(window_size, len(hist_data)):
                try:
                    if current_trade is not None:
                        current_price = hist_data.iloc[i]['Close']
                        exit_needed = False
                        exit_reason = ""
                        current_value = self._get_position_value(current_trade, current_price)
                        if current_value is not None:
                            loss_pct = (current_value - current_trade['total_cost']) / current_trade['total_cost'] * 100
                            if loss_pct <= -self.default_stop_loss:
                                exit_needed = True
                                exit_reason = "Stop Loss"
                            elif loss_pct >= self.default_take_profit:
                                exit_needed = True
                                exit_reason = "Take Profit"

                        if hist_data.index[i] >= pd.Timestamp(current_trade['expiration_date']):
                            exit_needed = True
                            exit_reason = "Expiration"

                        if exit_needed:
                            closed_info = self._close_trade(current_trade, hist_data.index[i])
                            if closed_info:
                                trade_result = {**current_trade, **closed_info, 'exit_reason': exit_reason}
                                trades.append(trade_result)
                                self.current_capital += closed_info['total_value']
                            current_trade = None
                        continue

                    if len(self.open_positions) >= self.max_positions:
                        continue

                    window = hist_data.iloc[i-window_size:i]
                    tech_data = self._collect_technical_data(window)
                    if not tech_data:
                        continue

                    signal = self._generate_trading_signal(tech_data, hist_data.iloc[i]['Close'])
                    if signal['strategy'] != 'NO_TRADE':
                        trade = self._simulate_trade(
                            symbol=symbol,
                            entry_date=hist_data.index[i],
                            strategy_info=signal,
                            entry_price=hist_data.iloc[i]['Close']
                        )
                        if trade:
                            current_trade = trade

                except Exception as e:
                    self.logger.error(f"Error in backtest loop for {symbol}: {e}", exc_info=True)
                    continue

            if current_trade:
                closed_info = self._close_trade(current_trade, hist_data.index[-1])
                if closed_info:
                    trade_result = {**current_trade, **closed_info, 'exit_reason': 'End of Backtest'}
                    trades.append(trade_result)

            return self._calculate_backtest_metrics(trades) if trades else None

        except Exception as e:
            self.logger.error(f"Error in backtest for {symbol}: {e}", exc_info=True)
            return None

    def _get_position_value(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        try:
            strategy = position['strategy']
            num_contracts = position['num_contracts']
            if strategy == 'BULL_CALL_SPREAD':
                value = max(0.0, min(
                    position['leg2_strike'] - position['leg1_strike'],
                    current_price - position['leg1_strike']
                ))
            elif strategy == 'BEAR_PUT_SPREAD':
                value = max(0.0, min(
                    position['leg1_strike'] - position['leg2_strike'],
                    position['leg1_strike'] - current_price
                ))
            elif strategy == 'LONG_CALL':
                value = max(0.0, current_price - position['strike'])
            elif strategy == 'LONG_PUT':
                value = max(0.0, position['strike'] - current_price)
            else:
                return None

            return value * num_contracts * 100
        except Exception as e:
            self.logger.error(f"Error calculating position value: {e}")
            return None
        
    def _calculate_backtest_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'sharpe_ratio': 0.0,
                    'profit_factor': 0.0,
                    'trades': []
                }

            trades_df = pd.DataFrame(trades)
            total_trades = len(trades)
            trades_df['profit_pct'] = pd.to_numeric(trades_df['profit_pct'], errors='coerce')
            trades_df['profit_pct'] = trades_df['profit_pct'].replace([np.inf, -np.inf], np.nan)
            winning_trades = len(trades_df[trades_df['profit_pct'] > 0].dropna())
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            clean_profits = trades_df['profit_pct'].dropna()
            if not clean_profits.empty:
                avg_profit = float(clean_profits.mean())
                max_profit = float(clean_profits.max())
                max_loss = float(clean_profits.min())
                std_dev = float(clean_profits.std()) if len(clean_profits) > 1 else 0.0
            else:
                avg_profit = max_profit = max_loss = std_dev = 0.0

            if len(clean_profits) > 1 and std_dev > 0:
                sharpe = float(np.sqrt(252) * (clean_profits.mean() / std_dev))
            else:
                sharpe = 0.0

            gains = clean_profits[clean_profits > 0].sum()
            losses = abs(clean_profits[clean_profits < 0].sum())
            profit_factor = float(gains / losses) if losses != 0 else float('inf')

            strategy_breakdown = trades_df['strategy'].value_counts().to_dict()
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            avg_holding_days = (trades_df['exit_date'] - trades_df['entry_date']).mean().days

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': float(win_rate),
                'avg_profit': float(avg_profit),
                'max_profit': float(max_profit),
                'max_loss': float(max_loss),
                'sharpe_ratio': float(sharpe),
                'profit_factor': float(profit_factor),
                'avg_holding_days': float(avg_holding_days),
                'strategy_breakdown': strategy_breakdown,
                'trades': trades
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return None

    def plot_results(self) -> None:
        if not self.results:
            self.logger.info("No results to plot.")
            return

        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        symbols = []
        profits = []
        for symbol, result in self.results.items():
            if result['total_trades'] > 0:
                symbols.append(symbol)
                profits.append(result['avg_profit'])
        colors = ['g' if p > 0 else 'r' for p in profits]
        ax1.bar(symbols, profits, color=colors)
        ax1.set_title('Average Profit by Symbol')
        ax1.set_ylabel('Profit %')
        for i, symbol in enumerate(symbols):
            trades = self.results[symbol]['total_trades']
            wins = self.results[symbol]['winning_trades']
            ax1.annotate(f'Trades: {trades}\nWins: {wins}', 
                         xy=(i, profits[i]),
                         ha='center', va='bottom')

        ax2 = plt.subplot2grid((3, 2), (0, 1))
        all_trades = []
        for symbol, result in self.results.items():
            for trade in result['trades']:
                all_trades.append({
                    'date': pd.to_datetime(trade['exit_date']),
                    'profit_pct': trade['profit_pct']
                })
        if all_trades:
            trade_df = pd.DataFrame(all_trades).sort_values('date')
            trade_df['cumulative_return'] = (1 + trade_df['profit_pct']/100).cumprod()
            ax2.plot(trade_df['date'], trade_df['cumulative_return'])
            ax2.set_title('Equity Curve')
            ax2.grid(True)

        ax3 = plt.subplot2grid((3, 2), (1, 0))
        strategy_performance = {}
        for result in self.results.values():
            if 'strategy_breakdown' in result:
                for strategy, count in result['strategy_breakdown'].items():
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = {'count': 0, 'profits': []}
                    strategy_performance[strategy]['count'] += count
                    trades = [t for t in result['trades'] if t['strategy'] == strategy]
                    strategy_performance[strategy]['profits'].extend([t['profit_pct'] for t in trades])

        strategies = list(strategy_performance.keys())
        avg_profits = [np.mean(perf['profits']) for perf in strategy_performance.values()]
        counts = [perf['count'] for perf in strategy_performance.values()]
        
        ax3.bar(strategies, avg_profits)
        ax3.set_title('Strategy Performance')
        for i, (profit, count) in enumerate(zip(avg_profits, counts)):
            ax3.annotate(f'N: {count}', xy=(i, profit), ha='center', va='bottom')

        ax4 = plt.subplot2grid((3, 2), (1, 1))
        if all_trades:
            trade_df['YearMonth'] = trade_df['date'].dt.to_period('M')
            monthly_returns = trade_df.groupby('YearMonth')['profit_pct'].sum()
            monthly_returns = monthly_returns.reset_index()
            # Basic bar plot vs. a heatmap for clarity
            ax4.bar(monthly_returns['YearMonth'].astype(str), monthly_returns['profit_pct'])
            ax4.set_title('Monthly Returns')
            ax4.set_xticklabels(monthly_returns['YearMonth'].astype(str), rotation=45)

        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        if all_trades:
            trade_df['profit_pct'].hist(bins=50, ax=ax5)
            ax5.set_title('Profit Distribution')
            ax5.set_xlabel('Profit %')
            ax5.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def print_results(self) -> None:
        """Print detailed backtest results."""
        if not self.results:
            self.logger.info("No results to display.")
            return

        self.logger.info("\n=== Backtest Results ===\n")
        
        for symbol, result in self.results.items():
            self.logger.info(f"\n{symbol} Performance:")
            self.logger.info(f"  Total Trades: {result['total_trades']}")
            self.logger.info(f"  Winning Trades: {result['winning_trades']}")
            self.logger.info(f"  Win Rate: {result.get('win_rate', 0):.2f}%")
            self.logger.info(f"  Average Profit: {result.get('avg_profit', 0):.2f}%")
            self.logger.info(f"  Max Profit: {result.get('max_profit', 0):.2f}%")
            self.logger.info(f"  Max Loss: {result.get('max_loss', 0):.2f}%")
            self.logger.info(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
            self.logger.info(f"  Avg Holding Days: {result.get('avg_holding_days', 0):.1f}")
            
            if 'strategy_breakdown' in result:
                self.logger.info("\n  Strategy Breakdown:")
                for strategy, count in result['strategy_breakdown'].items():
                    self.logger.info(f"    {strategy}: {count} trades")

        all_trades = []
        for result in self.results.values():
            all_trades.extend(result.get('trades', []))

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df = trades_df.sort_values('exit_date')
            cumulative_return = (1 + trades_df['profit_pct']/100).prod() - 1
            days = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days
            years = days / 365.25 if days>0 else 0
            annualized_return = ((1 + cumulative_return) ** (1/years)) - 1 if years>0 else 0

            self.logger.info("\nAggregate Portfolio Statistics:")
            self.logger.info(f"  Total Trades: {total_trades}")
            self.logger.info(f"  Overall Win Rate: {win_rate:.2f}%")
            self.logger.info(f"  Average Trade Profit: {trades_df['profit_pct'].mean():.2f}%")
            self.logger.info(f"  Cumulative Return: {cumulative_return*100:.2f}%")
            self.logger.info(f"  Annualized Return: {annualized_return*100:.2f}%")

            if years>0:
                vol = trades_df['profit_pct'].std() * np.sqrt(252)
                sharpe = (annualized_return / (vol/100)) if vol>0 else 0
                self.logger.info(f"  Portfolio Sharpe Ratio: {sharpe:.2f}")

            avg_position_size = trades_df['total_cost'].mean()
            max_drawdown = self._calculate_max_drawdown(trades_df['profit_pct'])
            self.logger.info(f"  Average Position Size: ${avg_position_size:,.2f}")
            self.logger.info(f"  Maximum Drawdown: {max_drawdown:.2f}%")

            final_capital = self.current_capital
            total_return = ((final_capital - self.initial_capital)/ self.initial_capital * 100)
            self.logger.info("\n---Final Results---")
            self.logger.info(f"  Initial Capital: ${self.initial_capital:,.2f}")
            self.logger.info(f"  Final Capital:   ${final_capital:,.2f}")
            self.logger.info(f"  Total Return: {total_return:.2f}%")

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns/100).cumprod()
        rolling_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        return float(drawdown.min())

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run the complete backtest with all symbols and return results.
        """
        self.logger.info(f"\nStarting backtest from {start_date} to {end_date}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        self.current_capital = self.initial_capital
        self.open_positions = []
        self.results = {}
        self.technical_data_collection = {}

        self.logger.info("\nDownloading historical data...")
        self.download_data(start_date, end_date)

        for symbol in self.watchlist:
            if symbol not in self.historical_data:
                self.logger.warning(f"No data available for {symbol}, skipping.")
                continue

            data = self.historical_data[symbol]

            # Update the data length check to 200
            if len(data) < 200:
                self.logger.warning(f"Insufficient data for {symbol}, requires at least 200 data points, skipping.")
                continue

            self.logger.info(f"\nBacktesting {symbol}...")
            symbol_results = self._backtest_symbol(data, symbol)

            if symbol_results:
                self.results[symbol] = symbol_results
                self.logger.info(f"Completed backtest for {symbol} with {symbol_results['total_trades']} trades")

        self.print_results()
        self.plot_results()

        return self.results

def main():
    logging.basicConfig(level=logging.INFO)
    watchlist = [
        "AAPL","MSFT","GOOGL","META","NVDA","AMD","TSLA","SPY","QQQ"
    ]
    
    print("\n=== Options Strategy Backtester ===")
    print("\nEnter backtest parameters:")
    while True:
        try:
            capital = float(input("\nInitial capital (e.g., 10000 = $10,000): ").strip())
            if capital <=0:
                print("Initial capital must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            start_date = input("Start date (YYYY-MM): ").strip()
            end_date = input("End date (YYYY-MM): ").strip()
            pd.to_datetime(f"{start_date}-01")
            pd.to_datetime(f"{end_date}-01")
            start_date = f"{start_date}-01"
            end_date = f"{end_date}-01"
            break
        except ValueError:
            print("Please enter valid YYYY-MM format.")
    
    try:
        backtester = OptionsBacktester(watchlist, initial_capital=capital)
        results = backtester.run_backtest(start_date, end_date)
    except Exception as e:
        logging.error(f"Error running backtest: {e}", exc_info=True)
        print("An error occurred during the backtest. Please check the logs.")


if __name__=="__main__":
    main()
