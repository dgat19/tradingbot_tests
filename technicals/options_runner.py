import asyncio
import logging
import pytz
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca.data.requests import OptionChainRequest
from alpaca.data.historical import OptionHistoricalDataClient
import yfinance as yf

from technicals import TechnicalIndicators
from market_hours import MarketHours
from strategy_optimizer import StrategyOptimizer


class UnifiedOptionsTrader:
    """
    Live/paper trading class for options using Alpaca’s API for both trade execution and option chain data.
    When a trade signal meets a confidence threshold, the bot will:
      - For a high-confidence (>85%) bullish signal => Bull Call Spread.
      - For a high-confidence (>85%) bearish signal => Bear Put Spread.
      - For moderate confidence (>50%) => Single-leg Call or Put.
      - Otherwise => No trade.
    Risk targets:
      - Minimum profit return of 50%
      - Maximum loss of 20%
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.logger = self._setup_logger()
        self.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA", "SPY", "QQQ"]
        self.market_hours = MarketHours()
        self.logger.info("Trading bot initialized. Waiting for market hours...")
        self.strategy_optimizer = StrategyOptimizer()
        self.options_data_client = OptionHistoricalDataClient(self.api._key_id, self.api._secret_key)

        # Risk and confidence parameters
        self.confidence_threshold_spread = 0.85  # 85% for spreads
        self.confidence_threshold_single = 0.50  # 50% for single-leg trades
        self.position_size_base = 1.0
        self.stop_loss_pct = 0.20         # maximum loss of 20%
        self.take_profit_pct = 0.50       # minimum profit return of 50%

        # Technical indicators calculator
        self.ti = TechnicalIndicators()

        # --- NEW: Initialize tracking for traded contracts and last trading day ---
        self.traded_contracts = set()  # Will store OCC symbols of options traded today
        self.last_trading_day = None   # e.g., "2025-02-03"

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('UnifiedOptionsTrader')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('options_trader.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    def _apply_costs(self, premium: float) -> float:
        commission_rate = 0.001
        slippage = 0.001
        return premium - (premium * (commission_rate + slippage))

    def _construct_option_symbol(self, underlying: str, expiration_str: str, strike: float, option_type: str) -> str:
        """
        Constructs an OCC option symbol (e.g., AAPL250321C00235000).
        
        Args:
            underlying (str): Underlying symbol (e.g., 'AAPL')
            expiration_str (str): Expiration date in YYYY-MM-DD format
            strike (float): Strike price
            option_type (str): Option type ('C' for call or 'P' for put)
        
        Returns:
            str: OCC-formatted option symbol
        """
        try:
            # Parse the expiration date
            expiry_date = datetime.strptime(expiration_str, '%Y-%m-%d')
            
            # Format the year, month, and day
            year_suffix = str(expiry_date.year)[-2:]  # Last 2 digits of year
            month = str(expiry_date.month).zfill(2)   # 2-digit month
            day = str(expiry_date.day).zfill(2)       # 2-digit day
            
            # Format the strike price (multiply by 1000 and round to nearest integer)
            strike_int = int(round(strike * 1000))
            strike_str = str(strike_int).zfill(8)  # Pad with leading zeros to 8 digits
            
            # Construct the OCC symbol
            option_symbol = f"{underlying}{year_suffix}{month}{day}{option_type}{strike_str}"
            
            self.logger.debug(f"Constructed option symbol: {option_symbol} from {underlying}, {expiration_str}, {strike}, {option_type}")
            return option_symbol
            
        except Exception as e:
            self.logger.error(f"Error constructing option symbol: {e}")
            return None


    def _choose_strike_by_delta(self, chain: pd.DataFrame, target_delta: float, underlying_price: float = None) -> pd.Series:
        if 'delta' in chain.columns and chain['delta'].notna().any():
            chain = chain.copy()
            chain.loc[:, 'delta_diff'] = (chain['delta'] - target_delta).abs()
            return chain.loc[chain['delta_diff'].idxmin()]
        elif underlying_price is not None:
            chain = chain.copy()
            if 'strike' not in chain.columns:
                self.logger.error(f"Option chain DataFrame is missing 'strike' column. Columns available: {chain.columns}")
                return None
            chain.loc[:, 'strike_diff'] = (chain['strike'] - underlying_price).abs()
            self.logger.info("Delta data not available; falling back to ATM strike selection.")
            return chain.loc[chain['strike_diff'].idxmin()]
        else:
            self.logger.error("Delta data not available and no underlying price provided for strike selection.")
            return None



    def _parse_option_strike(self, option_symbol: str) -> (float, str):
        """
        Parse the strike price and option type from an option symbol formatted like:
        AAPL250321C00235000
        Assumptions:
          - Underlying symbol: letters at beginning (variable length)
          - Expiration: 6 digits (YYMMDD)
          - Option type: one letter, "C" or "P"
          - Strike: 8 digits (with an implied decimal, divided by 1000)
        Returns a tuple (strike, option_type) as (float, str) or (None, None) on failure.
        """
        pattern = r'^[A-Z]+(\d{6})([CP])(\d{8})$'
        match = re.match(pattern, option_symbol)
        if match:
            try:
                strike_digits = match.group(3)  # e.g., "00235000"
                strike = float(strike_digits) / 1000.0
                option_type = match.group(2)  # "C" or "P"
                return strike, option_type
            except Exception as e:
                self.logger.error(f"Error parsing strike from {option_symbol}: {e}")
                return None, None
        else:
            self.logger.error(f"Option symbol {option_symbol} does not match expected pattern.")
            return None, None

    def _get_alpaca_option_chain(self, symbol: str, max_days: int = 45) -> dict:
        """
        Retrieve an option chain for the given symbol by searching candidate expirations that are <= max_days.
        Tries candidate expirations in descending order (e.g. [45,40,35,...,5]) and returns the first candidate
        that yields any valid options. Returns a dict with keys "calls", "puts", and "expiration_str".
        """
        candidate_days = [45, 40, 35, 30, 25, 20, 15, 10, 5]
        chain_response = None
        chosen_expiration = None

        for days in candidate_days:
            if days > max_days:
                continue
            candidate_date = datetime.now() + timedelta(days=days)
            if candidate_date <= datetime.now():
                continue
            candidate_expiration = candidate_date.strftime('%Y-%m-%d')
            self.logger.info(f"Trying candidate expiration {candidate_expiration} for {symbol}.")
            request_params = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date=candidate_expiration
            )
            chain_response = self.options_data_client.get_option_chain(request_params)
            self.logger.debug(f"Raw chain response for {symbol} at {candidate_expiration}: {chain_response}")
            if chain_response and any(item is not None for item in chain_response):
                chosen_expiration = candidate_expiration
                self.logger.info(f"Found options for {symbol} with expiration {chosen_expiration}.")
                break
            else:
                self.logger.warning(f"No options returned for {symbol} at expiration {candidate_expiration}.")
        if not chain_response or not chosen_expiration:
            self.logger.error(f"No options returned for {symbol} for any expiration within {max_days} days.")
            return None

        calls = []
        puts = []
        for option in chain_response:
            if option is None:
                self.logger.error("Received a None option in chain response; skipping.")
                continue
            # Handle three cases: option is a dict, an object, or a string.
            if isinstance(option, str):
                symbol_val = option
                strike_value, opt_type = self._parse_option_strike(option)
                bid = None
                ask = None
                delta = None
            elif isinstance(option, dict):
                symbol_val = option.get("symbol")
                strike_value = option.get("strike") or option.get("strike_price")
                bid = option.get("bid")
                ask = option.get("ask")
                greeks = option.get("greeks", {})
                delta = greeks.get("delta", 0.0) if greeks is not None else 0.0
                opt_type = option.get("option_type")
            else:
                symbol_val = getattr(option, "symbol", None)
                strike_value = getattr(option, "strike", None) or getattr(option, "strike_price", None)
                bid = getattr(option, "bid", None)
                ask = getattr(option, "ask", None)
                if hasattr(option, "greeks") and option.greeks is not None:
                    delta = getattr(option.greeks, "delta", 0.0)
                else:
                    delta = 0.0
                opt_type = getattr(option, "option_type", None)

            if strike_value is None:
                self.logger.error(f"Option {symbol_val} is missing strike data; skipping this option.")
                continue

            option_data = {
                "symbol": symbol_val,
                "strike": float(strike_value),
                "bid": float(bid) if bid is not None else 0.0,
                "ask": float(ask) if ask is not None else 0.0,
                "delta": float(delta) if delta is not None else 0.0
            }
            # Determine option type (use parsed value if available)
            if opt_type is None and isinstance(option, str):
                # If the option is a string, _parse_option_strike already returned opt_type
                _, opt_type = self._parse_option_strike(option)
            if opt_type and opt_type.lower() == "c":
                calls.append(option_data)
            elif opt_type and opt_type.lower() == "p":
                puts.append(option_data)
            else:
                self.logger.warning(f"Unable to determine option type for {symbol_val}; skipping.")

        calls_df = pd.DataFrame(calls) if calls else pd.DataFrame()
        puts_df = pd.DataFrame(puts) if puts else pd.DataFrame()

        if calls_df.empty and puts_df.empty:
            self.logger.error(f"No valid options (with strike data) returned for {symbol} at candidate expiration {chosen_expiration}.")
            return None

        if not calls_df.empty:
            calls_df.columns = [col.lower() for col in calls_df.columns]
            calls_df = calls_df.sort_values("strike")
        if not puts_df.empty:
            puts_df.columns = [col.lower() for col in puts_df.columns]
            puts_df = puts_df.sort_values("strike")

        self.logger.debug(f"Processed chain for {symbol}: {len(calls_df)} calls, {len(puts_df)} puts (expiration: {chosen_expiration}).")
        return {"calls": calls_df, "puts": puts_df, "expiration_str": chosen_expiration}

    async def get_technical_indicators(self, symbol: str, timeperiod: int = 14) -> dict:
        try:
            self.logger.debug(f"Downloading market data for {symbol} from yfinance...")
            stock = yf.Ticker(symbol)
            df = stock.history(period="6mo")
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}.")
                return {}
            df.index = df.index.tz_localize(None)
            close_prices = df['Close'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            volume = df['Volume'].values

            indicators = {}
            indicators['RSI'] = self.ti.calculate_rsi(close_prices, timeperiod)
            indicators['MACD'] = self.ti.calculate_macd(pd.Series(close_prices))
            indicators['BB'] = self.ti.calculate_bollinger_bands(pd.Series(close_prices), period=20, num_std=2)
            
            adx_dict = self.ti.calculate_adx(high_prices, low_prices, close_prices, period=timeperiod)
            indicators['ADX'] = adx_dict['adx']
            indicators['DI+'] = adx_dict['di_plus']
            indicators['DI-'] = adx_dict['di_minus']
            
            # Compute smoothed Heikin-Ashi candles and take the last candle.
            ha_df = self.ti.calculate_heikin_ashi(df)
            ha_candle = ha_df.iloc[-1].to_dict()
            indicators['HA'] = ha_candle

            indicators['volume'] = float(volume[-1])
            indicators['avg_volume'] = float(np.mean(volume[-20:])) if len(volume) >= 20 else 0
            indicators['current_price'] = float(close_prices[-1])
            
            self.logger.debug(f"Indicators for {symbol}: {indicators}")
            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}", exc_info=True)
            return {}


    async def execute_trade(self, symbol: str, strategy: dict) -> None:
        self.logger.info(f"Starting trade execution for {symbol} with strategy {strategy['strategy']}")
        try:
            expiration_days = strategy['strike_params']['expiration_days']
            chain_data = self._get_alpaca_option_chain(symbol, max_days=expiration_days)
            if chain_data is None:
                self.logger.error(f"Option chain not available for {symbol}, skipping trade.")
                return

            expiration_str = chain_data.get("expiration_str", "")
            # Get account buying power and allocate 20%
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            allocation = buying_power * 0.20
            contract_multiplier = 100

            underlying_price = strategy.get("underlying_price", None)

            if strategy['strategy'] == 'LONG_CALL':
                target_delta = strategy['strike_params']['delta_target']
                chosen_option = self._choose_strike_by_delta(chain_data['calls'], target_delta, underlying_price)
                if chosen_option is None:
                    self.logger.error(f"Could not select option strike for single call on {symbol}")
                    return
                mid_price = (chosen_option['bid'] + chosen_option['ask']) / 2.0
                net_debit = self._apply_costs(mid_price)
                net_debit = abs(net_debit) if net_debit < 0 else net_debit
                cost_per_contract = net_debit * contract_multiplier
                num_contracts = int(allocation // cost_per_contract)
                if num_contracts < 1:
                    self.logger.info(f"Insufficient buying power for {symbol}. Allocation: {allocation}, cost per contract: {cost_per_contract}")
                    return
                qty = num_contracts
                option_symbol = self._construct_option_symbol(symbol, expiration_str, chosen_option['strike'], 'C')
                self.logger.info(f"[LONG_CALL] BUY {option_symbol} (Qty={qty}) at net debit ~{net_debit:.2f}")
                order_buy = self.api.submit_order(
                    symbol=option_symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                self.logger.info(f"Call buy order placed for {symbol}: {order_buy}")

            elif strategy['strategy'] == 'LONG_PUT':
                target_delta = strategy['strike_params']['delta_target']
                chosen_option = self._choose_strike_by_delta(chain_data['puts'], target_delta, underlying_price)
                if chosen_option is None:
                    self.logger.error(f"Could not select option strike for single put on {symbol}")
                    return
                mid_price = (chosen_option['bid'] + chosen_option['ask']) / 2.0
                net_debit = self._apply_costs(mid_price)
                net_debit = abs(net_debit) if net_debit < 0 else net_debit
                cost_per_contract = net_debit * contract_multiplier
                num_contracts = int(allocation // cost_per_contract)
                if num_contracts < 1:
                    self.logger.info(f"Insufficient buying power for {symbol}. Allocation: {allocation}, cost per contract: {cost_per_contract}")
                    return
                qty = num_contracts
                option_symbol = self._construct_option_symbol(symbol, expiration_str, chosen_option['strike'], 'P')
                self.logger.info(f"[LONG_PUT] BUY {option_symbol} (Qty={qty}) at net debit ~{net_debit:.2f}")
                order_buy = self.api.submit_order(
                    symbol=option_symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                self.logger.info(f"Put buy order placed for {symbol}: {order_buy}")

            elif strategy['strategy'] == 'BULL_CALL_SPREAD':
                lower_target = strategy['strike_params']['lower_strike_delta']
                upper_target = strategy['strike_params']['upper_strike_delta']
                lower_option = self._choose_strike_by_delta(chain_data['calls'], lower_target, underlying_price)
                upper_option = self._choose_strike_by_delta(chain_data['calls'], upper_target, underlying_price)
                if lower_option is None or upper_option is None:
                    self.logger.error(f"Could not select option strikes for bull call spread on {symbol}")
                    return
                # Ensure lower strike is strictly less than upper strike.
                if lower_option['strike'] >= upper_option['strike']:
                    self.logger.warning(f"Lower and upper strikes are equal ({lower_option['strike']}); attempting to adjust upper strike.")
                    candidate_upper = chain_data['calls'][chain_data['calls']['strike'] > lower_option['strike']]
                    if not candidate_upper.empty:
                        upper_option = candidate_upper.iloc[0]
                        self.logger.info(f"Adjusted upper option strike to {upper_option['strike']}.")
                    else:
                        self.logger.error(f"Invalid strike selection for bull call spread: lower strike ({lower_option['strike']}) >= upper strike ({upper_option['strike']})")
                        return
                        
                lower_mid = (lower_option['bid'] + lower_option['ask']) / 2.0
                upper_mid = (upper_option['bid'] + upper_option['ask']) / 2.0
                net_debit = self._apply_costs(lower_mid - upper_mid)
                net_debit = abs(net_debit) if net_debit < 0 else net_debit

                # Check for a zero (or near-zero) net debit to avoid division by zero.
                if net_debit < 0.001:
                    self.logger.error("Net debit for spread trade is zero or nearly zero; skipping trade.")
                    return

                contract_multiplier = 100
                cost_per_contract = net_debit * contract_multiplier
                if cost_per_contract < 0.001:
                    self.logger.error("Cost per contract is zero; cannot determine quantity. Skipping trade.")
                    return
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                allocation = buying_power * 0.20
                num_contracts = int(allocation // cost_per_contract)
                if num_contracts < 1:
                    self.logger.info(f"Insufficient buying power for spread trade on {symbol}. Allocation: {allocation}, cost per contract: {cost_per_contract}")
                    return
                qty = num_contracts

                lower_symbol = self._construct_option_symbol(symbol, expiration_str, lower_option['strike'], 'C')
                upper_symbol = self._construct_option_symbol(symbol, expiration_str, upper_option['strike'], 'C')
                self.logger.info(f"[BULL_CALL_SPREAD] BUY {lower_symbol}, SELL {upper_symbol} (Qty={qty}) net debit ~{net_debit:.2f}")
                
                # Submit separate orders for each leg
                order_buy = self.api.submit_order(
                    symbol=lower_symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                order_sell = self.api.submit_order(
                    symbol=upper_symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                self.logger.info(f"Spread orders placed: BUY {lower_symbol}: {order_buy}, SELL {upper_symbol}: {order_sell}")


            elif strategy['strategy'] == 'BEAR_PUT_SPREAD':
                lower_target = strategy['strike_params']['lower_strike_delta']
                upper_target = strategy['strike_params']['upper_strike_delta']
                lower_option = self._choose_strike_by_delta(chain_data['puts'], lower_target, underlying_price)
                upper_option = self._choose_strike_by_delta(chain_data['puts'], upper_target, underlying_price)
                if lower_option is None or upper_option is None:
                    self.logger.error(f"Could not select option strikes for bear put spread on {symbol}")
                    return
                if upper_option['strike'] >= lower_option['strike']:
                    self.logger.warning("Upper and lower strikes are not in proper order; attempting to adjust.")
                    candidate_lower = chain_data['puts'][chain_data['puts']['strike'] < upper_option['strike']]
                    if not candidate_lower.empty:
                        lower_option = candidate_lower.iloc[-1]
                        self.logger.info(f"Adjusted lower option strike to {lower_option['strike']}.")
                    else:
                        self.logger.error(f"Invalid strike selection for bear put spread: upper strike ({upper_option['strike']}) >= lower strike ({lower_option['strike']})")
                        return
                lower_mid = (lower_option['bid'] + lower_option['ask']) / 2.0
                upper_mid = (upper_option['bid'] + upper_option['ask']) / 2.0
                net_debit = self._apply_costs(upper_mid - lower_mid)
                net_debit = abs(net_debit) if net_debit < 0 else net_debit
                cost_per_contract = net_debit * contract_multiplier
                num_contracts = int(allocation // cost_per_contract)
                if num_contracts < 1:
                    self.logger.info(f"Insufficient buying power for spread trade on {symbol}. Allocation: {allocation}, cost per contract: {cost_per_contract}")
                    return
                qty = num_contracts
                lower_symbol = self._construct_option_symbol(symbol, expiration_str, lower_option['strike'], 'P')
                upper_symbol = self._construct_option_symbol(symbol, expiration_str, upper_option['strike'], 'P')
                self.logger.info(f"[BEAR_PUT_SPREAD] BUY {lower_symbol}, SELL {upper_symbol} (Qty={qty}) net debit ~{net_debit:.2f}")
                order_buy = self.api.submit_order(
                    symbol=lower_symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                order_sell = self.api.submit_order(
                    symbol=upper_symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                self.logger.info(f"Spread orders placed: BUY {lower_symbol}: {order_buy}, SELL {upper_symbol}: {order_sell}")

            else:
                self.logger.info(f"No valid strategy found for {symbol}. Nothing to execute.")
                return

            # Record the contract (or spread pair) as traded for today.
            self.traded_contracts.add(option_symbol)
            self.logger.info(f"Recorded {option_symbol} as traded for today.")
            self.logger.info(f"Trade execution completed for {symbol} with strategy {strategy['strategy']}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}", exc_info=True)


    async def _decide_strategy(self, symbol: str, technical_data: dict) -> dict:
        self.logger.info(f"Deciding strategy for {symbol} based on technical data: {technical_data}")
        if not technical_data:
            self.logger.info(f"No technical data for {symbol}. Returning NO_TRADE.")
            return {'strategy': 'NO_TRADE', 'confidence': 0.0}

        rsi = technical_data.get('RSI', 50)
        current_volume = technical_data.get('volume', 0)
        avg_volume = technical_data.get('avg_volume', 1)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        call_put_ratio = await self.get_call_put_ratio(symbol)

        score_bullish = 0.0
        score_bearish = 0.0

        # RSI: bullish if < 40, bearish if > 60
        if rsi < 30:
            score_bullish += 0.4
        elif rsi > 70:
            score_bearish += 0.4

        # Volume momentum: if volume ratio > 1.5, add weight
        if volume_ratio > 1.5:
            score_bullish += 0.3
            score_bearish += 0.3

        # Call-to-put ratio: >1.1 bullish, <0.9 bearish
        if call_put_ratio > 1.1:
            score_bullish += 0.3
        elif call_put_ratio < 0.9:
            score_bearish += 0.3

        # MACD signals: Histogram and crossover
        macd_data = technical_data.get("MACD", {})
        macd_line = macd_data.get("macd", 0)
        signal_line = macd_data.get("signal", 0)
        hist = macd_data.get("hist", 0)
        if hist < 0:
            score_bearish += 0.3
        elif hist > 0:
            score_bullish += 0.3
        if macd_line < signal_line:
            score_bearish += 0.3
        elif macd_line > signal_line:
            score_bullish += 0.3

        # ADX and DI indicators: if trend is strong (ADX > 25), add weight based on DI comparison.
        adx = technical_data.get("ADX", 0)
        di_plus = technical_data.get("DI+")
        di_minus = technical_data.get("DI-")
        if adx > 25:
            if di_plus is not None and di_minus is not None:
                if di_plus > di_minus:
                    score_bullish += 0.2
                elif di_minus > di_plus:
                    score_bearish += 0.2

        # Smoothed Heikin-Ashi candle: if HA_Close > HA_Open, bullish; else bearish.
        ha = technical_data.get("HA", {})
        if ha:
            ha_open = ha.get("HA_Open", 0)
            ha_close = ha.get("HA_Close", 0)
            if ha_close > ha_open:
                score_bullish += 0.1
            else:
                score_bearish += 0.1

        self.logger.info(f"{symbol} Indicators => RSI: {rsi}, Volume Ratio: {volume_ratio:.2f}, Call-Put Ratio: {call_put_ratio:.2f}")
        self.logger.info(f"{symbol} MACD => macd: {macd_line:.2f}, signal: {signal_line:.2f}, hist: {hist:.2f}")
        self.logger.info(f"{symbol} ADX: {adx:.2f}, DI+: {di_plus}, DI-: {di_minus}")
        self.logger.info(f"{symbol} Scores => Bullish: {score_bullish:.2f}, Bearish: {score_bearish:.2f}")

        # Compute final confidence as the higher of bullish or bearish score.
        final_confidence = score_bullish if score_bullish > score_bearish else score_bearish
        if final_confidence < 0.85:
            self.logger.info(f"Chance for profit below 85% (final confidence {final_confidence:.2f}). Returning NO_TRADE.")
            return {'strategy': 'NO_TRADE', 'confidence': final_confidence}

        underlying_price = technical_data.get("current_price", None)

        if score_bullish > score_bearish:
            # Choose spread if high confidence; otherwise, single leg.
            if score_bullish >= self.confidence_threshold_spread:
                return {
                    'strategy': 'BULL_CALL_SPREAD',
                    'confidence': score_bullish,
                    'strike_params': {
                        'lower_strike_delta': 0.3,
                        'upper_strike_delta': 0.15,
                        'expiration_days': 45
                    },
                    'underlying_price': underlying_price
                }
            elif score_bullish >= self.confidence_threshold_single:
                return {
                    'strategy': 'LONG_CALL',
                    'confidence': score_bullish,
                    'strike_params': {
                        'delta_target': 0.3,
                        'expiration_days': 30
                    },
                    'underlying_price': underlying_price
                }
        else:
            if score_bearish >= self.confidence_threshold_spread:
                return {
                    'strategy': 'BEAR_PUT_SPREAD',
                    'confidence': score_bearish,
                    'strike_params': {
                        'lower_strike_delta': -0.15,
                        'upper_strike_delta': -0.3,
                        'expiration_days': 45
                    },
                    'underlying_price': underlying_price
                }
            elif score_bearish >= self.confidence_threshold_single:
                return {
                    'strategy': 'LONG_PUT',
                    'confidence': score_bearish,
                    'strike_params': {
                        'delta_target': -0.3,
                        'expiration_days': 30
                    },
                    'underlying_price': underlying_price
                }
        return {'strategy': 'NO_TRADE', 'confidence': final_confidence}


    async def get_call_put_ratio(self, symbol: str, expiration_days: int = 45) -> float:
        self.logger.info(f"Calculating call-to-put ratio for {symbol} (target expiration ~{expiration_days} days)...")
        chain_data = self._get_alpaca_option_chain(symbol, max_days=expiration_days)
        if not chain_data:
            self.logger.warning(f"No chain data for {symbol}; returning neutral ratio of 1.0")
            return 1.0
        calls_df = chain_data['calls']
        puts_df = chain_data['puts']
        if not calls_df.empty and not puts_df.empty:
            if 'bid' in calls_df.columns and 'bid' in puts_df.columns:
                calls_volume = calls_df['bid'].sum()
                puts_volume = puts_df['bid'].sum()
            else:
                calls_volume = calls_df.shape[0]
                puts_volume = puts_df.shape[0]
        else:
            self.logger.info(f"Either calls or puts are empty for {symbol}; returning 1.0")
            return 1.0
        ratio = calls_volume / puts_volume if puts_volume != 0 else 1.0
        self.logger.info(f"Call-to-put ratio for {symbol} = {ratio:.2f}")
        return ratio

    async def manage_positions(self):
        self.logger.info("Entering manage_positions()...")
        try:
            positions = self.api.list_positions()
            if not positions:
                self.logger.info("No open positions to manage.")
                return
            self.logger.info(f"Found {len(positions)} open position(s) to evaluate.")
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                unreal_pl = float(pos.unrealized_pl)
                cost_basis = entry_price * qty if qty != 0 else 1.0
                pnl_ratio = unreal_pl / cost_basis
                self.logger.info(f"Position: {symbol}, qty={qty}, entry={entry_price}, current={current_price}, unreal_pl={unreal_pl}, pnl_ratio={pnl_ratio:.2f}")
                if pnl_ratio <= -self.stop_loss_pct:
                    self.logger.info(f"Stop loss triggered for {symbol} (pnl_ratio={pnl_ratio:.2f}). Closing...")
                    try:
                        response = self.api.close_position(symbol)
                        self.logger.info(f"Closed position {symbol}: {response}")
                    except Exception as e:
                        self.logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)
                elif pnl_ratio >= self.take_profit_pct:
                    self.logger.info(f"Take profit triggered for {symbol} (pnl_ratio={pnl_ratio:.2f}). Closing...")
                    try:
                        response = self.api.close_position(symbol)
                        self.logger.info(f"Closed position {symbol}: {response}")
                    except Exception as e:
                        self.logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in manage_positions: {e}", exc_info=True)
        finally:
            self.logger.info("Exiting manage_positions()...")

    async def run_trading_bot(self):
        eastern = pytz.timezone("US/Eastern")
        self.logger.info("Starting the main trading loop...")
        while True:
            try:
                now = datetime.now(eastern)
                # --- NEW: Reset traded contracts if a new day has started ---
                current_day = now.strftime("%Y-%m-%d")
                if self.last_trading_day != current_day:
                    self.traded_contracts = set()
                    self.last_trading_day = current_day
                    self.logger.info(f"New trading day detected. Reset traded contracts for {current_day}.")

                market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                if now < market_open:
                    wait_secs = (market_open - now).total_seconds()
                    self.logger.info(f"Current time {now.strftime('%H:%M:%S')} ET is before market open. Sleeping for {wait_secs:.0f} seconds until 09:30 ET.")
                    await asyncio.sleep(wait_secs)
                    continue
                if now >= market_close:
                    next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
                    wait_secs = (next_open - now).total_seconds()
                    self.logger.info(f"Current time {now.strftime('%H:%M:%S')} ET is after market close. Sleeping for {wait_secs:.0f} seconds until next trading day 09:30 ET.")
                    await asyncio.sleep(wait_secs)
                    continue
                self.logger.info(f"Market is OPEN (current ET time: {now.strftime('%H:%M:%S')}). Beginning trading cycle...")
                self.logger.info("Managing open positions (stop-loss/take-profit)...")
                await self.manage_positions()
                self.logger.info("Position management complete.")
                self.logger.info("Analyzing watchlist for trade opportunities...")
                for symbol in self.watchlist:
                    now = datetime.now(eastern)
                    if now >= market_close:
                        self.logger.info("Market closed during symbol processing. Ending cycle.")
                        break
                    self.logger.info(f"Processing symbol: {symbol}")
                    technical_data = await self.get_technical_indicators(symbol)
                    self.logger.info(f"{symbol} technical indicators: {technical_data}")
                    strategy = await self._decide_strategy(symbol, technical_data)
                    self.logger.info(f"{symbol} strategy decision: {strategy}")
                    if strategy['strategy'] != 'NO_TRADE':
                        self.logger.info(f"Executing trade for {symbol}: {strategy['strategy']} with confidence {strategy['confidence']:.2f}")
                        await self.execute_trade(symbol, strategy)
                    else:
                        self.logger.info(f"No valid trade signal for {symbol} at this time.")
                    await asyncio.sleep(5)
                now = datetime.now(eastern)
                time_to_close = (market_close - now).total_seconds()
                self.logger.info(f"Time remaining until market close: {time_to_close:.0f} seconds.")
                if time_to_close < 60:
                    self.logger.info("Less than 60 seconds remain. Sleeping until market close.")
                    await asyncio.sleep(time_to_close)
                else:
                    sleep_interval = min(60, time_to_close - 60)
                    self.logger.info(f"Sleeping for {sleep_interval:.0f} seconds before next cycle.")
                    await asyncio.sleep(sleep_interval)
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}", exc_info=True)
                self.logger.info("Sleeping for 60 seconds before retrying...")
                await asyncio.sleep(60)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    API_KEY = "PKJCI6OURNAXWQHA16OS"
    API_SECRET = "EXqiYQD6uK9Yw736FsCUbE3ne96A5aFAc8mt2GDO"
    if not API_KEY or not API_SECRET:
        sys.exit("Alpaca API credentials not provided.")
    trader = UnifiedOptionsTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        base_url="https://paper-api.alpaca.markets"
    )
    asyncio.run(trader.run_trading_bot())