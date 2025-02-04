import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime

from technicals import TechnicalIndicators
from strategy_optimizer import StrategyOptimizer

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsBacktester:
    """
    A class for running historical backtests on an options-trading strategy using more realistic option pricing.
    This version attempts to fetch option chain data from yfinance, then picks strikes and computes PnL
    based on actual option premiums. Note yfinance's historical data is limited in practice.
    """

    def __init__(self, watchlist: List[str]):
        # Strategy parameters
        self.watchlist = watchlist
        self.ti = TechnicalIndicators()

        self.confidence_threshold = 0.7
        self.position_size = 0.02
        self.stop_loss = 0.15
        self.take_profit = 0.25

        # Transaction cost settings (example: 0.2% cost overall)
        self.commission_rate = 0.001  # 0.1%
        self.slippage = 0.001        # 0.1%

        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def download_data(self, start_date: str, end_date: str) -> None:
        """
        Download and store historical equity data for all symbols in the watchlist.
        Note: For actual daily option data, you need a specialized vendor.
        """
        logger.info("Downloading equity data for symbols...")
        for symbol in self.watchlist:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            data = data.copy()
            data.index = data.index.tz_localize(None)
            self.historical_data[symbol] = data
            if not data.empty:
                self.historical_data[symbol] = data
                logger.info(f"Retrieved {len(data)} rows for {symbol}.")
            else:
                logger.warning(f"No data found for {symbol}. Skipping.")
        logger.info("Data download complete.")

    def get_technical_indicators(self, hist_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate technical indicators from historical data.
        """
        try:
            if len(hist_data) < 20:
                return None

            close_prices = hist_data['Close'].values
            high_prices = hist_data['High'].values
            low_prices = hist_data['Low'].values
            volume = hist_data['Volume'].values

            indicators = {}

            # RSI
            rsi = self.ti.calculate_rsi(close_prices)
            rsi = rsi.iloc[-1] if isinstance(rsi, pd.Series) else rsi
            indicators['RSI'] = float(rsi)

            # MACD
            macd_data = self.ti.calculate_macd(pd.Series(close_prices))
            macd_val = macd_data['macd'].iloc[-1] if isinstance(macd_data['macd'], pd.Series) else macd_data['macd']
            macd_signal = macd_data['signal'].iloc[-1] if isinstance(macd_data['signal'], pd.Series) else macd_data['signal']
            macd_hist = macd_data['hist'].iloc[-1] if isinstance(macd_data['hist'], pd.Series) else macd_data['hist']
            indicators['MACD'] = {
                'macd': float(macd_val),
                'signal': float(macd_signal),
                'hist': float(macd_hist)
            }

            # Bollinger Bands
            bb_data = self.ti.calculate_bollinger_bands(pd.Series(close_prices))
            bb_upper = bb_data['upper'].iloc[-1] if isinstance(bb_data['upper'], pd.Series) else bb_data['upper']
            bb_middle = bb_data['middle'].iloc[-1] if isinstance(bb_data['middle'], pd.Series) else bb_data['middle']
            bb_lower = bb_data['lower'].iloc[-1] if isinstance(bb_data['lower'], pd.Series) else bb_data['lower']
            indicators['BB'] = {
                'upper': float(bb_upper),
                'middle': float(bb_middle),
                'lower': float(bb_lower)
            }

            # ADX
            adx = self.ti.calculate_adx(high_prices, low_prices, close_prices)
            if isinstance(adx, pd.Series):
                adx = adx.iloc[-1]
            indicators['ADX'] = float(adx)

            # SMA
            sma_50 = self.ti.calculate_sma(close_prices, 50)
            sma_200 = self.ti.calculate_sma(close_prices, 200)
            sma_50 = sma_50.iloc[-1] if isinstance(sma_50, pd.Series) else sma_50
            sma_200 = sma_200.iloc[-1] if isinstance(sma_200, pd.Series) else sma_200
            indicators['MA_DATA'] = {
                'SMA_50': float(sma_50),
                'SMA_200': float(sma_200)
            }

            # OBV
            obv = self.ti.calculate_obv(close_prices, volume)
            if isinstance(obv, pd.Series):
                obv = obv.iloc[-1]
            indicators['OBV'] = float(obv)

            # Current price & volume
            indicators['current_price'] = float(close_prices[-1])
            indicators['volume'] = float(volume[-1])
            indicators['avg_volume'] = float(np.mean(volume[-20:]))

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return None

    def determine_options_strategy(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide which spread to use based on indicators. 
        This example only sets high-level deltas and days to expiration.
        """
        try:
            rsi = technical_data['RSI']
            macd = technical_data['MACD']
            adx = technical_data['ADX']
            ma_data = technical_data['MA_DATA']

            bullish_signals = 0
            bearish_signals = 0

            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1

            if macd['macd'] > macd['signal']:
                bullish_signals += 1
            elif macd['macd'] < macd['signal']:
                bearish_signals += 1

            if adx > 25:
                if ma_data['SMA_50'] > ma_data['SMA_200']:
                    bullish_signals += 1
                elif ma_data['SMA_50'] < ma_data['SMA_200']:
                    bearish_signals += 1

            total_signals = bullish_signals + bearish_signals
            confidence = 0.0
            if total_signals > 0:
                confidence = max(bullish_signals, bearish_signals) / total_signals

            if confidence >= self.confidence_threshold:
                if bullish_signals > bearish_signals:
                    return {
                        'strategy': 'BULL_CALL_SPREAD',
                        'confidence': confidence,
                        'expiration_days': 45,
                        # For a call spread, we might want a lower strike delta ~0.30, an upper strike delta ~0.15
                        # These are placeholders
                        'lower_strike_delta': 0.3,
                        'upper_strike_delta': 0.15
                    }
                else:
                    return {
                        'strategy': 'BEAR_PUT_SPREAD',
                        'confidence': confidence,
                        'expiration_days': 45,
                        # For a put spread, we might want an upper strike delta ~ -0.30, lower ~ -0.15
                        'upper_strike_delta': -0.3,
                        'lower_strike_delta': -0.15
                    }

            return {'strategy': 'NO_TRADE', 'confidence': confidence}
        except Exception as e:
            logger.error(f"Error determining strategy: {e}", exc_info=True)
            return {'strategy': 'NO_TRADE', 'confidence': 0}

    def _apply_costs(self, premium: float) -> float:
        """
        Subtract commissions & slippage from an absolute premium cost, returning net premium.
        For example, if premium=5.00 means $5.00 per contract, we might reduce it by $0.10 total, etc.
        """
        transaction_cost = premium * (self.commission_rate + self.slippage)
        return premium - transaction_cost

    def _get_option_chain(self, symbol: str, expiration_date: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetch the current option chain from yfinance for the given expiration date.
        In practice, for backtesting historically, you need historical chain data.
        """
        try:
            ticker = yf.Ticker(symbol)
            chain = ticker.option_chain(expiration_date)
            return {
                'calls': chain.calls,
                'puts': chain.puts
            }
        except Exception as e:
            logger.warning(f"Failed to fetch option chain for {symbol} {expiration_date}: {e}")
            return None

    def _choose_strikes_by_delta(self, chain_df: pd.DataFrame, target_delta: float, call_put: str) -> Optional[pd.Series]:
        """
        Choose an option from calls/puts chain that best matches the target_delta.
        We'll look at the 'delta' column if available; if not, we approximate by in/out of the money approach.
        """
        # yfinance might not always give the 'delta' column. 
        # If it does not, we approximate by selecting the strike near current underlying price +/-.
        if 'delta' in chain_df.columns:
            # find closest delta
            chain_df['delta_diff'] = (chain_df['delta'] - target_delta).abs()
            # pick option with minimal difference
            chosen = chain_df.loc[chain_df['delta_diff'].idxmin()]
        else:
            # fallback approach if delta not provided: pick strike near "target_delta" OTM
            # This is extremely naive. We'll interpret delta>0 means a call, delta<0 means a put
            # Actually let's pick by distance from lastPrice or underlyingPrice if available
            # This is a placeholder approach
            # For a 0.30 call, pick a strike 70% of the way from underlying?
            # In reality, you'd need a deeper approach or greeks from another source.
            logger.info("No 'delta' in chain. Using approximate strike selection by close to money.")
            chain_df['mid'] = (chain_df['bid'] + chain_df['ask']) / 2
            if call_put == 'call':
                # assume target_delta=0.3 => pick an OTM call ~ 30% below current underlying
                chosen = chain_df.iloc[len(chain_df)//2]  # naive: pick middle
            else:
                chosen = chain_df.iloc[len(chain_df)//4]  # naive
        return chosen

    def _simulate_trade(self, symbol: str, entry_date: pd.Timestamp, strategy_info: Dict[str, float], 
                        entry_price: float) -> Optional[Dict[str, Any]]:
        """
        1) Determine the actual expiration date for the trade (approx).
        2) Fetch an option chain from yfinance, pick strikes by delta, compute net debit/credit.
        3) Return a dictionary describing the position so we can revalue or finalize it at exit.
        """
        # Actual expiration date
        holding_period = strategy_info['expiration_days']
        expiry_dt = entry_date + pd.Timedelta(days=holding_period)

        # Step 1: We'll find the next available monthly expiration from yfinance. 
        # In real usage, you'd filter the actual expiration date from ticker.options if it exists.
        ticker = yf.Ticker(symbol)
        all_expirations = ticker.options
        if not all_expirations:
            return None

        # Find the earliest expiration date that is >= expiry_dt
        # note: these are strings in format 'YYYY-MM-DD' from yfinance
        best_exp_str = None
        for exp_str in all_expirations:
            try:
                dt = datetime.strptime(exp_str, '%Y-%m-%d')
                if dt.date() >= expiry_dt.date():
                    best_exp_str = exp_str
                    break
            except:
                continue

        if not best_exp_str:
            logger.info(f"No suitable expiration found for {symbol} after {expiry_dt}.")
            return None

        # Step 2: Get the chain for that expiration
        chain_data = self._get_option_chain(symbol, best_exp_str)
        if not chain_data:
            return None

        # Decide if we are dealing with a bull call spread or bear put spread
        if strategy_info['strategy'] == 'BULL_CALL_SPREAD':
            call_lower_delta = strategy_info['lower_strike_delta']
            call_upper_delta = strategy_info['upper_strike_delta']
            leg1 = self._choose_strikes_by_delta(chain_data['calls'], call_lower_delta, 'call')
            leg2 = self._choose_strikes_by_delta(chain_data['calls'], call_upper_delta, 'call')
            if leg1 is None or leg2 is None:
                return None
            # net premium (debit) is price(leg1) - price(leg2)
            leg1_mid = (leg1['bid'] + leg1['ask'])/2
            leg2_mid = (leg2['bid'] + leg2['ask'])/2

            # We pay net premium for the bull call spread
            net_debit = leg1_mid - leg2_mid
            net_debit = self._apply_costs(net_debit)
            if net_debit < 0:
                # Means credit scenario, but let's keep it consistent
                net_debit = abs(net_debit)

            return {
                'symbol': symbol,
                'strategy': 'BULL_CALL_SPREAD',
                'entry_date': entry_date,
                'expiration_date': best_exp_str,
                'leg1_strike': float(leg1['strike']),
                'leg2_strike': float(leg2['strike']),
                'leg1_mid': float(leg1_mid),
                'leg2_mid': float(leg2_mid),
                'net_debit': float(net_debit),
                'confidence': strategy_info['confidence']
            }
        else:
            # Bear put spread
            put_upper_delta = strategy_info['upper_strike_delta']
            put_lower_delta = strategy_info['lower_strike_delta']
            leg1 = self._choose_strikes_by_delta(chain_data['puts'], put_upper_delta, 'put')
            leg2 = self._choose_strikes_by_delta(chain_data['puts'], put_lower_delta, 'put')
            if leg1 is None or leg2 is None:
                return None

            leg1_mid = (leg1['bid'] + leg1['ask'])/2
            leg2_mid = (leg2['bid'] + leg2['ask'])/2
            # net debit for bear put is price(leg1) - price(leg2)
            net_debit = leg1_mid - leg2_mid
            net_debit = self._apply_costs(net_debit)
            if net_debit < 0:
                net_debit = abs(net_debit)

            return {
                'symbol': symbol,
                'strategy': 'BEAR_PUT_SPREAD',
                'entry_date': entry_date,
                'expiration_date': best_exp_str,
                'leg1_strike': float(leg1['strike']),
                'leg2_strike': float(leg2['strike']),
                'leg1_mid': float(leg1_mid),
                'leg2_mid': float(leg2_mid),
                'net_debit': float(net_debit),
                'confidence': strategy_info['confidence']
            }

    def _close_trade(self, open_position: Dict[str, Any], exit_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Attempt to fetch the option chain near exit_date for the same expiration to find final premium.
        Then compute PnL relative to net_debit at entry.
        """
        symbol = open_position['symbol']
        expiration_str = open_position['expiration_date']
        net_debit = open_position['net_debit']

        # We'll check an approximate chain on the exit date or near it
        # In real usage, you'd want day-by-day data or the final day/time before expiration
        # If exit_date is after the real expiration, we assume final settlement.

        # If the exit_date is beyond the expiration_str, the spread is at final settlement
        expiry_dt = datetime.strptime(expiration_str, '%Y-%m-%d')
        if exit_date > pd.Timestamp(expiry_dt):
            # final intrinsic value approach
            # For a bull call spread, max value is leg2_strike - leg1_strike, etc. 
            # We'll do a quick approximation:
            if open_position['strategy'] == 'BULL_CALL_SPREAD':
                # If underlying close > leg2_strike, spread is fully in the money
                # Intrinsic = (leg2_strike - leg1_strike)
                # But we need the underlying's final close
                final_price = self._get_underlying_close(symbol, exit_date)
                leg1_strike = open_position['leg1_strike']
                leg2_strike = open_position['leg2_strike']
                if final_price is None:
                    final_value = 0.0
                else:
                    # payoff min( spread width, final_price - leg1_strike ) basically
                    spread_width = leg2_strike - leg1_strike
                    intrinsic = max(0.0, final_price - leg1_strike)
                    final_value = min(spread_width, intrinsic)
            else:
                # Bear put
                final_price = self._get_underlying_close(symbol, exit_date)
                leg1_strike = open_position['leg1_strike']  # higher strike
                leg2_strike = open_position['leg2_strike']
                spread_width = leg1_strike - leg2_strike
                if final_price is None:
                    final_value = 0.0
                else:
                    intrinsic = max(0.0, leg1_strike - final_price)
                    final_value = min(spread_width, intrinsic)
            # final_value is in $ *per share*, net_debit was also per share if 1 contract => 100 shares
            # If we assume 1 contract, multiply by 100
            final_value *= 100

            # net_debit was also presumably "per contract" * 100. We'll do the difference:
            net_debit *= 100
            profit = final_value - net_debit
            profit_pct = (profit / net_debit) * 100.0 if net_debit != 0 else 0.0
            return {
                'exit_date': exit_date,
                'exit_price': final_value,
                'profit_pct_net': profit_pct
            }
        else:
            # If the exit_date < expiration, we try to fetch the chain at that day to see current mid:
            chain_data = self._get_option_chain(symbol, expiration_str)
            if not chain_data:
                # fallback: treat as if we held to expiration
                return self._close_trade(open_position, pd.Timestamp(expiry_dt))

            if open_position['strategy'] == 'BULL_CALL_SPREAD':
                # Find the same strikes in the chain calls
                leg1 = chain_data['calls'][chain_data['calls']['strike'] == open_position['leg1_strike']]
                leg2 = chain_data['calls'][chain_data['calls']['strike'] == open_position['leg2_strike']]
                if not leg1.empty and not leg2.empty:
                    leg1_mid = (leg1.iloc[0]['bid'] + leg1.iloc[0]['ask'])/2
                    leg2_mid = (leg2.iloc[0]['bid'] + leg2.iloc[0]['ask'])/2
                    net_credit_now = leg1_mid - leg2_mid
                    # multiply by 100 if we assume 1 contract
                    profit_cash = (net_credit_now - net_debit)*100
                    profit_pct = (profit_cash / (net_debit*100))*100
                    return {
                        'exit_date': exit_date,
                        'exit_price': net_credit_now,
                        'profit_pct_net': profit_pct
                    }
                else:
                    # If not found, fallback
                    logger.info(f"Strikes not found near exit for {symbol} bull call. Using final settlement.")
                    return self._close_trade(open_position, pd.Timestamp(expiry_dt))
            else:
                # Bear put
                leg1 = chain_data['puts'][chain_data['puts']['strike'] == open_position['leg1_strike']]
                leg2 = chain_data['puts'][chain_data['puts']['strike'] == open_position['leg2_strike']]
                if not leg1.empty and not leg2.empty:
                    leg1_mid = (leg1.iloc[0]['bid'] + leg1.iloc[0]['ask'])/2
                    leg2_mid = (leg2.iloc[0]['bid'] + leg2.iloc[0]['ask'])/2
                    net_credit_now = leg1_mid - leg2_mid
                    profit_cash = (net_credit_now - net_debit)*100
                    profit_pct = (profit_cash / (net_debit*100))*100
                    return {
                        'exit_date': exit_date,
                        'exit_price': net_credit_now,
                        'profit_pct_net': profit_pct
                    }
                else:
                    logger.info(f"Strikes not found near exit for {symbol} bear put. Using final settlement.")
                    return self._close_trade(open_position, pd.Timestamp(expiry_dt))

    def _get_underlying_close(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """
        Retrieve the underlying's closing price from self.historical_data on the given date,
        or the final close if the date is out of range. 
        """
        df = self.historical_data.get(symbol, pd.DataFrame())
        if df.empty:
            return None
        if date in df.index:
            return float(df.loc[date]['Close'])
        else:
            # fallback: use last row if date out of range
            return float(df.iloc[-1]['Close'])

    def _backtest_symbol(self, hist_data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Run backtest for a single symbol. But now, we use real option premium from yfinance.
        For each potential trade:
          1. Determine strategy
          2. Select strikes and compute net debit
          3. Hold until target exit date, then compute final spread value
        """
        trades = []
        current_trade = None
        window_size = 100

        for i in range(window_size, len(hist_data)):
            if current_trade is not None:
                # If we've reached or passed the exit date, finalize the trade
                if hist_data.index[i] >= current_trade['exit_date']:
                    closed_info = self._close_trade(current_trade, hist_data.index[i])
                    if closed_info:
                        trade_result = {
                            **current_trade,
                            'exit_date': closed_info['exit_date'],
                            'exit_price': closed_info['exit_price'],
                            'profit_pct_net': closed_info['profit_pct_net']
                        }
                        trades.append(trade_result)
                    current_trade = None
                else:
                    # still in trade
                    continue

            # Lookback window
            hist_window = hist_data.iloc[i - window_size : i]
            tech_data = self.get_technical_indicators(hist_window)
            if not tech_data:
                continue

            strategy = self.determine_options_strategy(tech_data)
            if strategy['strategy'] == 'NO_TRADE':
                continue

            # Construct the actual option spread with net debit
            entry_date = hist_data.index[i]
            sim_trade = self._simulate_trade(symbol, entry_date, strategy, hist_data.iloc[i]['Close'])
            if sim_trade:
                # We'll define an exit_date from the trade info
                # A naive approach: hold until the chosen expiration date or the end of hist_data
                expiry_dt = datetime.strptime(sim_trade['expiration_date'], '%Y-%m-%d')
                future_dates = hist_data.index[hist_data.index >= expiry_dt]
                exit_dt = future_dates[0] if len(future_dates) else hist_data.index[-1]
                sim_trade['exit_date'] = exit_dt
                current_trade = sim_trade

        # If a trade is still open at the end, close it
        if current_trade is not None:
            closed_info = self._close_trade(current_trade, hist_data.index[-1])
            if closed_info:
                trade_result = {
                    **current_trade,
                    'exit_date': closed_info['exit_date'],
                    'exit_price': closed_info['exit_price'],
                    'profit_pct_net': closed_info['profit_pct_net']
                }
                trades.append(trade_result)

        if trades:
            trades_df = pd.DataFrame(trades)
            return {
                'total_trades': len(trades),
                'winning_trades': len(trades_df[trades_df['profit_pct_net'] > 0]),
                'avg_profit': trades_df['profit_pct_net'].mean(),
                'max_profit': trades_df['profit_pct_net'].max(),
                'max_loss': trades_df['profit_pct_net'].min(),
                'sharpe_ratio': self.calculate_sharpe_ratio(trades_df['profit_pct_net']),
                'trades': trades
            }

        return None

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate the annualized Sharpe ratio.
        """
        if len(returns) < 2:
            return 0.0
        decimal_returns = returns / 100.0
        return np.sqrt(252) * (decimal_returns.mean() / decimal_returns.std())

    def run_backtest(self, start_date: str, end_date: str, optimize: bool = True) -> Dict[str, Any]:
        """
        Download data, run initial backtest, optionally optimize, and store results.
        """
        logger.info("Starting backtest with more realistic option pricing...")

        self.download_data(start_date, end_date)

        initial_results = {}
        technical_data_collection = {}

        logger.info("Running initial backtest...")
        for symbol in self.watchlist:
            if symbol not in self.historical_data or self.historical_data[symbol].empty:
                continue
            logger.info(f"Backtesting {symbol}...")

            data = self.historical_data[symbol]
            results = self._backtest_symbol(data, symbol)
            if results:
                initial_results[symbol] = results

                # For ML, we can store the final trade's profit, etc.
                # Just as a demonstration:
                last_trade_profit = results['trades'][-1]['profit_pct_net'] if results['trades'] else 0.0
                # We'll do a minimal technical data extraction
                tech_data = self.get_technical_indicators(data.tail(100))
                if tech_data:
                    # Convert numeric
                    processed = {}
                    for k,v in tech_data.items():
                        if isinstance(v, dict):
                            psub = {}
                            for k2,v2 in v.items():
                                psub[k2] = float(v2)
                            processed[k] = psub
                        else:
                            processed[k] = float(v)
                    processed['profit_pct'] = float(last_trade_profit)
                    technical_data_collection[symbol] = processed

        self.results = initial_results

        # If you have a StrategyOptimizer and want to optimize
        if optimize and technical_data_collection:
            logger.info("Running strategy optimization...")
            optimizer = StrategyOptimizer()
            success = optimizer.train_model(technical_data_collection)
            if success:
                # We could call optimizer.generate_strategy_recommendations(...)
                # Then update parameters. Omitted for brevity.
                pass

        return initial_results

    def print_results(self) -> None:
        """
        Print the results to console
        """
        if not self.results:
            logger.info("No results to display.")
            return
        logger.info("\n=== Backtest Results ===\n")
        for symbol, result in self.results.items():
            logger.info(f"{symbol}:")
            logger.info(f"  Total Trades: {result['total_trades']}")
            logger.info(f"  Winning Trades: {result['winning_trades']}")
            logger.info(f"  Avg Profit: {result['avg_profit']:.2f}%")
            logger.info(f"  Max Profit: {result['max_profit']:.2f}%")
            logger.info(f"  Max Loss: {result['max_loss']:.2f}%")
            logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    def plot_results(self) -> None:
        """
        You can add your own equity curve plot or bar chart for each symbol's performance here.
        """
        if not self.results:
            logger.info("No results to plot.")
            return
        # For demonstration, plot avg_profit by symbol
        symbols = list(self.results.keys())
        avg_profits = [self.results[s]['avg_profit'] for s in symbols]

        plt.figure(figsize=(8,5))
        plt.bar(symbols, avg_profits)
        plt.xlabel("Symbol")
        plt.ylabel("Average Profit %")
        plt.title("Average Profit by Symbol (Options Spread)")
        plt.show()

if __name__ == "__main__":
    watchlist = ["AAPL","MSFT","GOOGL", "META", "NVDA", "SPY", "QQQ"]
    backtester = OptionsBacktester(watchlist)

    start_input = input("Start date (YYYY-MM): ")
    end_input = input("End date (YYYY-MM): ")

    start_date = f"{start_input}-01"
    end_date = f"{end_input}-01"

    results = backtester.run_backtest(start_date, end_date, optimize=False)
    backtester.print_results()
    backtester.plot_results()
