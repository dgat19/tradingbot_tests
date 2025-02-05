import alpaca_trade_api as tradeapi
import requests
import pandas as pd
import asyncio
import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any

from technicals import TechnicalIndicators
from market_hours import MarketHours
from strategy_optimizer import StrategyOptimizer
from backtest import OptionsBacktester

class UnifiedOptionsTrader:
    """
    A class that handles paper trading, live trading, and backtesting of options strategies.
    """

    # API credentials
    API_KEY = "PKVLCO0143VFLPQE4FXK"
    API_SECRET = "KMDx7LC890SXDbBGagSn6Ezf4GIzZ8aEfg5rULXb"

    # Class constants for API endpoints
    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"
    OPTIONS_BASE_URL = "https://data.alpaca.markets"

    def __init__(self, mode: str = "paper"):
        """
        Initialize the trader with specified mode.
        
        :param mode: One of "paper", "live", or "backtest"
        """
        self.mode = mode.lower()
        self.api_key = self.API_KEY
        self.api_secret = self.API_SECRET
        
        # Set trading URL based on mode
        if self.mode == "paper":
            self.trading_url = self.PAPER_BASE_URL
        elif self.mode == "live":
            self.trading_url = self.LIVE_BASE_URL
        elif self.mode != "backtest":
            raise ValueError("Invalid mode. Must be 'paper', 'live', or 'backtest'")
        
        self.logger = self._setup_logger()
        
        if self.mode != "backtest":
            self.trading_api = tradeapi.REST(self.api_key, self.api_secret, self.trading_url)
            self.logger.info(f"Initialized UnifiedOptionsTrader in {mode.upper()} mode")

        # Initialize other components
        self.watchlist: List[str] = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","TSLA","SPY","QQQ"]
        self.market_hours = MarketHours()
        self.strategy_optimizer = StrategyOptimizer()
        self.backtester = None if self.mode != "backtest" else OptionsBacktester(self.watchlist)

        # Trading parameters
        self.confidence_threshold = 0.7
        self.stop_loss_pct = 0.15
        self.take_profit_pct = 0.25
        self.ti = TechnicalIndicators()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for the trader."""
        logger = logging.getLogger('UnifiedOptionsTrader')
        logger.setLevel(logging.INFO)
        
        # Create a file handler with the mode in the filename
        fh = logging.FileHandler(f'options_trader_{self.mode}.log')
        fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fm)
        logger.addHandler(fh)
        return logger

    def _format_option_symbol(self, underlying: str, expiration: str, option_type: str, strike: float) -> str:
        """
        Format option symbol in OSI format that Alpaca expects.
        Example: AAPL230616C00150000 for AAPL $150 Call expiring June 16, 2023
        """
        # Convert expiration from YYYY-MM-DD to YYMMDD
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        exp_str = exp_date.strftime('%y%m%d')
        
        # Format strike price to 8 digits with padded zeros
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"
        
        # C for calls, P for puts
        option_type_char = 'C' if option_type.lower() == 'call' else 'P'
        
        # Combine in OSI format: Underlying + Expiration + C/P + Strike
        return f"{underlying}{exp_str}{option_type_char}{strike_str}"

    def _get_expirations(self, underlying: str) -> List[str]:
        """
        Fetch available expirations for the underlying symbol.
        Falls back to yfinance if Alpaca data isn't available.
        """
        try:
            # First try yfinance as it's more reliable for options data
            ticker = yf.Ticker(underlying)
            expirations = ticker.options
            
            if expirations:
                self.logger.info(f"Found {len(expirations)} expirations for {underlying} using yfinance")
                return sorted(expirations)
                
            # If yfinance fails, try Alpaca
            url = f"{self.OPTIONS_BASE_URL}/v2/options/expirations"
            params = {
                "underlying_symbol": underlying
            }
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            }
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if "expirations" in data:
                expirations = data["expirations"]
                self.logger.info(f"Found {len(expirations)} expirations for {underlying} using Alpaca")
                return sorted(expirations)
                
            return []
        except Exception as e:
            self.logger.error(f"Error fetching expirations for {underlying}: {e}", exc_info=True)
            return []

    def _fetch_options_chain(self, symbol: str, expiration: str, call_or_put: str) -> List[Dict[str,Any]]:
        """
        Fetch the options chain with fallback to yfinance.
        """
        try:
            # First try yfinance
            ticker = yf.Ticker(symbol)
            chain = ticker.option_chain(expiration)
            
            if chain:
                # Convert yfinance chain to our expected format
                contracts = []
                if call_or_put.lower() == 'call':
                    df = chain.calls
                else:
                    df = chain.puts
                    
                for _, row in df.iterrows():
                    contract = {
                        'underlying': symbol,
                        'expiration': expiration,
                        'type': call_or_put.lower(),
                        'strike': float(row['strike']),
                        'bid': float(row['bid']),
                        'ask': float(row['ask'])
                    }
                    contracts.append(contract)
                
                self.logger.info(f"Fetched {len(contracts)} {call_or_put} contracts for {symbol} exp={expiration} using yfinance")
                return contracts

            # Fallback to Alpaca if yfinance fails
            url = f"{self.OPTIONS_BASE_URL}/v2/options/chain"
            params = {
                "underlying_symbol": symbol,
                "expiration": expiration,
                "option_type": call_or_put.lower()
            }
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            }
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            contracts = data.get("options", [])
            self.logger.info(f"Fetched {len(contracts)} {call_or_put} contracts for {symbol} exp={expiration} using Alpaca")
            return contracts
        except Exception as e:
            self.logger.error(f"Error fetching option chain for {symbol} {expiration} {call_or_put}: {e}", exc_info=True)
            return []

    def _submit_single_leg_order(self, contract_info: Dict[str, Any], side: str, qty: int):
        """Submit a single-leg options order with properly formatted symbol."""
        try:
            symbol = self._format_option_symbol(
                underlying=contract_info['underlying'],
                expiration=contract_info['expiration'],
                option_type=contract_info['type'],
                strike=contract_info['strike']
            )
            
            order = self.trading_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
            self.logger.info(f"[{self.mode.upper()}] Order => symbol={symbol} side={side} ID={order.id}")
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}", exc_info=True)

    def _submit_spread_orders(self, buy_leg: Dict[str, Any], sell_leg: Dict[str, Any], qty: int):
        """Submit a two-legged spread order with properly formatted symbols."""
        try:
            # Format buy leg symbol
            buy_symbol = self._format_option_symbol(
                underlying=buy_leg['underlying'],
                expiration=buy_leg['expiration'],
                option_type=buy_leg['type'],
                strike=buy_leg['strike']
            )
            
            # Format sell leg symbol
            sell_symbol = self._format_option_symbol(
                underlying=sell_leg['underlying'],
                expiration=sell_leg['expiration'],
                option_type=sell_leg['type'],
                strike=sell_leg['strike']
            )

            # Submit buy leg
            buy_order = self.trading_api.submit_order(
                symbol=buy_symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            self.logger.info(f"[{self.mode.upper()}] Spread buy => {buy_symbol}, ID={buy_order.id}")

            # Submit sell leg
            sell_order = self.trading_api.submit_order(
                symbol=sell_symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            self.logger.info(f"[{self.mode.upper()}] Spread sell => {sell_symbol}, ID={sell_order.id}")
        except Exception as e:
            self.logger.error(f"Error placing spread orders: {e}", exc_info=True)

    async def get_technical_indicators(self, symbol: str) -> Dict[str,Any]:
        """Calculate technical indicators for a symbol."""
        df = yf.Ticker(symbol).history(period="6mo")
        if df.empty:
            return {}
        df.index = df.index.tz_localize(None)
        c = df['Close'].values
        h = df['High'].values
        l = df['Low'].values
        indicators = {}
        indicators['RSI'] = self.ti.calculate_rsi(c, 14)
        indicators['MACD'] = self.ti.calculate_macd(pd.Series(c))
        indicators['BB'] = self.ti.calculate_bollinger_bands(pd.Series(c),20,2)
        indicators['ADX'] = self.ti.calculate_adx(h,l,c,14)
        indicators['current_price'] = float(c[-1])
        return indicators

    async def analyze_financials(self, symbol: str) -> Dict[str,Any]:
        """Get financial metrics for a symbol."""
        info = yf.Ticker(symbol).info
        return {
            'PE': info.get('forwardPE'),
            'PB': info.get('priceToBook'),
            'PS': info.get('priceToSalesTrailing12Months')
        }

    def fetch_order_flow_imbalance(self, symbol: str) -> float:
        """Simulate order flow imbalance (placeholder)."""
        import random
        return round(1.0 + (random.random()-0.5)*0.4, 2)

    def combine_signals(self, tech_data: Dict[str,Any], fund_data: Dict[str,Any]) -> Dict[str,Any]:
        """Combine technical and fundamental signals to make trading decisions."""
        if not tech_data:
            return {'strategy':'NO_TRADE','confidence':0.0}
        
        bull = 0
        bear = 0

        rsi = tech_data.get('RSI',50)
        if rsi<30:
            bull+=2
        elif rsi>70:
            bear+=2

        macd_data = tech_data.get('MACD',{})
        if macd_data.get('macd',0)>macd_data.get('signal',0):
            bull+=1
        else:
            bear+=1

        adx = tech_data.get('ADX',0)
        if adx>25:
            bull+=0.5

        imbalance = self.fetch_order_flow_imbalance(symbol="SYMBOL")
        if imbalance>1.05:
            bull+=1
        elif imbalance<0.95:
            bear+=1

        total=bull+bear
        if total==0:
            return {'strategy':'NO_TRADE','confidence':0.0}
        
        conf = max(bull,bear)/total
        if conf<self.confidence_threshold:
            return {'strategy':'NO_TRADE','confidence':conf}

        if bull>=3:
            return {'strategy':'LONG_CALL','confidence':conf}
        elif bull>bear:
            return {'strategy':'BULL_CALL_SPREAD','confidence':conf}
        elif bear>=3:
            return {'strategy':'LONG_PUT','confidence':conf}
        else:
            return {'strategy':'BEAR_PUT_SPREAD','confidence':conf}

    async def execute_option_trade(self, symbol: str, decision: Dict[str,Any], qty: int=1) -> bool:
        """Execute an options trade based on the strategy decision."""
        if decision['strategy'] == 'NO_TRADE':
            return False

        try:
            # Get expiration dates
            all_exps = self._get_expirations(symbol)
            if not all_exps:
                self.logger.info(f"No expirations for {symbol}, skipping.")
                return False

            # Pick expiration 30+ days out
            dt_target = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            chosen_exp = self._pick_expiration_gte(all_exps, dt_target)
            if not chosen_exp:
                self.logger.info(f"No suitable expiration >= {dt_target} for {symbol}.")
                return False

            # Get current price
            df = yf.Ticker(symbol).history(period="1d")
            if df.empty:
                return False
            cur_price = float(df['Close'].iloc[-1])

            if decision['strategy'] == 'LONG_CALL':
                chain = self._fetch_options_chain(symbol, chosen_exp, 'call')
                if not chain:
                    return False
                target_strike = cur_price * 1.05
                chain_sorted = sorted(chain, key=lambda c: c['strike'])
                chosen = None
                for c in chain_sorted:
                    if c['strike'] >= target_strike:
                        chosen = c
                        break
                if not chosen:
                    chosen = chain_sorted[-1]

                self._submit_single_leg_order(chosen, side='buy', qty=qty)
                return True

            elif decision['strategy'] == 'LONG_PUT':
                chain = self._fetch_options_chain(symbol, chosen_exp, 'put')
                if not chain:
                    return False
                target_strike = cur_price * 0.95
                chain_sorted = sorted(chain, key=lambda c: c['strike'])
                chosen = None
                for c in chain_sorted[::-1]:
                    if c['strike'] <= target_strike:
                        chosen = c
                        break
                if not chosen:
                    chosen = chain_sorted[0]

                self._submit_single_leg_order(chosen, side='buy', qty=qty)
                return True

            elif decision['strategy'] == 'BULL_CALL_SPREAD':
                chain = self._fetch_options_chain(symbol, chosen_exp, 'call')
                if len(chain) < 2:
                    return False
                    
                buy_target = cur_price * 0.98
                sell_target = cur_price * 1.05
                chain_sorted = sorted(chain, key=lambda c: c['strike'])
                
                # Find buy leg
                buy_leg = None
                for c in chain_sorted:
                    if c['strike'] <= buy_target:
                        buy_leg = c
                if not buy_leg:
                    buy_leg = chain_sorted[0]
                
                # Find sell leg
                sell_leg = None
                for c in chain_sorted:
                    if c['strike'] >= sell_target:
                        sell_leg = c
                        break
                if not sell_leg:
                    sell_leg = chain_sorted[-1]
                
                self._submit_spread_orders(buy_leg, sell_leg, qty)
                return True

            elif decision['strategy'] == 'BEAR_PUT_SPREAD':
                chain = self._fetch_options_chain(symbol, chosen_exp, 'put')
                if len(chain) < 2:
                    return False
                    
                buy_target = cur_price * 1.02
                sell_target = cur_price * 0.95
                chain_sorted = sorted(chain, key=lambda c: c['strike'], reverse=True)
                
                # Find buy leg
                buy_leg = None
                for c in chain_sorted:
                    if c['strike'] >= buy_target:
                        buy_leg = c
                        break
                if not buy_leg:
                    buy_leg = chain_sorted[0]
                
                # Find sell leg
                sell_leg = None
                reversed_chain = chain_sorted[::-1]
                for c in reversed_chain:
                    if c['strike'] <= sell_target:
                        sell_leg = c
                        break
                if not sell_leg:
                    sell_leg = reversed_chain[-1]
                
                self._submit_spread_orders(buy_leg, sell_leg, qty)
                return True

            return False
            
        except Exception as e:
            self.logger.error(f"Error executing option trade: {e}", exc_info=True)
            return False

    async def manage_positions(self):
        """Manage existing positions, checking for stop-loss and take-profit conditions."""
        try:
            positions = self.trading_api.list_positions()
            for p in positions:
                sym = p.symbol
                entry_price = float(p.avg_entry_price)
                current_price = float(p.current_price)
                if entry_price == 0:
                    continue
                pct = (current_price - entry_price)/entry_price
                if pct <= -self.stop_loss_pct:
                    self.logger.info(f"Stop-loss triggered for {sym} at {pct*100:.2f}% drop")
                    # Close position
                    self.trading_api.submit_order(
                        symbol=sym,
                        qty=p.qty,
                        side='sell' if p.side == 'long' else 'buy',
                        type='market',
                        time_in_force='day'
                    )
                elif pct >= self.take_profit_pct:
                    self.logger.info(f"Take-profit triggered for {sym} at {pct*100:.2f}% gain")
                    # Close position
                    self.trading_api.submit_order(
                        symbol=sym,
                        qty=p.qty,
                        side='sell' if p.side == 'long' else 'buy',
                        type='market',
                        time_in_force='day'
                    )
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}", exc_info=True)

    async def run_backtest(self, start_date: str, end_date: str):
        """Run backtest for the specified date range."""
        if self.mode != "backtest":
            self.logger.error("Cannot run backtest in non-backtest mode")
            return

        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        results = self.backtester.run_backtest(start_date, end_date)
        self.backtester.print_results()
        self.backtester.plot_results()
        return results

    async def run_trading_bot(self):
        """Main trading loop with mode awareness."""
        self.logger.info(f"Starting trading bot in {self.mode.upper()} mode")
        
        while True:
            try:
                if not self.market_hours.is_market_open():
                    st = self.market_hours.time_until_market_open()
                    self.logger.info(f"Market closed. Sleeping {st} s.")
                    await asyncio.sleep(st)
                    continue

                to_close = self.market_hours.time_until_market_close()
                if to_close <= 0:
                    st = self.market_hours.time_until_market_open()
                    self.logger.info(f"Near close. Sleeping {st} s.")
                    await asyncio.sleep(st)
                    continue

                await self.manage_positions()

                for symbol in self.watchlist:
                    if not self.market_hours.is_market_open():
                        break

                    tech_data = await self.get_technical_indicators(symbol)
                    fund_data = await self.analyze_financials(symbol)
                    decision = self.combine_signals(tech_data, fund_data)
                    
                    if decision['strategy'] != 'NO_TRADE':
                        success = await self.execute_option_trade(symbol, decision, qty=1)
                        if success:
                            self.logger.info(f"[{self.mode.upper()}] Executed {decision['strategy']} for {symbol}")
                    
                    await asyncio.sleep(5)

                nap = min(300, max(60, to_close-60))
                await asyncio.sleep(nap)
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying


def get_mode_selection() -> str:
    """Display menu and get user's mode selection."""
    while True:
        print("\n=== Options Trading Mode Selection ===")
        print("1. Paper Trading")
        print("2. Live Trading")
        print("3. Backtesting")
        print("4. Exit")
        
        choice = input("\nSelect mode (1-4): ").strip()
        
        if choice == "1":
            return "paper"
        elif choice == "2":
            print("\nWARNING: You have selected LIVE trading mode!")
            confirm = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirm != "CONFIRM":
                print("Live trading not confirmed. Returning to menu...")
                continue
            return "live"
        elif choice == "3":
            return "backtest"
        elif choice == "4":
            print("\nExiting program...")
            exit()
        else:
            print("\nInvalid choice. Please select 1-4.")


async def main():
    """Main function to run the trading bot or backtest."""
    mode = get_mode_selection()
    
    if mode == "backtest":
        print("\n=== Backtesting Configuration ===")
        
        # Get initial capital
        while True:
            try:
                capital_input = input("Enter initial capital (e.g., 100000 for $100,000): ").strip()
                initial_capital = float(capital_input)
                if initial_capital <= 0:
                    print("Initial capital must be positive.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        start_date = input("Enter start date (YYYY-MM): ").strip()
        end_date = input("Enter end date (YYYY-MM): ").strip()
        
        # Add -01 to convert YYYY-MM to YYYY-MM-DD
        start_date = f"{start_date}-01"
        end_date = f"{end_date}-01"
        
        # Create backtester with initial capital
        trader = UnifiedOptionsTrader(mode=mode, initial_capital=initial_capital)
        await trader.run_backtest(start_date, end_date)
    else:
        trader = UnifiedOptionsTrader(mode=mode)
        print(f"\nStarting options trader in {mode.upper()} mode...")
        await trader.run_trading_bot()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())