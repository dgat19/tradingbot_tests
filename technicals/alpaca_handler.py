import os
import time
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Initialize dotenv credentials
load_dotenv()

class AlpacaHandler:
    def __init__(self):
        self.ALPACA_API_KEY = os.getenv('APCA_PAPER_API_KEY')
        self.ALPACA_SECRET_KEY = os.getenv('APCA_PAPER_SECRET_KEY')
        self.PAPER_BASE_URL = 'https://paper-api.alpaca.markets'
        self.DATA_BASE_URL = 'https://data.alpaca.markets'
        
        # Initialize API
        self.api = tradeapi.REST(
            key_id=self.ALPACA_API_KEY,
            secret_key=self.ALPACA_SECRET_KEY,
            base_url=self.PAPER_BASE_URL,
            api_version='v2'
        )
        
        self.verify_connection()

    def verify_connection(self):
        """Test the connection to Alpaca API"""
        try:
            account = self.api.get_account()
            print("\nSuccessfully connected to Alpaca API")
            print(f"Account Status: {account.status}")
            print(f"Cash Balance: ${float(account.cash):.2f}")
            return True
        except Exception as e:
            print(f"Error connecting to Alpaca API: {e}")
            return False

    def get_account_info(self):
        """Get current account information"""
        try:
            account = self.api.get_account()
            return float(account.cash)
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return 0.0

    def get_position_qty(self, ticker):
        """Get current position quantity for a ticker (long positions)"""
        try:
            position = self.api.get_position(ticker)
            return float(position.qty)
        except Exception:
            return 0.0

    def get_position_entry_price(self, ticker):
        """Get entry price for current position"""
        try:
            position = self.api.get_position(ticker)
            return float(position.avg_entry_price)
        except Exception:
            return None

    def get_filled_price(self, order_id):
        """Get filled price for an order with reduced delay to minimize slippage"""
        try:
            # Reduced sleep time to minimize delay.
            time.sleep(0.5)
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                order = self.api.get_order(order_id)
                
                if order.status == 'filled':
                    # Use the filled_avg_price directly provided by the API.
                    return float(order.filled_avg_price)
                elif order.status in ['canceled', 'expired', 'rejected']:
                    return None
                    
                attempt += 1
                time.sleep(0.5)  # Reduced delay between attempts
                
            return None
            
        except Exception as e:
            print(f"Error getting fill price for order {order_id}: {e}")
            try:
                position = self.api.get_position(order.symbol)
                return float(position.avg_entry_price)
            except:
                return None

    def get_historical_data(self, ticker, timeframe='1Min', limit=1000):
        """Get historical bar data with IEX feed"""
        try:
            bars = self.api.get_bars(
                ticker,
                timeframe,
                limit=limit,
                adjustment='raw',
                feed='iex'
            ).df
            
            if bars.empty:
                print(f"No data returned for {ticker}")
                return None

            return bars

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            print(f"Type of error: {type(e)}")
            return None

    def get_daily_data(self, ticker, limit=2):
        """Get daily bar data"""
        try:
            bars = self.api.get_bars(
                ticker,
                "1Day",
                limit=limit,
                adjustment='raw',
                feed='iex'
            ).df
            
            if bars.empty:
                print(f"No daily data returned for {ticker}")
                return None

            return bars

        except Exception as e:
            print(f"Error fetching daily data for {ticker}: {e}")
            return None

    def place_bracket_order(self, ticker, side, qty, entry_price, take_profit_pct=2.0, stop_loss_pct=0.8):
        """Place a bracket order with take profit and stop loss"""
        try:
            q = float(qty)
            if q != int(q):
                return self.place_simple_order(ticker, side, qty, entry_price)

            take_profit_price = round(entry_price * (1 + take_profit_pct/100), 2)
            stop_loss_price = round(entry_price * (1 - stop_loss_pct/100), 2)
            
            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day',
                order_class='bracket',
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_loss_price}
            )
            
            return self.get_filled_price(order.id)
            
        except Exception as e:
            print(f"Order error for {ticker}: {e}")
            return None

    def place_simple_order(self, ticker, side, qty, entry_price):
        """Place a simple market order"""
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=str(qty),
                side=side,
                type='market',
                time_in_force='day'
            )
            
            return self.get_filled_price(order.id)
            
        except Exception as e:
            print(f"Simple order error for {ticker}: {e}")
            return None

    def place_market_sell(self, ticker, sell_qty, current_price):
        """Place a market sell order for long positions"""
        try:
            current_qty = self.get_position_qty(ticker)
            if current_qty <= 0:
                print(f"{ticker}: No position available to sell.")
                return None

            sell_qty = min(abs(sell_qty), current_qty)
            if sell_qty <= 0:
                print(f"{ticker}: Invalid sell quantity {sell_qty}")
                return None

            order = self.api.submit_order(
                symbol=ticker,
                qty=str(round(sell_qty, 3)),
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            return self.get_filled_price(order.id)
            
        except Exception as e:
            print(f"Error closing position for {ticker}: {e}")
            return None

    # --- New Methods for Short Trading ---

    def place_short_order(self, ticker, qty, current_price):
        """
        Place a market sell order to open a short position.
        Note: This assumes you have margin enabled and short selling is allowed.
        """
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=str(qty),
                side='sell',
                type='market',
                time_in_force='day'
            )
            return self.get_filled_price(order.id)
        except Exception as e:
            print(f"Error placing short order for {ticker}: {e}")
            return None

    def get_short_position_qty(self, ticker):
        """
        Get the current short position quantity for a ticker.
        If the position is short, Alpaca will return a negative quantity.
        This function returns the absolute value for short positions, or 0 if no short exists.
        """
        try:
            position = self.api.get_position(ticker)
            qty = float(position.qty)
            if qty < 0:
                return abs(qty)
            else:
                return 0.0
        except Exception:
            return 0.0

    def place_cover_order(self, ticker, qty, current_price):
        """
        Place a market buy order to cover a short position.
        """
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=str(qty),
                side='buy',
                type='market',
                time_in_force='day'
            )
            return self.get_filled_price(order.id)
        except Exception as e:
            print(f"Error placing cover order for {ticker}: {e}")
            return None

    def is_extended_hours_eligible(self, ticker):
        """Check if a stock is eligible for extended hours trading"""
        try:
            asset = self.api.get_asset(ticker)
            return asset.tradable and asset.easy_to_borrow and asset.marginable
        except Exception as e:
            print(f"Error checking extended hours eligibility for {ticker}: {e}")
            return False
