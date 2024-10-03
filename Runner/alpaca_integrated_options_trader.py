import time
import logging
import sys
import os
import yfinance as yf
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from news_scraper import get_top_active_movers
from indicators import get_trend_indicator, get_volume_indicator
from ml_trade_performance_evaluation import load_trade_data, train_model, evaluate_trade, adjust_strategy_based_on_model
import sqlite3

# Alpaca API keys - replace with your actual keys or environment variables
ALPACA_API_KEY = "PKOS59NZIX9P0L8VNJK8"
ALPACA_API_SECRET = "7eFxPPP15H3eYW3V0B5i1UdV21sCbS9oW1L4WcoA"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)  # Paper trading mode

# Hardcoded list of stocks that can be traded based on 1% price movement
hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']


# Database setup
conn = sqlite3.connect('trades.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_symbol TEXT,
                    option_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    timestamp TEXT
                 )''')
conn.commit()

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Logging setup
logging.basicConfig(filename="logs/trades.log", level=logging.INFO, format='%(asctime)s - %(message)s')

def analyze_and_place_options_trade(stock_symbol, current_price, qty):
    stock = yf.Ticker(stock_symbol)
    
    expiration_dates = stock.options
    one_month_later = (datetime.datetime.now() + datetime.timedelta(days=30)).date()
    selected_expiration = None
    
    for expiration in expiration_dates:
        exp_date = datetime.datetime.strptime(expiration, '%Y-%m-%d').date()
        if exp_date >= one_month_later:
            selected_expiration = expiration
            break
    
    if not selected_expiration:
        logging.warning(f"No suitable expiration found for {stock_symbol}. Skipping trade.")
        return

    options_chain = stock.option_chain(selected_expiration)
    
    target_call_strike = current_price * 1.10
    suitable_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
    
    target_put_strike = current_price * 0.90
    suitable_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
    
    if not suitable_calls.empty:
        selected_call = suitable_calls.iloc[0]
        logging.info(f"Selected CALL option for {stock_symbol} with strike {selected_call['strike']} and expiration {selected_expiration}")
        place_option_trade(selected_call['contractSymbol'], qty, option_type='call')
    
    elif not suitable_puts.empty:
        selected_put = suitable_puts.iloc[0]
        logging.info(f"Selected PUT option for {stock_symbol} with strike {selected_put['strike']} and expiration {selected_expiration}")
        place_option_trade(selected_put['contractSymbol'], qty, option_type='put')

    else:
        logging.warning(f"No suitable options found for {stock_symbol} around the 10% strike range.")

def place_option_trade(contract_symbol, qty=1, option_type='call'):
    try:
        logging.info(f"Placing {option_type.upper()} order for {contract_symbol}, Quantity: {qty}")
        order = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY if option_type == 'call' else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        logging.info(f"{option_type.capitalize()} order placed for {contract_symbol}")
    except Exception as e:
        logging.error(f"Error placing order for {contract_symbol}: {e}")

def place_trade_and_confirm(stock_symbol, qty=1, option_type='call'):
    """
    Place a trade and confirm its status after placing the order.
    """
    try:
        logging.info(f"Placing {option_type.upper()} order for {stock_symbol}, Quantity: {qty}")
        
        # Create and submit a market order
        order = MarketOrderRequest(
            symbol=stock_symbol,
            qty=qty,
            side=OrderSide.BUY if option_type == 'call' else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        placed_order = trading_client.submit_order(order)
        
        # Confirm the trade was placed successfully
        order_id = placed_order.id
        logging.info(f"Order submitted for {stock_symbol}. Checking status...")
        
        # Wait briefly to allow time for order processing
        time.sleep(5)  # Wait 5 seconds before checking the status
        
        # Check and confirm the trade status
        confirm_trade_placement(order_id)
    
    except Exception as e:
        logging.error(f"Error placing trade for {stock_symbol}: {e}")

def automated_trading(stock_symbol, qty=1, evaluated_stocks=set()):
    if stock_symbol in evaluated_stocks:
        return  # Skip if already evaluated in this cycle

    evaluated_stocks.add(stock_symbol)
    stock_price = yf.Ticker(stock_symbol).history(period="1d")["Close"].iloc[-1]

    trend = get_trend_indicator(stock_symbol)
    volume_signal = get_volume_indicator(stock_symbol)

    predicted_outcome = evaluate_trade(model, stock_symbol, price_at_trade=stock_price, volatility=0.35, volume_signal=volume_signal, market_sentiment=trend)

    if predicted_outcome == 0:
        logging.info(f"Trade evaluation for {stock_symbol}: Expected outcome: Loss")
        logging.info(f"Adjusting strategy for {stock_symbol} to avoid losses...")
        logging.info(f"Consider adjusting entry point, price levels, or wait for better sentiment/volatility for {stock_symbol}")
        adjust_strategy_based_on_model(model, stock_symbol, price_at_trade=stock_price, volatility=0.35, volume_signal=volume_signal, market_sentiment=trend)
    elif predicted_outcome == 1:
        logging.info(f"Trade evaluation for {stock_symbol}: Expected outcome: Profitable")
        logging.info(f"Analyzing options chain for {stock_symbol} to place a trade.")
        analyze_and_place_options_trade(stock_symbol, stock_price, qty)

def confirm_trade_placement(order_id):
    """
    Confirm if the trade was successfully placed.
    It checks the status of the order by order_id.
    """
    try:
        # Retrieve the order details using the order ID
        order = trading_client.get_order_by_id(order_id)
        logging.info(f"Order Status for {order.symbol}: {order.status}")
        
        # Check if the order is filled, partially filled, or still open
        if order.status == "filled":
            logging.info(f"Trade for {order.symbol} was successfully filled.")
        elif order.status == "partially_filled":
            logging.info(f"Trade for {order.symbol} was partially filled.")
        else:
            logging.warning(f"Trade for {order.symbol} is still {order.status}.")
        return order.status

    except Exception as e:
        logging.error(f"Error checking trade status for order ID {order_id}: {e}")
        return None

def countdown_timer(interval):
    for remaining in range(interval, 0, -1):
        sys.stdout.write(f"\rNext refresh in {remaining} seconds...")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rNext refresh in 0 seconds...    \n")
    sys.stdout.flush()

def continuous_trading(qty=1, interval=180):
    while True:
        logging.info("\n--- Starting new trading cycle ---")
        stock_list = get_top_active_movers()

        evaluated_stocks = set()

        for stock in stock_list:
            if stock['symbol'] not in evaluated_stocks:
                try:
                    logging.info(f"Symbol: {stock['symbol']}, Change %: {stock['change_percent']}")
                    automated_trading(stock['symbol'], qty, evaluated_stocks)
                except Exception as e:
                    logging.error(f"Error trading {stock['symbol']}: {e}")

        countdown_timer(interval)

if __name__ == "__main__":
    trade_data = load_trade_data()
    model = train_model(trade_data)
    continuous_trading(qty=1, interval=180)