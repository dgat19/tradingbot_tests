from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import yfinance as yf
import time
import sys
from datetime import datetime, timedelta
from news_scraper import get_top_active_movers, get_trending_stocks
from swing_trader import manage_swing_trades, swing_trade_stock
from indicators import analyze_indicators
from ml_trade_performance_evaluation import train_or_load_model, preprocess_data, load_trade_data
from common_functions import get_stock_info

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKV1PSBFZJSVP0SVHZ7U"
ALPACA_API_SECRET = "vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
    print(f"Account cash withdrawal: {account.options_buying_power}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")

# Global variable to track open positions
open_positions = {}

# Load the ML model and preprocess the trade data
trade_data = load_trade_data()
X, X_scaled, y, scaler = preprocess_data(trade_data)  # Preprocess data
model = train_or_load_model(X, y)

# Hardcoded stock list
hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']

# Function to check if PDT rule is triggered
def check_pdt_status():
    account = trading_client.get_account()
    if account.daytrade_count >= 3:
        print(f"PDT limit reached with {account.daytrade_count} day trades. Switching to swing trading.")
        return False
    return True

# Function to place an options trade using Alpaca
def place_option_trade(stock_symbol, qty, current_price, day_change, indicators):
    try:
        # Fetch the options chain for the stock symbol
        options_chain = get_options_chain(stock_symbol)

        if options_chain is None:
            print(f"No options data available for {stock_symbol}")
            return None

        # Check for trend alignment and determine option type (call or put)
        if day_change > 0 and indicators['positive_trend']:
            # Positive trend and day change: place a CALL option
            target_strike = current_price * 1.1  # 10% above current price
            itm_options = options_chain.calls[options_chain.calls['strike'] >= target_strike]
        elif day_change < 0 and not indicators['positive_trend']:
            # Negative trend and day change: place a PUT option
            target_strike = current_price * 0.9  # 10% below current price
            itm_options = options_chain.puts[options_chain.puts['strike'] <= target_strike]
        else:
            # If trend and day change do not align, skip the trade
            print(f"{stock_symbol} - Trend and price movement do not align for trading. Skipping.")
            return None

        # Check if we have any matching in-the-money options
        if itm_options.empty:
            print(f"No matching options data for {stock_symbol}.")
            return None

        # Select the first available option contract
        option_contract = itm_options.iloc[0]
        contract_symbol = option_contract['contractSymbol']
        option_price = option_contract['lastPrice']

        # Calculate required buying power (contract size is typically 100)
        required_buying_power = option_price * qty * 100

        if not check_buying_power(required_buying_power):
            print(f"Skipping trade for {contract_symbol} due to insufficient buying power. Required: {required_buying_power}")
            return None

        # Place the options order
        order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)

        enter_trade(contract_symbol, option_price, qty)
        print(f"Order placed for {contract_symbol} at option price: {option_price}")
        return order

    except Exception as e:
        print(f"Error placing order for {stock_symbol}: {e}")
        return None

# Function to track an open trade
def enter_trade(contract_symbol, entry_price, qty):
    global open_positions
    open_positions[contract_symbol] = {
        'entry_price': entry_price,
        'qty': qty
    }
    print(f"Entered trade for {contract_symbol} at {entry_price} with quantity {qty}")
    return True

# Trade logic for hardcoded stocks (only trade if day change is ±3% or more)
def trade_hardcoded_stocks(stock_symbol, qty=1):
    stock_info = get_stock_info(stock_symbol)
    day_change = stock_info['day_change']
    
    # Only trade if the day change is ±3%
    if abs(day_change) < 3:
        print(f"Skipping {stock_symbol} due to insufficient day change (±3%). Day change: {day_change:.2f}%")
        return
    
    indicators = analyze_indicators(stock_symbol)
    if day_change > 3 and indicators['positive_trend']:  # Buy CALL options
        options_chain = get_options_chain(stock_symbol)
        if options_chain:
            target_call_strike = stock_info['price'] * 1.1
            itm_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
            if not itm_calls.empty:
                contract_symbol = itm_calls.iloc[0]['contractSymbol']
                print(f"{stock_symbol} - Day change +3%. Placing CALL order for {contract_symbol}")
                place_option_trade(contract_symbol, qty, option_type='call')
            else:
                print(f"{stock_symbol} - No suitable CALL options found.")
        else:
            print(f"{stock_symbol} - Switching to swing trading due to no options data.")
            swing_trade_stock(stock_symbol, qty, model, scaler)  # If no options available, switch to swing trading
    elif day_change < -3 and not indicators['positive_trend']:  # Buy PUT options
        options_chain = get_options_chain(stock_symbol)
        if options_chain:
            target_put_strike = stock_info['price'] * 0.9
            itm_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
            if not itm_puts.empty:
                contract_symbol = itm_puts.iloc[0]['contractSymbol']
                print(f"{stock_symbol} - Day change -3%. Placing PUT order for {contract_symbol}")
                place_option_trade(contract_symbol, qty, option_type='put')
            else:
                print(f"{stock_symbol} - No suitable PUT options found.")
        else:
            print(f"{stock_symbol} - Switching to swing trading due to no options data.")
            swing_trade_stock(stock_symbol, qty, model, scaler)  # If no options available, switch to swing trading
    else:
        print(f"Skipping {stock_symbol}. Day change is 3% but indicators did not confirm the trend.")

# Trade logic for dynamically fetched stocks
def trade_dynamic_stocks(stock_symbol, qty=1):
    stock_info = get_stock_info(stock_symbol)
    indicators = analyze_indicators(stock_symbol)
    
    # Trade if current volume is higher than the average volume
    if stock_info['volume'] <= stock_info['avg_volume']:
        print(f"Skipping {stock_symbol} due to insufficient volume. Volume: {int(stock_info['volume']):,}, Avg Volume: {int(stock_info['avg_volume']):,}")
        return

    if indicators['positive_trend']:  # Buy CALL options
        options_chain = get_options_chain(stock_symbol)
        if options_chain:
            target_call_strike = stock_info['price'] * 1.1
            itm_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
            if not itm_calls.empty:
                contract_symbol = itm_calls.iloc[0]['contractSymbol']
                print(f"{stock_symbol} - Positive trend detected. Placing CALL order for {contract_symbol}")
                place_option_trade(contract_symbol, qty, option_type='call')
            else:
                print(f"{stock_symbol} - No suitable CALL options found.")
                swing_trade_stock(stock_symbol, qty, model, scaler)  # Switch to swing trading
        else:
            print(f"{stock_symbol} - No options data available. Switching to swing trading.")
            swing_trade_stock(stock_symbol, qty, model, scaler)  # Switch to swing trading
    elif not indicators['positive_trend']:  # Buy PUT options
        options_chain = get_options_chain(stock_symbol)
        if options_chain:
            target_put_strike = stock_info['price'] * 0.9
            itm_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
            if not itm_puts.empty:
                contract_symbol = itm_puts.iloc[0]['contractSymbol']
                print(f"{stock_symbol} - Negative trend detected. Placing PUT order for {contract_symbol}")
                place_option_trade(contract_symbol, qty, option_type='put')
            else:
                print(f"{stock_symbol} - No suitable PUT options found.")
                swing_trade_stock(stock_symbol, qty, model, scaler)  # Switch to swing trading
        else:
            print(f"{stock_symbol} - No options data available. Switching to swing trading.")
            swing_trade_stock(stock_symbol, qty, model, scaler)  # Switch to swing trading
    else:
        print(f"{stock_symbol} - No options data available. Skipping...")

# Function to fetch the options chain for a stock symbol
def get_options_chain(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        options_dates = stock.options

        if not options_dates:
            print(f"No options available for {stock_symbol}.")
            return None

        # Get the current date and calculate the date one month from today
        current_date = datetime.now()
        one_month_later = current_date + timedelta(days=30)

        # Find an expiration date that is closest to one month from today
        valid_dates = [date for date in options_dates if datetime.strptime(date, "%Y-%m-%d") >= current_date]

        if not valid_dates:
            print(f"No valid expiration dates within the next month for {stock_symbol}.")
            return None

        # Use the closest valid expiration date that is at least one month away
        chosen_expiry = min(valid_dates, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - one_month_later))

        # Fetch the options chain for the chosen expiration date
        options_chain = stock.option_chain(chosen_expiry)
        print(f"Options data retrieved for {stock_symbol} with expiry {chosen_expiry}")
        return options_chain

    except Exception as e:
        print(f"Error fetching options data for {stock_symbol}: {e}")
        return None

# Function to check if buying power is sufficient
def check_buying_power(required_buying_power):
    account = trading_client.get_account()
    buying_power = float(account.cash)
    
    return buying_power >= required_buying_power

# Main continuous trading loop
def continuous_trading(qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")

        # Fetch stocks for trading
        stock_list = get_top_active_movers() + get_trending_stocks()

        for stock in stock_list:
            stock_symbol = stock['symbol']
            
            # Fetch stock data
            stock_info = get_stock_info(stock_symbol)
            current_price = stock_info['price']
            day_change = stock_info['day_change']
            volume = stock_info['volume']
            avg_volume = stock_info['avg_volume']
            
            # Fetch indicators
            indicators = analyze_indicators(stock_symbol)

            # Display stock info
            print(f"{stock_symbol} - Price: ${current_price:.2f}    "
                  f"Day Change: {day_change:.2f}%    "
                  f"Volume: {int(volume):,}    "
                  f"Avg. Volume: {int(avg_volume):,}")

            # Make a decision to trade or skip based on volume and indicators
            if volume > avg_volume and (day_change > 0 or day_change < 0):
                print(f"Placing options trade for {stock_symbol}")
                # Attempt to place an options trade
                if place_option_trade(stock_symbol, qty, current_price, day_change, indicators) is None:
                    print(f"Switching to swing trading for {stock_symbol} due to no options data.")
                    manage_swing_trades([stock_symbol], qty, model, scaler)  # Use swing trading as fallback
            else:
                print(f"Skipping {stock_symbol} due to volume or trend indicators.")
            print("-" * 100)

        # Countdown timer before the next trading cycle
        print("\nWaiting for the next cycle...")
        countdown(interval)


# Countdown timer function
def countdown(interval):
    for remaining in range(interval, 0, -1):
        sys.stdout.write(f"\rNext refresh in {remaining} seconds...   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rNext refresh in 0 seconds...    \n")
    sys.stdout.flush()

if __name__ == "__main__":
    continuous_trading(qty=1, interval=180)
