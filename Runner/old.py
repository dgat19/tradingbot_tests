import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import time
import sys
import contextlib
import os
import pytz
from datetime import datetime, timedelta
from news_scraper import get_top_active_movers, get_trending_stocks
from indicators import analyze_indicators, get_stock_volatility
from ml_trade_performance_evaluation import load_trade_data, train_model, preprocess_data

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKOS59NZIX9P0L8VNJK8"
ALPACA_API_SECRET = "7eFxPPP15H3eYW3V0B5i1UdV21sCbS9oW1L4WcoA"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
    print(f"Account cash withdrawal: {account.options_buying_power}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")

# Dictionary to keep track of open positions (key: contract symbol, value: dict with entry price and qty)
open_positions = {}
print(f"Open positions: {open_positions}")

# Load trade data and train the model
trade_data = load_trade_data()
model = train_model(trade_data)  # Train the machine learning model on historical data
_, _, scaler = preprocess_data(trade_data)  # Preprocess to get scaler for new data

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_open_price_at_market_open(symbol):
    try:
        stock = yf.Ticker(symbol)
        ny_tz = pytz.timezone('America/New_York')  # U.S. Eastern Timezone
        current_time = datetime.now(ny_tz).replace(hour=9, minute=30, second=0, microsecond=0)
        history = stock.history(period="1d", interval="1m")
        
        if history.empty:
            raise ValueError(f"No market data found for {symbol} at 9:30 AM")

        open_price = history.loc[history.index == current_time]['Open'].values
        
        if len(open_price) == 0:
            raise ValueError(f"No data for {symbol} at 9:30 AM")
        
        return open_price[0]
    except Exception as e:
        print(f"Error retrieving open price at market open for {symbol}: {e}")
        return None

def check_buying_power(required_buying_power):
    try:
        account = trading_client.get_account()
        buying_power = float(account.cash)
        
        if buying_power >= required_buying_power:
            return True
        else:
            print(f"Insufficient buying power. Available: {buying_power}, Required: {required_buying_power}")
            return False
    except Exception as e:
        print(f"Error checking buying power: {e}")
        return False

def get_current_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        with suppress_stdout():
            hist = stock.history(period="1d")
        
        if hist.empty:
            print(f"No data available for {symbol}")
            return None, None, None, None  # Ensure 4 values are returned
        
        current_price = hist['Close'].iloc[-1]
        open_price = get_open_price_at_market_open(symbol)  # Get the 8:30 AM UTC open price
        current_volume = hist['Volume'].iloc[-1]
        volatility = get_stock_volatility(symbol)

        if current_price is None or open_price is None:
            raise ValueError(f"Invalid data for {symbol}: current_price or open_price is None")

        return current_price, open_price, current_volume, volatility
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None, None, None, None

def get_options_chain(symbol):
    try:
        stock = yf.Ticker(symbol)
        options_dates = stock.options
        
        if not options_dates:
            print(f"No options available for {symbol}")
            return None
        
        today = datetime.now()
        one_month_later = today + timedelta(days=30)
        valid_dates = [date for date in options_dates if datetime.strptime(date, "%Y-%m-%d") <= one_month_later]
        
        if not valid_dates:
            print(f"No valid options dates within a month for {symbol}")
            return None
        
        latest_valid_date = max(valid_dates)
        options_chain = stock.option_chain(latest_valid_date)
        
        # Ensure 'calls' and 'puts' exist
        if hasattr(options_chain, 'calls') and hasattr(options_chain, 'puts'):
            return options_chain
        else:
            print(f"No 'calls' or 'puts' data available for {symbol} on {latest_valid_date}")
            return None
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {str(e)}")
        return None

def enter_trade(contract_symbol, entry_price, qty):
    """
    Function to enter a trade and store the position.
    """
    global open_positions
    open_positions[contract_symbol] = {
        'entry_price': entry_price,
        'qty': qty
    }
    print(f"Entered trade for {contract_symbol} at {entry_price} with quantity {qty}")
    # Place the order using Alpaca here if needed
    return True

def check_exit_conditions(contract_symbol):
    """
    Function to check if we need to exit the trade based on stop-loss or take-profit conditions.
    The decision will now be based on the option contract's unrealized gain/loss from Alpaca.
    """
    try:
        global open_positions

        # Check if we have an open position for this contract
        if contract_symbol not in open_positions:
            print(f"No open position found for {contract_symbol}")
            return

        # Fetch position data from Alpaca
        position = trading_client.get_position(contract_symbol)
        if not position:
            print(f"No position data found for {contract_symbol}")
            return

        # Fetch unrealized P&L (gain/loss) # Total unrealized profit/loss in USD
        unrealized_plpc = float(position.unrealized_plpc)  # Unrealized profit/loss as a percentage

        # Exit trade based on stop-loss or take-profit percentages
        stop_loss_threshold = -0.25  # Example: exit if position has lost 25%
        take_profit_threshold = 0.75  # Example: exit if position has gained 75%

        if unrealized_plpc <= stop_loss_threshold:
            print(f"{contract_symbol} - Stop-loss triggered. Unrealized loss: {unrealized_plpc*100:.2f}%. Exiting trade.")
            exit_trade(contract_symbol, position.qty)
            return

        if unrealized_plpc >= take_profit_threshold:
            print(f"{contract_symbol} - Take-profit triggered. Unrealized gain: {unrealized_plpc*100:.2f}%. Exiting trade.")
            exit_trade(contract_symbol, position.qty)
            return

    except Exception as e:
        print(f"Error checking exit conditions for {contract_symbol}: {str(e)}")

def exit_trade(contract_symbol, qty):
    """
    Function to exit a trade and remove the position from the open positions.
    """
    try:
        global open_positions

        # Remove the position from the open positions tracking
        if contract_symbol in open_positions:
            del open_positions[contract_symbol]

        # Place the sell order via Alpaca
        order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        print(f"Exited trade for {contract_symbol} with quantity {qty}")
        return order
    except Exception as e:
        print(f"Error exiting trade for {contract_symbol}: {str(e)}")
        return None


def place_option_trade(contract_symbol, qty, option_type='call'):
    try:
        # Fetch the option's current price dynamically
        options_chain = get_options_chain(contract_symbol[:4])  # Retrieve options chain with a valid function
        option_data = None

        if options_chain:
            if option_type == 'call' and not options_chain.calls.empty:
                option_data = options_chain.calls[options_chain.calls['contractSymbol'] == contract_symbol]
            elif option_type == 'put' and not options_chain.puts.empty:
                option_data = options_chain.puts[options_chain.puts['contractSymbol'] == contract_symbol]

        if option_data is None or option_data.empty:
            print(f"No data available for {contract_symbol}")
            return None

        # Get the current price for the option
        option_price = option_data['lastPrice'].iloc[0]

        # Calculate the required buying power (contract size is typically 100)
        required_buying_power = option_price * qty * 100

        # Check if buying power is sufficient
        if not check_buying_power(required_buying_power):
            print(f"Skipping trade for {contract_symbol} due to insufficient buying power. Required: {required_buying_power}")
            return None

        # Place the order using Alpaca
        order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)

        # Track the trade in open positions
        enter_trade(contract_symbol, option_price, qty)

        print(f"Order placed for {contract_symbol} at option price: {option_price}")
        return order

    except Exception as e:
        print(f"Error placing order for {contract_symbol}: {e}")
        return None

def trade_hardcoded_stocks(stock_symbol, qty=1):
    hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']
    current_price, open_price, current_volume, volatility = get_current_stock_data(stock_symbol)
    if current_price is None:
        print(f"Skipping {stock_symbol} due to data retrieval error")
        return

    price_change = ((current_price - open_price) / open_price) * 100

    # Use ML model to help predict the trend
    ml_prediction = model.predict([[current_price, open_price, current_volume, volatility]])[0]
    print(f"ML prediction for {stock_symbol}: {ml_prediction}")

    # If stock is in hardcoded list, follow the ±3% price movement rule combined with ML prediction
    if stock_symbol in hardcoded_stocks and abs(price_change) >= 3:
        options_chain = get_options_chain(stock_symbol)
        if price_change > 3 and ml_prediction == 1:  # ML indicates upward trend
            # Buy CALL options with a target strike 10% above current price for upward trend
            target_call_strike = current_price * 1.1
            itm_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
            if not itm_calls.empty:
                option_contract = itm_calls.iloc[0]  # Closest in-the-money call
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - +3% price movement detected with positive ML trend. Placing CALL order for {contract_symbol}")
                place_option_trade(contract_symbol, qty, option_type='call')
            else:
                print(f"{stock_symbol} - No suitable ITM CALL options found")
        elif price_change < -3 and ml_prediction == -1:  # ML indicates downward trend
            # Buy PUT options with a target strike 10% below current price for downward trend
            target_put_strike = current_price * 0.9
            itm_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
            if not itm_puts.empty:
                option_contract = itm_puts.iloc[0]  # Closest in-the-money put
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - -3% price movement detected with negative ML trend. Placing PUT order for {contract_symbol}")
                place_option_trade(contract_symbol, qty, option_type='put')
            else:
                print(f"{stock_symbol} - No suitable ITM PUT options found")
        else:
            print(f"{stock_symbol} - No significant price movement (3%) or ML did not confirm trend")

def trade_dynamic_stocks(stock_symbol, qty=1):
    # Separate function for dynamic stocks (trending and active movers)
    current_price, open_price, current_volume, volatility = get_current_stock_data(stock_symbol)
    if current_price is None:
        print(f"Skipping {stock_symbol} due to data retrieval error")
        return

    indicators = analyze_indicators(stock_symbol)
    options_chain = get_options_chain(stock_symbol)

    # Use ML prediction for dynamic stocks as well
    ml_prediction = model.predict([[current_price, open_price, current_volume, volatility]])[0]
    print(f"ML prediction for {stock_symbol}: {ml_prediction}")

    if indicators['positive_trend'] and ml_prediction == 1:
        # Buy CALL options with a target strike 10% above current price for upward trend
        target_call_strike = current_price * 1.1
        itm_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
        if not itm_calls.empty:
            option_contract = itm_calls.iloc[0]  # Closest in-the-money call
            contract_symbol = option_contract['contractSymbol']
            print(f"{stock_symbol} - Positive trend detected with ML support. Placing CALL order for {contract_symbol}")
            place_option_trade(contract_symbol, qty, option_type='call')
        else:
            print(f"{stock_symbol} - No suitable ITM CALL options found")
    elif not indicators['positive_trend'] and ml_prediction == -1:
        # Buy PUT options with a target strike 10% below current price for downward trend
        target_put_strike = current_price * 0.9
        itm_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
        if not itm_puts.empty:
            option_contract = itm_puts.iloc[0]  # Closest in-the-money put
            contract_symbol = option_contract['contractSymbol']
            print(f"{stock_symbol} - Negative trend detected with ML support. Placing PUT order for {contract_symbol}")
            place_option_trade(contract_symbol, qty, option_type='put')
        else:
            print(f"{stock_symbol} - No suitable ITM PUT options found")
    else:
        print(f"{stock_symbol} - No significant action or ML did not confirm trend")

# Continuous trading function to handle both hardcoded and dynamic stocks separately
def continuous_trading(qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")
        
        # Fetch stocks from get_top_active_movers and get_trending_stocks
        stock_list = get_top_active_movers() + get_trending_stocks()
        
        # Trade hardcoded stocks separately
        hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']  # Exclude RIVN
        for stock_symbol in hardcoded_stocks:
            print(f"Checking hardcoded stock: {stock_symbol}")
            trade_hardcoded_stocks(stock_symbol, qty)
            print("----------------------------------------------------------------------------------------------------")
        
        # Trade dynamically fetched stocks separately
        for stock in stock_list:
            stock_symbol = stock['symbol']
            if stock_symbol not in hardcoded_stocks:  # Skip hardcoded stocks in this loop
                try:
                    print(f"Symbol: {stock['symbol']}, Change %: {stock['change_percent']}, Volume: {stock['volume']}, Avg Volume: {stock['avg_volume']}")
                    trade_dynamic_stocks(stock_symbol, qty)
                    print("----------------------------------------------------------------------------------------------------")
                except Exception as e:
                    print(f"Error trading {stock_symbol}: {e}")
        
        # Check exit conditions for all open positions
        for contract_symbol in list(open_positions.keys()):  # List is used to prevent mutation during iteration
            check_exit_conditions(contract_symbol)
        
        print(f"\nWaiting {interval} seconds before next cycle...")
        for remaining in range(interval, 0, -1):
            sys.stdout.write(f"\rNext refresh in {remaining} seconds...   ")
            sys.stdout.flush()
            time.sleep(1)
        
        sys.stdout.write("\rNext refresh in 0 seconds...    \n")
        sys.stdout.flush()

if __name__ == "__main__":
    trade_data = load_trade_data()
    model = train_model(trade_data)
    
    continuous_trading(qty=1, interval=180)
