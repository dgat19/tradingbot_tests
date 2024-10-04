from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
import time
import sys
from joblib import load
from news_scraper import get_top_active_movers, get_trending_stocks
from swing_trader import manage_swing_trades
from indicators import analyze_indicators
from ml_trade_performance_evaluation import train_or_load_model, preprocess_data, load_trade_data
from common_functions import get_stock_info

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "YOUR_ALPACA_API_KEY"
ALPACA_API_SECRET = "YOUR_ALPACA_API_SECRET"

# Initialize the Trading Client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
# Initialize the Data Client
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
    print(f"Options buying power: {account.options_buying_power}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")

# Global variable to track open positions
open_positions = {}

# Load the ML model and feature columns
try:
    model = load('trade_model.pkl')
    scaler = load('scaler.pkl')
    training_feature_columns = load('training_feature_columns.pkl')
    print("Loaded existing model, scaler, and feature columns.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print("Training a new model.")
    # Load trade data and train the model
    trade_data = load_trade_data()
    X, y, feature_columns = preprocess_data(trade_data)
    model, scaler, training_feature_columns = train_or_load_model(X, y, feature_columns)

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
def place_option_trade(contract_symbol, qty):
    try:
        # Create an order request for an option
        order_data = OrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            asset_class=AssetClass.OPTION
        )
        # Place the options order
        order = trading_client.submit_order(order_data)
        print(f"Order placed for {contract_symbol}")
        return order
    except Exception as e:
        print(f"Error placing order for {contract_symbol}: {e}")
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
    options_chain = get_options_chain(stock_symbol)
    if options_chain is not None:
        if day_change > 3 and indicators['positive_trend']:
            # Positive trend, buy CALL option
            target_strike = stock_info['price'] * 1.1
            suitable_options = options_chain[
                (options_chain['type'] == 'call') &
                (options_chain['strike'] >= target_strike)
            ]
        elif day_change < -3 and not indicators['positive_trend']:
            # Negative trend, buy PUT option
            target_strike = stock_info['price'] * 0.9
            suitable_options = options_chain[
                (options_chain['type'] == 'put') &
                (options_chain['strike'] <= target_strike)
            ]
        else:
            print(f"Skipping {stock_symbol}. Indicators do not confirm the trend.")
            return

        if not suitable_options.empty:
            # Select the first suitable option
            option_contract = suitable_options.iloc[0]
            contract_symbol = option_contract['symbol']
            print(f"Placing order for {contract_symbol}")
            place_option_trade(contract_symbol, qty)
        else:
            print(f"No suitable options found for {stock_symbol}. Switching to swing trading.")
            manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)
    else:
        print(f"No options data available for {stock_symbol}. Switching to swing trading.")
        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)

# Trade logic for dynamically fetched stocks
def trade_dynamic_stocks(stock_symbol, qty=1):
    stock_info = get_stock_info(stock_symbol)
    current_price = stock_info['price']
    day_change = stock_info['day_change']
    volume = stock_info['volume']
    avg_volume = stock_info['avg_volume']
    indicators = analyze_indicators(stock_symbol)

    # Trade if current volume is higher than the average volume
    if volume <= avg_volume:
        print(f"Skipping {stock_symbol} due to insufficient volume. Volume: {int(volume):,}, Avg Volume: {int(avg_volume):,}")
        return

    options_chain = get_options_chain(stock_symbol)
    if options_chain is not None:
        if indicators['positive_trend']:
            # Positive trend, buy CALL option
            target_strike = current_price * 1.1
            suitable_options = options_chain[
                (options_chain['type'] == 'call') &
                (options_chain['strike'] >= target_strike)
            ]
        elif not indicators['positive_trend']:
            # Negative trend, buy PUT option
            target_strike = current_price * 0.9
            suitable_options = options_chain[
                (options_chain['type'] == 'put') &
                (options_chain['strike'] <= target_strike)
            ]
        else:
            print(f"{stock_symbol} - Trend and price movement do not align for trading. Skipping.")
            return

        if not suitable_options.empty:
            # Select the first suitable option
            option_contract = suitable_options.iloc[0]
            contract_symbol = option_contract['symbol']
            print(f"Placing order for {contract_symbol}")
            place_option_trade(contract_symbol, qty)
        else:
            print(f"No suitable options found for {stock_symbol}. Switching to swing trading.")
            manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)
    else:
        print(f"No options data available for {stock_symbol}. Switching to swing trading.")
        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)

# Function to fetch the options chain for a stock symbol
def get_options_chain(stock_symbol):
    try:
        # Set up the request parameters with the correct field name
        chain_request = OptionChainRequest(
            underlying_symbol=stock_symbol,
            limit=1000
        )

        # Fetch options contracts using the Data Client
        options_chain = data_client.get_option_chain(chain_request)

        if options_chain.df.empty:
            print(f"No options available for {stock_symbol}.")
            return None
        else:
            print(f"Options data retrieved for {stock_symbol}")
            return options_chain.df  # Return the DataFrame
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

                # Get options chain
                options_chain = get_options_chain(stock_symbol)
                if options_chain is not None:
                    # Determine option type and target strike price
                    if day_change > 0 and indicators['positive_trend']:
                        # Positive trend, buy CALL option
                        target_strike = current_price * 1.1
                        suitable_options = options_chain[
                            (options_chain['type'] == 'call') &
                            (options_chain['strike'] >= target_strike)
                        ]
                    elif day_change < 0 and not indicators['positive_trend']:
                        # Negative trend, buy PUT option
                        target_strike = current_price * 0.9
                        suitable_options = options_chain[
                            (options_chain['type'] == 'put') &
                            (options_chain['strike'] <= target_strike)
                        ]
                    else:
                        print(f"{stock_symbol} - Trend and price movement do not align for trading. Skipping.")
                        continue

                    if not suitable_options.empty:
                        # Select the first suitable option
                        option_contract = suitable_options.iloc[0]
                        contract_symbol = option_contract['symbol']
                        place_option_trade(contract_symbol, qty)
                    else:
                        print(f"No suitable options found for {stock_symbol}. Switching to swing trading.")
                        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)
                else:
                    print(f"No options data available for {stock_symbol}. Switching to swing trading.")
                    manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)
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
