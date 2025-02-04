from alpaca.trading.client import TradingClient, OrderRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, AssetClass
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame
import time
import sys
from joblib import load
import data.news_scraper
from swing_trader import manage_swing_trades
from src.indicators import analyze_indicators
from models.ml_trade_evaluation import MLTradeEvaluator
from utils.common_functions import get_stock_info, get_current_stock_data

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PK2NIF6EZ927PKU3L4AC"
ALPACA_API_SECRET = "dPlhgkNQJ3RaD0CvKZEaYua4ku6wtKaBtp8uONAN"

# Initialize the Trading Client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
# Initialize the Data Client
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# Verify Alpaca connection
def verify_connection():
    try:
        account = trading_client.get_account()
        print(f"Account status: {account.status}")
        print(f"Account balance: {account.cash}")
        print(f"Options buying power: {account.options_buying_power}")
    except Exception as e:
        print(f"Error connecting to Alpaca API: {e}")

verify_connection()

# Global variable to track open positions
open_positions = {}

# Load the ML model and feature columns
def load_model():
    try:
        model = load('trade_model.pkl')
        scaler = load('scaler.pkl')
        feature_columns = load('training_feature_columns.pkl')
        print("Loaded existing model, scaler, and feature columns.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Training a new model.")
        # Load trade data and train the model
        trade_data = MLTradeEvaluator()
        X, y, feature_columns = MLTradeEvaluator(trade_data)
        model, scaler, feature_columns = MLTradeEvaluator(X, y, feature_columns)
    return model, scaler, feature_columns

model, scaler, training_feature_columns = load_model()

# Hardcoded stock list
hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']

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
    if options_chain is None:
        print(f"No options data available for {stock_symbol}. Switching to swing trading.")
        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)
        return

    suitable_options = get_suitable_options(stock_info, options_chain, day_change, indicators)
    if not suitable_options.empty:
        # Select the first suitable option
        option_contract = suitable_options.iloc[0]
        contract_symbol = option_contract['symbol']
        print(f"Placing order for {contract_symbol}")
        place_option_trade(contract_symbol, qty)
    else:
        print(f"No suitable options found for {stock_symbol}. Switching to swing trading.")
        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)

# Get suitable options based on trend
def get_suitable_options(stock_info, options_chain, day_change, indicators):
    if day_change > 3 and indicators['positive_trend']:
        # Positive trend, buy CALL option
        target_strike = stock_info['price'] * 1.1
        return options_chain[(options_chain['type'] == 'call') & (options_chain['strike'] >= target_strike)]
    elif day_change < -3 and not indicators['positive_trend']:
        # Negative trend, buy PUT option
        target_strike = stock_info['price'] * 0.9
        return options_chain[(options_chain['type'] == 'put') & (options_chain['strike'] <= target_strike)]
    else:
        print(f"Skipping {stock_info['symbol']}. Indicators do not confirm the trend.")
        return options_chain.iloc[0:0]  # Return empty DataFrame

# Function to fetch the options chain for a stock symbol
def get_options_chain(stock_symbol):
    try:
        chain_request = OptionChainRequest(underlying_symbol=stock_symbol, limit=1000)
        options_chain = data_client.get_option_chain(chain_request)
        if options_chain.df.empty:
            print(f"No options available for {stock_symbol}.")
            return None
        else:
            print(f"Options data retrieved for {stock_symbol}")
            return options_chain.df
    except Exception as e:
        print(f"Error fetching options data for {stock_symbol}: {e}")
        return None

# Main continuous trading loop
def continuous_trading(qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")

        # Fetch stocks for trading
        stock_list = get_current_stock_data() + data.news_scraper.StockDiscoveryAndAnalysis()

        for stock in stock_list:
            stock_symbol = stock['symbol']

            # Fetch stock data
            stock_info = get_stock_info(stock_symbol)
            volume = stock_info['volume']
            avg_volume = stock_info['avg_volume']

            # Make a decision to trade or skip based on volume
            if volume > avg_volume:
                trade_dynamic_stocks(stock_symbol, qty)
            else:
                print(f"Skipping {stock_symbol} due to insufficient volume. Volume: {int(volume):,}, Avg Volume: {int(avg_volume):,}")
            print("-" * 100)

        # Countdown timer before the next trading cycle
        print("\nWaiting for the next cycle...")
        countdown(interval)

# Trade logic for dynamically fetched stocks
def trade_dynamic_stocks(stock_symbol, qty=1):
    stock_info = get_stock_info(stock_symbol)
    day_change = stock_info['day_change']
    indicators = analyze_indicators(stock_symbol)
    options_chain = get_options_chain(stock_symbol)

    if options_chain is None:
        print(f"No options data available for {stock_symbol}. Switching to swing trading.")
        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)
        return

    suitable_options = get_suitable_options(stock_info, options_chain, day_change, indicators)
    if not suitable_options.empty:
        # Select the first suitable option
        option_contract = suitable_options.iloc[0]
        contract_symbol = option_contract['symbol']
        print(f"Placing order for {contract_symbol}")
        place_option_trade(contract_symbol, qty)
    else:
        print(f"No suitable options found for {stock_symbol}. Switching to swing trading.")
        manage_swing_trades([stock_symbol], qty, model, scaler, training_feature_columns)

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