import yfinance as yf
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from options_trader import get_stock_info

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKV1PSBFZJSVP0SVHZ7U"
ALPACA_API_SECRET = "vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

open_positions = {}

def get_current_stock_data(stock_symbol):
    """
    Fetches the current stock data including price, open price, volume, and average volume.
    """
    try:
        # Get stock data from Yahoo Finance
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="1d", interval="1d")

        if stock_data.empty:
            print(f"No data available for {stock_symbol}")
            return None, None, None, None

        # Extract the current stock price, open price, current volume, and calculate average volume over a month
        current_price = stock_data['Close'].iloc[-1]
        open_price = stock_data['Open'].iloc[-1]
        current_volume = stock_data['Volume'].iloc[-1]

        # Fetch monthly data to calculate the average volume over the last 30 days
        historical_data = stock.history(period="1mo", interval="1d")
        avg_volume = historical_data['Volume'].mean()

        return current_price, open_price, current_volume, avg_volume

    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        return None, None, None, None

# Function to execute a swing trade based on the machine learning model prediction
def swing_trade_stock(stock_symbol, qty, model, scaler):
    try:
        current_price, open_price, current_volume, avg_volume = get_current_stock_data(stock_symbol)
        if current_price is None:
            return None

        print(f"Attempting swing trade: Buying {stock_symbol} at {current_price}")

        # Check if account has enough buying power
        account = trading_client.get_account()
        required_buying_power = qty * current_price
        available_cash = float(account.cash)

        if available_cash < required_buying_power:
            print(f"Insufficient buying power for swing trade of {stock_symbol}. Available: {available_cash}, Required: {required_buying_power}")
            return None

        # Prepare input for the ML model with all relevant features
        stock_data = pd.DataFrame([{
            'price_at_trade': current_price,
            'volatility': get_stock_volatility(stock_symbol),
            'volume': current_volume,
            'avg_volume': avg_volume
        }])

        stock_data_scaled = scaler.transform(stock_data)

        # Make prediction
        predicted_outcome = model.predict(stock_data_scaled)[0]
        print(f"ML prediction for {stock_symbol}: {predicted_outcome}")

        # If prediction indicates a positive outcome (e.g., 1), proceed with buying
        if predicted_outcome == 1:
            order_data = MarketOrderRequest(
                symbol=stock_symbol,
                qty=qty,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC  # Good till canceled for swing trades
            )
            order = trading_client.submit_order(order_data)
            print(f"Swing trade order placed for {stock_symbol} at {current_price}")
            open_positions[stock_symbol] = current_price  # Record the purchase price
            return order
        else:
            print(f"Skipping swing trade for {stock_symbol} due to predicted negative outcome.")
            return None

    except Exception as e:
        print(f"Error placing swing trade for {stock_symbol}: {e}")
        return None

# Function to manage swing trades (only trade stocks with positive trend)
def manage_swing_trades(stock_list, qty, model, scaler):
    for stock_symbol in stock_list:
        current_price, open_price, current_volume, avg_volume = get_current_stock_data(stock_symbol)
        
        if current_price is None:
            print(f"Skipping {stock_symbol} due to data retrieval error")
            continue

        print(f"Attempting swing trade: Buying {stock_symbol} at {current_price}")

        # Prepare features for prediction
        features = pd.DataFrame([{
            'price_at_trade': current_price,
            'volatility': get_stock_volatility(stock_symbol),
            'volume': current_volume,
            'avg_volume': avg_volume
        }])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Use the model to predict the outcome
        predicted_outcome = model.predict(scaled_features)[0]

        if predicted_outcome == 1:  # Model predicts profit
            print(f"Placing swing trade for {stock_symbol}")
            swing_trade_stock(stock_symbol, qty, model, scaler)
        else:
            print(f"Skipping swing trade for {stock_symbol} due to predicted negative return.")

# Function to check exit conditions (sell the stock)
def check_exit_conditions(stock_symbol):
    global open_positions
    try:
        stock_info = get_stock_info(stock_symbol)  # Fetch stock data
        current_price = stock_info['price']

        # Example logic: If stock price rises 10% from the purchase price, sell
        purchase_price = open_positions[stock_symbol]  # Assuming the purchase price is stored here
        if current_price >= purchase_price * 1.1:
            print(f"Selling {stock_symbol} as it hit the 10% profit target.")
            sell_stock(stock_symbol)

    except Exception as e:
        print(f"Error checking exit conditions for {stock_symbol}: {e}")

# Function to sell the stock
def sell_stock(stock_symbol):
    global open_positions
    try:
        qty = open_positions[stock_symbol]  # Get the quantity to sell
        order_data = MarketOrderRequest(
            symbol=stock_symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order_data)
        print(f"Sold {qty} shares of {stock_symbol}")
        del open_positions[stock_symbol]  # Remove from open positions after selling
    except Exception as e:
        print(f"Error selling {stock_symbol}: {e}")

# Utility function to fetch stock volatility
def get_stock_volatility(stock_symbol):
    stock_data = yf.download(stock_symbol, period="1mo", interval="1d")
    if stock_data.empty:
        return None
    close_prices = stock_data['Close'].values
    volatility = close_prices.std()
    return volatility
