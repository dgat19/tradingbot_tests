import yfinance as yf
import pandas as pd
import warnings
from joblib import load
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from common_functions import get_stock_info
from indicators import calculate_rsi

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKV1PSBFZJSVP0SVHZ7U"
ALPACA_API_SECRET = "vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

open_positions = {}

try:
    model = load('trade_model.pkl')
    scaler = load('scaler.pkl')
    training_feature_columns = load('training_feature_columns.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    # Handle the error (e.g., exit the script or retrain the model)


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

def create_features(data):
    # Calculate technical indicators
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data)
    data = data.dropna()
    features = data[['MA_10', 'MA_50', 'RSI']]
    return features


# Function to execute a swing trade based on the machine learning model prediction
def swing_trade_stock(stock_symbol, qty):
    print(f"Attempting swing trade: Buying {stock_symbol}")

    # Ensure `model`, `scaler`, and `training_feature_columns` are accessible
    global model, scaler, training_feature_columns

    # Fetch historical data for the stock
    data = yf.download(stock_symbol, period='1y', interval='1d')

    if data.empty:
        print(f"No data available for {stock_symbol}. Skipping.")
        return

    # Preprocess the data to create features
    X_new = create_features(data)

    # Ensure that we have all the necessary features
    missing_cols = set(training_feature_columns) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0  # Or another appropriate default value

    # Reorder columns to match the training data
    X_new = X_new[training_feature_columns]

    # Drop rows with missing values if necessary
    X_new = X_new.dropna()

    if X_new.empty:
        print(f"No data available for prediction for {stock_symbol}. Skipping.")
        return

    # Scale the features using the scaler fitted during training
    X_new_scaled_array = scaler.transform(X_new)

    # Convert the scaled array back to a DataFrame with the correct column names
    X_new_scaled = pd.DataFrame(X_new_scaled_array, columns=training_feature_columns)

    # Predict using the model
    predicted_return = model.predict(X_new_scaled)

    # Assuming the model predicts binary outcomes (1 for profit, 0 for loss)
    if predicted_return[-1] == 1:
        # Place the swing trade
        print(f"Placing swing trade for {stock_symbol}")
        # Implement the code to place the trade using Alpaca's API
        # ... [code to place the order] ...
    else:
        print(f"Skipping swing trade for {stock_symbol} due to predicted negative return.")



# Function to manage swing trades (only trade stocks with positive trend)
def manage_swing_trades(stock_list, qty, model, scaler):
    for stock_symbol in stock_list:
        current_price, open_price, current_volume, avg_volume = get_current_stock_data(stock_symbol)
        
        if current_price is None:
            print(f"Skipping {stock_symbol} due to data retrieval error")
            continue

        print(f"Attempting swing trade: Buying {stock_symbol} at {current_price}")

        # Prepare features for prediction (ensure feature order matches training)
        features = pd.DataFrame([{
            'price_at_trade': current_price,
            'volatility': 0.3,  # Set volatility or retrieve it using a proper function
            'volume': current_volume,
            'avg_volume': avg_volume
        }])

        # Ensure columns match the order and names used during training
        feature_columns = ['price_at_trade', 'volatility', 'volume', 'avg_volume']  
        features = features[feature_columns]  # Ensure the correct order

        # Scale the features
        scaled_features = scaler.transform(features)

        # Use the model to predict the outcome
        predicted_outcome = model.predict(scaled_features)[0]

        if predicted_outcome == 1:  # Model predicts profit
            print(f"Placing swing trade for {stock_symbol}")
            swing_trade_stock(stock_symbol, qty)
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
