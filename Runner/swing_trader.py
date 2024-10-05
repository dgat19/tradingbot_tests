import yfinance as yf
import pandas as pd
import warnings
from joblib import load
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from common_functions import get_stock_info, get_current_stock_data, get_stock_volatility
from indicators import calculate_rsi

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKV1PSBFZJSVP0SVHZ7U"
ALPACA_API_SECRET = "vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Global variable to track open positions
open_positions = {}

# Load the ML model and feature columns
def load_model():
    try:
        model = load('trade_model.pkl')
        scaler = load('scaler.pkl')
        training_feature_columns = load('training_feature_columns.pkl')
        print("Loaded existing model, scaler, and feature columns.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        # Handle the error (e.g., exit the script or retrain the model)
        return None, None, None
    return model, scaler, training_feature_columns

model, scaler, training_feature_columns = load_model()

# Create features for prediction
def create_features(data):
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data)
    return data.dropna()[['MA_10', 'MA_50', 'RSI']]

# Execute a swing trade based on ML prediction
def swing_trade_stock(stock_symbol, qty):
    if model is None or scaler is None or training_feature_columns is None:
        print("Model or scaler not loaded. Cannot execute swing trade.")
        return

    data = yf.download(stock_symbol, period='1y', interval='1d')
    if data.empty:
        print(f"No data available for {stock_symbol}. Skipping.")
        return

    X_new = create_features(data)
    missing_cols = set(training_feature_columns) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0
    X_new = X_new[training_feature_columns].dropna()

    if X_new.empty:
        print(f"No data available for prediction for {stock_symbol}. Skipping.")
        return

    X_new_scaled = scaler.transform(X_new)
    predicted_return = model.predict(X_new_scaled)

    if predicted_return[-1] == 1:
        place_swing_trade(stock_symbol, qty)
    else:
        print(f"Skipping swing trade for {stock_symbol} due to predicted negative return.")

# Place a swing trade using Alpaca
def place_swing_trade(stock_symbol, qty):
    try:
        order_data = MarketOrderRequest(
            symbol=stock_symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order_data)
        print(f"Placed swing trade for {stock_symbol}")
        open_positions[stock_symbol] = qty
    except Exception as e:
        print(f"Error placing swing trade for {stock_symbol}: {e}")

# Manage swing trades for multiple stocks
def manage_swing_trades(stock_list, qty, model, scaler):
    for stock_symbol in stock_list:
        stock_data = get_current_stock_data(stock_symbol)
        if stock_data is None:
            print(f"Skipping {stock_symbol} due to data retrieval error")
            continue

        features = pd.DataFrame([{
            'price_at_trade': stock_data['current_price'],
            'volatility': get_stock_volatility(stock_symbol),
            'volume': stock_data['current_volume'],
            'avg_volume': stock_data['avg_volume']
        }])

        features = features[['price_at_trade', 'volatility', 'volume', 'avg_volume']]
        scaled_features = scaler.transform(features)
        predicted_outcome = model.predict(scaled_features)[0]

        if predicted_outcome == 1:
            swing_trade_stock(stock_symbol, qty)
        else:
            print(f"Skipping swing trade for {stock_symbol} due to predicted negative return.")

# Check exit conditions for open trades
def check_exit_conditions(stock_symbol):
    try:
        if stock_symbol not in open_positions:
            return

        stock_info = get_stock_info(stock_symbol)
        current_price = stock_info['price']
        purchase_price = open_positions[stock_symbol]
        if current_price >= purchase_price * 1.1:
            sell_stock(stock_symbol)
    except Exception as e:
        print(f"Error checking exit conditions for {stock_symbol}: {e}")

# Sell stock when exit conditions are met
def sell_stock(stock_symbol):
    try:
        qty = open_positions[stock_symbol]
        order_data = MarketOrderRequest(
            symbol=stock_symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order_data)
        print(f"Sold {qty} shares of {stock_symbol}")
        del open_positions[stock_symbol]
    except Exception as e:
        print(f"Error selling {stock_symbol}: {e}")