import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os

# Load and preprocess trade data
def load_trade_data():
    # Your trade data structure with extended features
    data = {
        'stock_symbol': ['AAPL', 'MSFT', 'NVDA', 'INTC', 'AVGO', 'LUNR', 'ASTS'],
        'price_at_trade': [150.0, 290.0, 560.0, 48.0, 620.0, 10.0, 5.0],
        'option_type': ['call', 'put', 'call', 'put', 'call', 'call', 'put'],
        'volatility': [0.35, 0.40, 0.25, 0.30, 0.50, 0.20, 0.60],
        'volume': [5000000, 10000000, 7500000, 2000000, 3000000, 4000000, 2500000],
        'avg_volume': [4500000, 9500000, 7000000, 1800000, 2800000, 3800000, 2300000],
        'volume_signal': [1, 1, 0, 0, 1, 1, 0],
        'market_sentiment': [1, -1, 1, -1, 1, 1, -1],
        'profit_loss': [1, 0, 1, 0, 1, 1, 0],
        'profit_percentage': [12, -5, 15, -3, 20, 11, -7],
    }
    return pd.DataFrame(data)

# Preprocess data (ensure consistent features between training and prediction)
def preprocess_data(trade_data):
    # Consistent feature columns
    feature_columns = ['price_at_trade', 'volatility', 'volume_signal', 'market_sentiment']
    X = trade_data[feature_columns]  # Features
    y = trade_data['profit_loss']    # Target variable

    # Return X and y; scaling will be done during model training
    return X, y, feature_columns

# Function to train the model or load an existing one if available
def train_or_load_model(X, y, feature_columns):
    model_file = 'trade_model.pkl'
    scaler_file = 'scaler.pkl'
    feature_columns_file = 'training_feature_columns.pkl'

    # Check if the model and scaler files exist
    if os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(feature_columns_file):
        print("Loading saved model, scaler, and feature columns...")
        model = load(model_file)
        scaler = load(scaler_file)
        training_feature_columns = load(feature_columns_file)
    else:
        print("Training new model...")
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and fit the scaler on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train the model
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save the trained model, scaler, and feature columns
        dump(model, model_file)
        dump(scaler, scaler_file)
        dump(feature_columns, feature_columns_file)
        print("Model, scaler, and feature columns saved.")
        training_feature_columns = feature_columns

    return model, scaler, training_feature_columns

# Evaluate a new trade using the trained model
def evaluate_trade(model, price_at_trade, volatility, volume_signal, market_sentiment, scaler, feature_columns):
    trade_features = pd.DataFrame([{
        'price_at_trade': price_at_trade,
        'volatility': volatility,
        'volume_signal': volume_signal,
        'market_sentiment': market_sentiment
    }])

    # Ensure columns are in the same order
    trade_features = trade_features[feature_columns]

    # Scale the features
    trade_features_scaled_array = scaler.transform(trade_features)
    # Convert back to DataFrame with feature names
    trade_features_scaled = pd.DataFrame(trade_features_scaled_array, columns=feature_columns)

    # Predict using the model
    prediction = model.predict(trade_features_scaled)
    return prediction[0]  # 1 for profit, 0 for loss

# Adjust strategy based on model's prediction
def adjust_strategy_based_on_model(model, stock_symbol, price_at_trade, volatility, volume_signal, market_sentiment, scaler, feature_columns):
    predicted_outcome = evaluate_trade(
        model,
        price_at_trade,
        volatility,
        volume_signal,
        market_sentiment,
        scaler,
        feature_columns
    )

    if predicted_outcome == 1:  # Model predicts profit
        print(f"Proceeding with the trade for {stock_symbol}. Expected to be profitable.")
    else:
        print(f"Adjusting strategy for {stock_symbol} to avoid losses...")
        # Logic to adjust strategy
        print(f"Consider adjusting entry point or waiting for better conditions for {stock_symbol}")

# Main execution
if __name__ == "__main__":
    trade_data = load_trade_data()  # Load trade data
    X, y, feature_columns = preprocess_data(trade_data)  # Preprocess data

    # Load or train model and adjust strategy for a specific stock
    model, scaler, training_feature_columns = train_or_load_model(X, y, feature_columns)
    adjust_strategy_based_on_model(
        model,
        "AAPL",
        price_at_trade=155.0,
        volatility=0.38,
        volume_signal=1,
        market_sentiment=1,
        scaler=scaler,
        feature_columns=training_feature_columns
    )
