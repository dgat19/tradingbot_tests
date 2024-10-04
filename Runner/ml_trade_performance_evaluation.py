import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os

# Load and preprocess trade data
def load_trade_data():
    # Assuming this is your trade data structure, extend it with volume and avg_volume
    data = {
        'stock_symbol': ['AAPL', 'MSFT', 'NVDA', 'INTC', 'AVGO', 'LUNR', 'ASTS'],
        'price_at_trade': [150.0, 290.0, 560.0, 48.0, 620.0, 10.0, 5.0],
        'option_type': ['call', 'put', 'call', 'put', 'call', 'call', 'put'],
        'volatility': [0.35, 0.40, 0.25, 0.30, 0.50, 0.20, 0.60],
        'volume': [5000000, 10000000, 7500000, 2000000, 3000000, 4000000, 2500000],  # Example volume data
        'avg_volume': [4500000, 9500000, 7000000, 1800000, 2800000, 3800000, 2300000],  # Example average volume data
        'volume_signal': [1, 1, 0, 0, 1, 1, 0],  # Signal for high/low volume
        'market_sentiment': [1, -1, 1, -1, 1, 1, -1],  # Sentiment as binary for profit or loss
        'profit_loss': [1, 0, 1, 0, 1, 1, 0],  # 1 for profit, 0 for loss
        'profit_percentage': [12, -5, 15, -3, 20, 11, -7],  # Actual profit percentage
    }
    return pd.DataFrame(data)

# Preprocessing: One-hot encoding, scaling, and splitting data
def preprocess_data(trade_data):
    # Prepare feature columns and target
    X = trade_data[['price_at_trade', 'volatility', 'volume', 'avg_volume', 'volume_signal', 'market_sentiment']]  # Ensure these are the correct features
    y = trade_data['profit_loss']

    # Scale the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler  # Return the feature matrix X, scaled X, target y, and the scaler


# Train a model using GradientBoostingClassifier and return the trained model
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)
    
    return model, scaler


# Function to train the model or load an existing one if available
def train_or_load_model(X, y):
    model_file = 'trade_model.pkl'
    
    # Check if the model file exists
    if os.path.exists(model_file):
        print("Loading saved model...")
        model = load(model_file)  # Load the model from file
    else:
        print("Training new model...")
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train the model
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        dump(model, model_file)
        print("Model trained and saved.")
    
    return model

# Train ensemble models for more robust predictions
def train_ensemble_model(X_train, y_train):
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    ensemble_model = VotingClassifier(estimators=[('rf', model1), ('gb', model2)], voting='soft')
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

# Evaluate a new trade using the trained model
def evaluate_trade(model, price_at_trade, volatility, volume_signal, market_sentiment, scaler):
    trade_features = pd.DataFrame([{
        'price_at_trade': price_at_trade,
        'volatility': volatility,
        'volume_signal': volume_signal,
        'market_sentiment': market_sentiment
    }])
    
    # Scale only the numerical features
    trade_features[['price_at_trade', 'volatility', 'volume_signal', 'market_sentiment']] = scaler.transform(
        trade_features[['price_at_trade', 'volatility', 'volume_signal', 'market_sentiment']]
    )
    prediction = model.predict(trade_features)
    return prediction[0]  # 1 for profit, 0 for loss

# Adjust strategy based on model's prediction
def adjust_strategy_based_on_model(model, stock_symbol, price_at_trade, volatility, volume_signal, market_sentiment, scaler):
    predicted_outcome = evaluate_trade(model, price_at_trade, volatility, volume_signal, market_sentiment, scaler)
    
    if predicted_outcome == 0:  # Model predicts loss
        print(f"Adjusting strategy for {stock_symbol} to avoid losses...")
        # Logic to adjust strategy
        print(f"Consider adjusting entry point, or wait for better sentiment/volatility for {stock_symbol}")

# Main execution
if __name__ == "__main__":
    trade_data = load_trade_data()  # Load trade data
    X, X_scaled, y, scaler = preprocess_data(trade_data)  # Preprocess data

    # Load or train model and adjust strategy for a specific stock
    model = train_or_load_model(X, y)
    adjust_strategy_based_on_model(model, "AAPL", price_at_trade=155.0, volatility=0.38, volume_signal=1, market_sentiment=1, scaler=scaler)
