import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from joblib import dump, load

# Placeholder for trade data, could be expanded with real-time collected data
def load_trade_data():
    data = {
        'stock_symbol': ['AAPL', 'MSFT', 'NVDA', 'INTC', 'AVGO', 'LUNR', 'ASTS'],
        'price_at_trade': [150.0, 290.0, 560.0, 48.0, 620.0, 10.0, 5.0],
        'option_type': ['call', 'put', 'call', 'put', 'call', 'call', 'put'],
        'volatility': [0.35, 0.40, 0.25, 0.30, 0.50, 0.20, 0.60],
        'volume_signal': [1, 1, 0, 0, 1, 1, 1],  # 1 for high volume, 0 for normal
        'market_sentiment': [1, -1, 1, -1, 1, 1, -1],  # 1 for positive, -1 for negative
        'profit_loss': [1, 0, 1, 0, 1, 1, 0],  # 1 for profit, 0 for loss
        'profit_percentage': [12, -5, 15, -3, 20, 11, -7],  # Actual profit percentage
    }
    return pd.DataFrame(data)

def preprocess_data(trade_data):
    # Preprocessing steps:
    # 1. One-hot encode the 'option_type'
    trade_data = pd.get_dummies(trade_data, columns=['option_type'], drop_first=True)
    
    # 2. Scale numerical features
    scaler = StandardScaler()
    trade_data[['price_at_trade', 'volatility']] = scaler.fit_transform(trade_data[['price_at_trade', 'volatility']])
    
    return trade_data

def train_model(trade_data):
    X = trade_data[['price_at_trade', 'volatility', 'volume_signal', 'market_sentiment']]
    y = trade_data['profit_loss']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model

def load_model():
    # Load the previously trained model from file
    return joblib.load("trade_model.pkl")

def train_or_load_model(trade_data):
    try:
        model = load('trade_model.pkl')  # Load the model if it exists
    except FileNotFoundError:
        model = train_model(trade_data)  # Train the model if no saved model is found
        dump(model, 'trade_model.pkl')   # Save the model for future use
    return model

def train_ensemble_model(X_train, y_train):
    model1 = RandomForestClassifier(n_estimators=100)
    model2 = GradientBoostingClassifier(n_estimators=100)
    ensemble_model = VotingClassifier(estimators=[('rf', model1), ('gb', model2)], voting='soft')
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def evaluate_trade(model, stock_symbol, price_at_trade, volatility, volume_signal, market_sentiment):
    trade_features = pd.DataFrame([{
        'price_at_trade': price_at_trade,
        'volatility': volatility,
        'volume_signal': volume_signal,
        'market_sentiment': market_sentiment
    }])
    
    prediction = model.predict(trade_features)
    return prediction[0]  # 1 for profit, 0 for loss

def adjust_strategy_based_on_model(model, stock_symbol, price_at_trade, volatility, volume_signal, market_sentiment):
    # Evaluate the trade using the machine learning model
    predicted_outcome = evaluate_trade(model, stock_symbol, price_at_trade, volatility, volume_signal, market_sentiment)
    
    if predicted_outcome == 0:  # If the model predicts a loss
        print(f"Adjusting strategy for {stock_symbol} to avoid losses...")
        # Logic to adjust strategy: e.g., avoid similar trades, change entry/exit criteria
        # Could change strike price, volume thresholds, or market sentiment rules
        # For now, we print the adjustment:
        print(f"Consider adjusting entry point, price levels, or wait for better sentiment/volatility for {stock_symbol}")

# Main function to load data, train model, and evaluate future trades
if __name__ == "__main__":
    # Load trade data
    trade_data = load_trade_data()
    
    # Train the model based on past trades
    model = train_model(trade_data)
    
    # Evaluate a new trade (example: AAPL with specific characteristics)
    adjust_strategy_based_on_model(model, "AAPL", price_at_trade=155.0, volatility=0.38, volume_signal=1, market_sentiment=1)