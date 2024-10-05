import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import alpaca_trade_api as tradeapi

class PerformanceTracker:
    def __init__(self):
        self.trade_history = []
        self.model = None
        self.scaler = StandardScaler()
        self.api = tradeapi.REST('PKV1PSBFZJSVP0SVHZ7U', 'vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il', 'https://paper-api.alpaca.markets')  # Initialize Alpaca API
        self.load_model()
        # Optimized trading parameters
        self.trailing_stop_loss = 0.10  # Reduced trailing stop loss to lock in profits sooner
        self.take_profit_threshold = 0.25  # Reduced take profit threshold to secure gains earlier

    def evaluate_trades(self, trades):
        for trade in trades:
            # Track trades and evaluate outcomes based on optimized conditions
            if self.should_exit_trade(trade):
                self.exit_trade(trade)

            # Save trade history
            self.trade_history.append(trade)

    def should_exit_trade(self, trade):
        # Optimized exit strategy: Consider both unrealized gains and trailing stop loss
        current_gain = trade['unrealized_gain']
        entry_price = trade['entry_price']
        current_price = self.get_current_price(trade['symbol'])

        # Calculate percentage change
        price_change = (current_price - entry_price) / entry_price

        # Check if conditions for exit are met
        if price_change >= self.take_profit_threshold:
            return True  # Take profit
        elif price_change <= -self.trailing_stop_loss:
            return True  # Stop loss

        return False

    def exit_trade(self, trade):
        # Implement exit logic (e.g., using Alpaca API to close positions)
        try:
            self.api.close_position(trade['symbol'])
            print(f"Exiting trade for {trade['symbol']} at current price.")
        except Exception as e:
            print(f"Error exiting trade for {trade['symbol']}: {e}")

    def get_current_price(self, symbol):
        # Fetch current price using Alpaca API
        try:
            barset = self.api.get_barset(symbol, 'minute', limit=1)
            current_price = barset[symbol][0].c
            return current_price
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None

    def learn_from_past_trades(self):
        if len(self.trade_history) < 10:  # Ensure enough data for training
            return

        # Prepare data for machine learning
        data = pd.DataFrame(self.trade_history)
        features = data[['entry_price', 'volatility', 'volume', 'news_sentiment', 'delta', 'theta', 'gamma']]
        labels = data['return']

        # Scale the features
        features_scaled = self.scaler.fit_transform(features)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

        # Hyperparameter optimization using GridSearchCV for RandomForestRegressor
        param_grid_rf = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
        grid_search_rf.fit(X_train, y_train)
        self.model = grid_search_rf.best_estimator_

        # Hyperparameter optimization using GridSearchCV for MLPRegressor
        param_grid_mlp = {
            'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
            'max_iter': [1000, 2000],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        mlp = MLPRegressor(random_state=42)
        grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=3, n_jobs=-1, verbose=2)
        grid_search_mlp.fit(X_train, y_train)
        best_mlp_model = grid_search_mlp.best_estimator_

        # Evaluate RandomForest model accuracy
        rf_predictions = self.model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_r2 = r2_score(y_test, rf_predictions)
        print(f"RandomForestRegressor - MSE: {rf_mse}, R2 Score: {rf_r2}")

        # Evaluate MLP model accuracy
        mlp_predictions = best_mlp_model.predict(X_test)
        mlp_mse = mean_squared_error(y_test, mlp_predictions)
        mlp_r2 = r2_score(y_test, mlp_predictions)
        print(f"MLPRegressor - MSE: {mlp_mse}, R2 Score: {mlp_r2}")

        # Save the trained models
        joblib.dump(self.model, 'trade_model.pkl')
        joblib.dump(best_mlp_model, 'mlp_trade_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')

    def load_model(self):
        try:
            self.model = joblib.load('trade_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            self.model = None

    def predict_trade_success(self, trade):
        if not self.model:
            return 0.0  # Default value if model is not trained yet

        # Prepare trade data for prediction
        trade_data = pd.DataFrame([trade])[['entry_price', 'volatility', 'volume', 'news_sentiment', 'delta', 'theta', 'gamma']]
        trade_data_scaled = self.scaler.transform(trade_data)

        # Predict return
        predicted_return = self.model.predict(trade_data_scaled)[0]
        return predicted_return