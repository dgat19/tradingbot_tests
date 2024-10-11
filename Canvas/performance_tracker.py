import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import alpaca_trade_api as tradeapi

class PerformanceTracker:
    def __init__(self):
        self.trade_history = []
        self.model = None
        self.scaler = StandardScaler()
        self.api = tradeapi.REST(
            os.getenv('PKV1PSBFZJSVP0SVHZ7U'),
            os.getenv('vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il'),
            'https://paper-api.alpaca.markets'
        )
        self.load_model()
        # Optimized trading parameters
        self.trailing_stop_loss = 0.05  # Adjusted for tighter risk management
        self.take_profit_threshold = 0.15  # Adjusted to secure gains earlier

    def evaluate_trades(self, trades):
        for trade in trades:
            # Track trades and evaluate outcomes based on optimized conditions
            if self.should_exit_trade(trade):
                self.exit_trade(trade)

            # Save trade history
            self.trade_history.append(trade)

    def should_exit_trade(self, trade):
        # Optimized exit strategy
        entry_price = trade['entry_price']
        current_price = self.get_current_price(trade['symbol'])

        if current_price is None:
            return False

        # Calculate percentage change
        price_change = (current_price - entry_price) / entry_price

        # Check if conditions for exit are met
        if price_change >= self.take_profit_threshold:
            return True  # Take profit
        elif price_change <= -self.trailing_stop_loss:
            return True  # Stop loss

        return False

    def exit_trade(self, trade):
        # Implement exit logic
        try:
            self.api.close_position(trade['symbol'])
            print(f"Exiting trade for {trade['symbol']} at current price.")
        except Exception as e:
            print(f"Error exiting trade for {trade['symbol']}: {e}")

    def get_current_price(self, symbol):
        # Fetch current price using Alpaca API
        try:
            barset = self.api.get_latest_bar(symbol)
            current_price = barset.c
            return current_price
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None

    def learn_from_past_trades(self):
        if len(self.trade_history) < 10:
            return

        # Prepare data for machine learning
        data = pd.DataFrame(self.trade_history)
        features = data[['entry_price', 'volatility', 'volume', 'news_sentiment', 'delta', 'theta', 'gamma']]
        labels = data['return']

        # Include additional features
        features['risk_reward_ratio'] = data['return'] / self.trailing_stop_loss

        # Scale the features
        features_scaled = self.scaler.fit_transform(features)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

        # Hyperparameter optimization using GridSearchCV for XGBoost
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        grid_search_xgb = GridSearchCV(estimator=xg_reg, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
        grid_search_xgb.fit(X_train, y_train)
        self.model = grid_search_xgb.best_estimator_

        # Evaluate model accuracy
        xgb_predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, xgb_predictions)
        mae = mean_absolute_error(y_test, xgb_predictions)
        r2 = r2_score(y_test, xgb_predictions)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

        print(f"XGBoost Regressor - MSE: {mse}, MAE: {mae}, R2 Score: {r2}, Adjusted R2 Score: {adj_r2}")

        # Save the trained model
        joblib.dump(self.model, 'trade_model.pkl')
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
        trade_data['risk_reward_ratio'] = trade['return'] / self.trailing_stop_loss
        trade_data_scaled = self.scaler.transform(trade_data)

        # Predict return
        predicted_return = self.model.predict(trade_data_scaled)[0]
        return predicted_return
