# model_trainer.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.model = None
        self.initial_cash = 100000

    def preprocess_data(self):
        # Load data
        data = pd.read_csv(self.data_file)

        # Ensure data is sorted by date
        data = data.sort_values('date')
        data.reset_index(drop=True, inplace=True)

        # Create target variable: 1 if option price increases next period, else 0
        data['option_price_next'] = data['option_price'].shift(-1)
        data['target'] = (data['option_price_next'] > data['option_price']).astype(int)
        data.dropna(inplace=True)

        # Define features (exclude 'date' and 'option_price_next')
        features = ['option_price', 'underlying_price', 'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility']
        target = 'target'

        return data, features, target

    def train_model(self, X_train, y_train):
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

        # Fit model
        self.model.fit(X_train, y_train)

    def simulate_trading(self, data_test, y_pred):
        # Create a DataFrame for analysis
        trading_df = data_test.copy()
        trading_df['y_pred'] = y_pred

        # Initialize variables
        cash = self.initial_cash
        trade_log = []

        for index, row in trading_df.iterrows():
            if row['y_pred'] == 1:
                # Buy at current price
                buy_price = row['option_price']
                cash -= buy_price
                # Sell at next period's price
                sell_price = row['option_price_next']
                cash += sell_price
                profit = sell_price - buy_price
                trade_log.append({
                    'date': row['date'],
                    'action': 'buy and sell',
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'profit': profit,
                    'cash': cash
                })
            else:
                # Do nothing
                pass

        total_profit = cash - self.initial_cash

        return {'total_profit': total_profit, 'trade_log': trade_log}

    def backtest(self, X_test, y_test, data_test):
        # Predict on test data
        y_pred = self.model.predict(X_test)

        # Compute performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Simulate trading
        results = self.simulate_trading(data_test, y_pred)

        return {'accuracy': accuracy, 'report': report, 'trading_results': results}

    def record_performance_metrics(self, backtest_results):
        accuracy = backtest_results['accuracy']
        report = backtest_results['report']
        total_profit = backtest_results['trading_results']['total_profit']
        trade_log = backtest_results['trading_results']['trade_log']

        print("Model Accuracy:", accuracy)
        print("Classification Report:")
        print(report)
        print("Total Profit from Backtesting:", total_profit)

        # Save the trade log to a CSV file
        trade_log_df = pd.DataFrame(trade_log)
        trade_log_df.to_csv('trade_log.csv', index=False)

    def run(self):
        # Preprocess data
        data, features, target = self.preprocess_data()

        # Prepare features and target
        X = data[features]
        y = data[target]

        # Split data (ensure no shuffling to maintain time series order)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        data_train, data_test = train_test_split(data, test_size=0.2, shuffle=False)

        # Train model
        self.train_model(X_train, y_train)

        # Save the trained model
        dump(self.model, 'model.joblib')

        # Backtest
        backtest_results = self.backtest(X_test, y_test, data_test)

        # Record performance metrics
        self.record_performance_metrics(backtest_results)


if __name__ == '__main__':
    trainer = ModelTrainer('options_data.csv')
    trainer.run()