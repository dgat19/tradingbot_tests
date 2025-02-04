"""Trains or retrains regression models using historical data from data/processed.

After training, saves the model to data/models/ (e.g., model.joblib).

Could also evaluate model performance (MSE, MAE, R^2) and log results in data/logs/.

Possibly run periodically (via scripts/train_model.sh) to keep the model up-to-date."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import data.news_scraper
from indicators.tech_indicators import (
    calculate_macd,
    calculate_rsi,
    get_bollinger_bands,
    analyze_indicators,
    get_stock_volatility
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class TradingEnvironment(gym.Env):
    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker.upper()
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period="6mo")
            if self.data.empty:
                raise ValueError(f"No data found for {self.ticker}")
        except Exception as e:
            print(f"Error initializing data for {self.ticker}: {e}")
            self.data = pd.DataFrame()

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward(action)
        return self._get_observation(), reward, done, False, {}

    def _calculate_reward(self, action):
        if self.current_step + 1 >= len(self.data):
            return 0
        next_return = (
            self.data['Close'].iloc[self.current_step + 1] / self.data['Close'].iloc[self.current_step] - 1
        )
        return next_return if action == 0 else -next_return if action == 1 else 0

    def _get_observation(self):
        if self.data.empty or self.current_step >= len(self.data):
            return np.zeros(12, dtype=np.float32)

        current_data = self.data.iloc[self.current_step]
        return np.array([
            current_data['Close'],
            current_data['Volume'],
            calculate_rsi(self.data.iloc[:self.current_step + 1]).iloc[-1],
            *calculate_macd(self.data.iloc[:self.current_step + 1]),
            *get_bollinger_bands(self.ticker),
            self.current_step / len(self.data),
        ], dtype=np.float32)


class MLTradeEvaluator:
    def __init__(self):
        self.news_analyzer = data.news_scraper.StockDiscoveryAndAnalysis()
        self.scaler = StandardScaler()
        self.rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
        self.gb_regressor = GradientBoostingRegressor(n_estimators=200, random_state=42)
        self.rl_models = {}

    def _get_stock_features(self, ticker):
        try:
            indicator = analyze_indicators(ticker)
            stock_data = yf.Ticker(ticker).history(period="6mo")

            if stock_data.empty:
                return None

            upper_band, lower_band = get_bollinger_bands(ticker)
            features = {
                'volatility': get_stock_volatility(ticker),
                'monthly_performance': indicator['monthly_performance'],
                'rsi': indicator['rsi'],
                'macd_signal': indicator['positive_trend'],
                'volume_trend': indicator['volume_trend'],
                'trend': indicator['trend'],
                'bb_position': (stock_data['Close'].iloc[-1] - lower_band) / (upper_band - lower_band),
            }
            return features
        except Exception as e:
            print(f"Error getting features for {ticker}: {e}")
            return None

    def _prepare_training_data(self):
        data = []
        tickers = self.news_analyzer.get_yahoo_trending().union(self.news_analyzer.get_swaggy_sentiment())

        for ticker in tickers:
            features = self._get_stock_features(ticker)
            if not features:
                continue

            articles = self.news_analyzer.fetch_stock_news(ticker)
            sentiment = self.news_analyzer.analyze_sentiment(articles)
            features.update({
                'sentiment_score': sentiment.get('sentiment_score', 0),
                'article_count': sentiment.get('article_count', 0),
            })

            future_data = yf.Ticker(ticker).history(period="5d")
            if not future_data.empty:
                features['future_return'] = future_data['Close'].pct_change().mean()
                data.append(features)

        return pd.DataFrame(data)

    def train_models(self):
        data = self._prepare_training_data()
        if data.empty:
            print("No data available for training.")
            return

        X = data.drop('future_return', axis=1)
        y = data['future_return']
        X_scaled = self.scaler.fit_transform(X)

        self.rf_regressor.fit(X_scaled, y)
        self.gb_regressor.fit(X_scaled, y)

        for ticker in X.index:
            env = DummyVecEnv([lambda: TradingEnvironment(ticker)])
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=10000)
            self.rl_models[ticker] = model

    def evaluate_trade(self, ticker):
        features = self._get_stock_features(ticker)
        if not features:
            return None

        sentiment = self.news_analyzer.analyze_sentiment(self.news_analyzer.fetch_stock_news(ticker))
        features.update({
            'sentiment_score': sentiment.get('sentiment_score', 0),
            'article_count': sentiment.get('article_count', 0),
        })

        features_scaled = self.scaler.transform(pd.DataFrame([features]))
        rf_pred = self.rf_regressor.predict(features_scaled)[0]
        gb_pred = self.gb_regressor.predict(features_scaled)[0]
        ensemble_pred = (rf_pred + gb_pred) / 2

        rl_action = self.rl_models.get(ticker, None)
        if rl_action:
            obs, _ = TradingEnvironment(ticker).reset()
            action, _ = rl_action.predict(obs)
        else:
            action = 1  # Default to Hold

        return {
            'ticker': ticker,
            'predicted_return': ensemble_pred,
            'rl_action': ['BUY', 'SELL', 'HOLD'][action],
            'confidence': abs(ensemble_pred),
        }


def main():
    evaluator = MLTradeEvaluator()
    evaluator.train_models()

    for ticker in evaluator.news_analyzer.get_yahoo_trending():
        evaluation = evaluator.evaluate_trade(ticker)
        if evaluation and evaluation['confidence'] > 0.6:
            print(f"{ticker} - Predicted Return: {evaluation['predicted_return']:.2%}")
