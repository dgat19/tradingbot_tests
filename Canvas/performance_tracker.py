import os
import pandas as pd
import joblib
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as tradeapi

class PerformanceTracker:
    def __init__(self):
        self.trade_history = []
        self.scaler = StandardScaler()
        self.api = tradeapi.REST(
            os.getenv('PKV1PSBFZJSVP0SVHZ7U'),
            os.getenv('vnTZhGmchG0xNOGXvJyQIFqSmfkPMYvBIcOcA5Il'),
            'https://paper-api.alpaca.markets'
        )
        self.load_model()

    def load_model(self):
        try:
            # Load PPO model and scaler
            self.model = PPO.load("ppo_trade_model")
            self.scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            self.model = None

    def train_ppo_model(self):
        if len(self.trade_history) < 10:
            print("Not enough trades to train the model.")
            return

        # Prepare data for PPO training
        data = pd.DataFrame(self.trade_history)
        features = data[['entry_price', 'volatility', 'volume', 'news_sentiment', 'delta', 'theta', 'gamma']]
        labels = data['return']

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Custom Environment for PPO
        env = DummyVecEnv([lambda: TradingEnv(features_scaled, labels)])

        # Train PPO model
        self.model = PPO('MlpPolicy', env, verbose=1)
        self.model.learn(total_timesteps=10000)

        # Save model and scaler
        self.model.save("ppo_trade_model")
        joblib.dump(self.scaler, 'scaler.pkl')

    def predict_trade_success(self, trade):
        if not self.model:
            return 0.0  # Default value if model is not trained yet

        # Prepare trade data for prediction
        trade_data = pd.DataFrame([trade])[['entry_price', 'volatility', 'volume', 'news_sentiment', 'delta', 'theta', 'gamma']]
        trade_data_scaled = self.scaler.transform(trade_data)

        # Simulate environment step and predict
        env = DummyVecEnv([lambda: TradingEnv(trade_data_scaled)])
        return self.model.predict(env)[0][0]


class TradingEnv(gym.Env):
    def __init__(self, features, labels=None):
        super(TradingEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.current_step = 0

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)  # Buy or Hold/Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.features[self.current_step]

    def step(self, action):
        reward = 0
        done = False

        # Calculate reward based on action
        if action == 1:  # Buy/Hold
            reward = self.labels[self.current_step] if self.labels is not None else 0

        self.current_step += 1
        if self.current_step >= len(self.features) - 1:
            done = True

        obs = self.features[self.current_step] if not done else self.features[-1]
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass
