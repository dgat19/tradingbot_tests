import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import os
from datetime import datetime
from indicators.tech_indicators import (
    calculate_rsi,
    calculate_macd,
    get_bollinger_bands,
    get_trend_indicator,
    get_volume_indicator,
    get_stock_volatility,
    get_historical_monthly_performance
)

# =============================================================================
# Model Manager
# =============================================================================
class ModelManager:
    """Handles saving and loading RL models"""

    def __init__(self, base_path='data/models/'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def get_model_path(self, symbol):
        return os.path.join(self.base_path, f'{symbol}_rl_model.h5')

    def save_model(self, model, symbol):
        model.save(self.get_model_path(symbol))

    def load_model(self, symbol):
        model_path = self.get_model_path(symbol)
        return load_model(model_path) if os.path.exists(model_path) else None

# =============================================================================
# Market Simulator
# =============================================================================
class MarketSimulator:
    """Simulates market conditions for backtesting"""

    def __init__(self, symbol, start_date, end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self.data = self._load_historical_data()

    def _load_historical_data(self):
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        return self._add_technical_indicators(data)

    def _add_technical_indicators(self, data):
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'] = calculate_macd(data)
        bb_upper, bb_lower = get_bollinger_bands(self.symbol)
        data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        return data
    
    @staticmethod
    def run_scenarios(symbols, date_ranges):
        """
        Demonstrate multi-scenario backtesting. 
        symbols: list of symbols to test (e.g. ['AAPL','TSLA'])
        date_ranges: list of (start_date, end_date) tuples
        Returns a dict of {symbol -> {scenario_idx -> DataFrame}}
        """
        all_data = {}
        for sym in symbols:
            all_data[sym] = {}
            for idx, (start, end) in enumerate(date_ranges):
                sim = MarketSimulator(sym, start, end)
                all_data[sym][idx] = sim.data
        return all_data


# =============================================================================
# Enhanced RL Environment
# =============================================================================
class EnhancedStockTradingEnv:
    """
    RL environment for stock trading with an expanded action space:
        0: Hold
        1: Enter (or add to) Call
        2: Enter (or add to) Put
        3: Close position
    """

    def __init__(self, symbol, data, news_analyzer, 
                 initial_balance=10000, mode='train',
                 max_position_size=1):
        self.symbol = symbol
        self.data = data
        self.news_analyzer = news_analyzer
        self.initial_balance = initial_balance
        self.mode = mode
        self.scaler = StandardScaler()
        self.max_position_size = max_position_size  # e.g. 1 means 1 "unit" max
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        # position > 0 means Call side exposure, < 0 means Put side exposure
        self.position = 0
        self.current_step = 0
        self.done = False
        self.positions_history = []
        return self._get_observation()

    def _get_observation(self):
        if self.current_step >= len(self.data):
            return None

        current_data = self.data.iloc[self.current_step]
        # We'll scale the numeric features but keep position & balance as is
        # or you can scale them separately if desired
        obs = np.array([
            current_data['Close'], 
            current_data['Volume'], 
            current_data['RSI'],
            current_data['MACD'], 
            current_data['MACD_Signal'], 
            current_data['BB_Position'],
            get_trend_indicator(self.symbol),
            1 if get_volume_indicator(self.symbol) else 0,
            get_stock_volatility(self.symbol),
            get_historical_monthly_performance(self.symbol),
            self._get_sentiment_score(),
            self.balance,
            self.position
        ], dtype=float)

        # Scale only the first 11 features in training mode
        if self.mode == 'train':
            scaled_part = self.scaler.fit_transform(obs[:11].reshape(1, -1))[0]
            return np.concatenate([scaled_part, obs[11:]])  # append unscaled
        else:
            return obs

    def _get_sentiment_score(self):
        """
        Cache the sentiment in backtest mode to keep it consistent per step.
        """
        if hasattr(self, '_cached_sentiment') and self.mode == 'backtest':
            return self._cached_sentiment
        sentiment = self.news_analyzer.analyze_sentiment(
            self.news_analyzer.fetch_stock_news(self.symbol)
        )
        score = sentiment['sentiment_score'] if sentiment else 0
        if self.mode == 'backtest':
            self._cached_sentiment = score
        return score

    def step(self, action):
        """
        Action Space:
        0 = hold
        1 = enter/add call
        2 = enter/add put
        3 = close position
        """
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            # We are at the end, finalize reward
            self.done = True
            return self._get_observation(), self._calculate_final_reward(), True

        reward = self._calculate_step_reward(action)
        self.positions_history.append({
            'step': self.current_step,
            'action': action,
            'position': self.position,
            'reward': reward,
            'balance': self.balance
        })
        return self._get_observation(), reward, self.done

    def _calculate_step_reward(self, action):
        """
        Enhanced reward logic:
            - Encourage correct directional moves
            - Penalize frequent position flipping
            - Include time decay & transaction costs
            - Optionally risk-adjust (e.g. reduce reward for large positions)
        """
        current_price = self.data.iloc[self.current_step]['Close']
        next_price   = self.data.iloc[self.current_step + 1]['Close']
        delta_price  = next_price - current_price

        # Base reward from price move
        reward = 0

        # Evaluate action
        if action == 0:
            # Hold
            if self.position > 0:
                # Gains if price goes up
                reward += max(0, delta_price) / current_price
            elif self.position < 0:
                # Gains if price goes down
                reward += max(0, -delta_price) / current_price

        elif action == 1:
            # Enter or add to a Call
            if self.position < 0:
                # Switch from Put to Call quickly = penalty
                reward -= 0.01
                self.position = 0
            if abs(self.position) < self.max_position_size:
                self.position += 1
            # Gains if price goes up
            reward += max(0, delta_price) / current_price
            # Transaction cost
            reward -= 0.0005 

        elif action == 2:
            # Enter or add to a Put
            if self.position > 0:
                # Switch from Call to Put quickly = penalty
                reward -= 0.01
                self.position = 0
            if abs(self.position) < self.max_position_size:
                self.position -= 1
            # Gains if price goes down
            reward += max(0, -delta_price) / current_price
            # Transaction cost
            reward -= 0.0005

        elif action == 3:
            # Close position
            if self.position != 0:
                # Potential final "unrealized" gain/loss from the move
                if self.position > 0:
                    reward += max(0, delta_price) / current_price
                else:
                    reward += max(0, -delta_price) / current_price
                self.position = 0
            # Transaction cost
            reward -= 0.0005

        # Time decay penalty if holding a position overnight
        if self.position != 0:
            reward -= 0.001

        return reward

    def _calculate_final_reward(self):
        """Example final reward: difference from initial balance (if you track PnL).
           Here, we just do a final calculation. You can keep track of real PnL. 
        """
        return (self.balance - self.initial_balance) / self.initial_balance


# =============================================================================
# Enhanced DQN Agent
# =============================================================================
class EnhancedDQNAgent:
    """Deep Q-Learning agent with experience replay and target network"""

    def __init__(self, state_size, action_size, model_manager=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.model_manager = model_manager
        self.update_target_network()

    def _build_model(self):
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')  # e.g. 4 if we have [Hold, Call, Put, Close]
        ])
        model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        # Epsilon-greedy policy
        if not evaluate and random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # returns 0..3
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        # Predict Q-values for the next states (using the target model)
        target_qs = self.target_model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.amax(target_qs, axis=1) * (1 - dones)

        # Current Q-values
        current_qs = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            current_qs[i][action] = targets[i]

        self.model.fit(states, current_qs, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, symbol):
        if self.model_manager:
            self.model_manager.save_model(self.model, symbol)

    def load(self, symbol):
        if self.model_manager:
            loaded_model = self.model_manager.load_model(symbol)
            if loaded_model:
                self.model = loaded_model
                self.update_target_network()
                return True
        return False
