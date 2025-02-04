import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from datetime import datetime

class DQN(nn.Module):
    """Deep Q-Network using PyTorch"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)
class TradingAction:
    """Represents a complete trading action with all parameters"""
    def __init__(self, action_type, position_size=0.0, stop_loss=0.0, take_profit=0.0):
        self.action_type = action_type  # 0: hold, 1: buy, 2: sell
        self.position_size = position_size  # Percentage of available capital
        self.stop_loss = stop_loss  # Percentage from entry
        self.take_profit = take_profit  # Percentage from entry

class EnhancedTradingState:
    """Enhanced state representation with market context"""
    def __init__(self, di_plus, di_minus, price, market_features=None, position=None):
        self.di_plus = di_plus
        self.di_minus = di_minus
        self.price = price
        self.market_features = market_features or {}
        self.position = position or {}
        
    def get_market_context(self):
        """Calculate market context scores"""
        trend_score = self.market_features.get('trend', 0) / 10  # Normalize to -1 to 1
        volatility_score = min(1.0, self.market_features.get('volatility', 0) / 20)
        momentum_score = self.market_features.get('momentum', 0) / 10
        return trend_score, volatility_score, momentum_score

    def to_array(self):
        """Convert state to array for neural network"""
        trend_score, vol_score, mom_score = self.get_market_context()
        
        return np.array([
            self.di_plus / 100,  # Normalize indicators
            self.di_minus / 100,
            trend_score,
            vol_score,
            mom_score,
            self.position.get('size', 0),  # Current position size
            self.position.get('pl_pct', 0) / 100,  # Current P/L
            self.position.get('holding_time', 0) / 480  # Normalize to trading day
        ])
class EnhancedTradingEnvironment:
    """Enhanced trading environment with direct control"""
    def __init__(self, initial_balance=100000.0):
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset the environment"""
        self.balance = self.initial_balance
        self.position = None
        self.trade_history = []
        self.market_features = {}
        self.current_state = None
        return self.current_state
        
    def calculate_market_features(self, df, lookback_window=20):
        """Calculate market features with momentum indicators"""
        if len(df) < lookback_window:
            return {}
            
        current_price = df['close'].iloc[-1]
        lookback_prices = df['close'].tail(lookback_window)
        typical_prices = (df['high'] + df['low'] + df['close']) / 3
        
        # Basic features
        features = {
            'trend': (current_price - lookback_prices.mean()) / lookback_prices.mean() * 100,
            'volatility': lookback_prices.pct_change().std() * 100,
            'momentum': (current_price / lookback_prices.iloc[0] - 1) * 100,
            'price_range': (df['high'].tail(lookback_window).max() - 
                          df['low'].tail(lookback_window).min()) / current_price * 100,
        }
        
        # Add RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Add volume analysis
        features['volume_trend'] = (df['volume'].tail(5).mean() / 
                                  df['volume'].tail(20).mean() - 1) * 100
        
        return features
        
    def update_state(self, df):
        """Update environment state with market data"""
        if len(df) < 2:
            return None
            
        current_price = df['close'].iloc[-1]
        self.market_features = self.calculate_market_features(df)
        
        position_info = {}
        if self.position is not None:
            holding_time = len(df) - self.position['entry_bar']
            pl_pct = ((current_price - self.position['entry_price']) / 
                     self.position['entry_price'] * 100)
            position_info = {
                'size': self.position['size'],
                'pl_pct': pl_pct,
                'holding_time': holding_time
            }
        
        self.current_state = EnhancedTradingState(
            di_plus=df['DI+'].iloc[-1],
            di_minus=df['DI-'].iloc[-1],
            price=current_price,
            market_features=self.market_features,
            position=position_info
        )

        return self.current_state
        
    def step(self, action: TradingAction, df):
        """Execute trading action and return reward"""
        if len(df) < 2:
            return 0, False
            
        current_price = df['close'].iloc[-1]
        reward = 0
        done = False
        
        # Process the action
        if action.action_type == 1:  # Buy
            if self.position is None:
                # Calculate position size with market-aware sizing
                base_size = self.balance * action.position_size
                vol_adjustment = 1.0 - (self.market_features.get('volatility', 0) / 100)
                trend_adjustment = 1.0 + (self.market_features.get('trend', 0) / 100)
                adjusted_size = base_size * vol_adjustment * trend_adjustment
                position_dollars = min(adjusted_size, self.balance * 0.9)  # Cap at 90% of balance
                size = position_dollars / current_price
                
                self.position = {
                    'entry_price': current_price,
                    'size': size,
                    'stop_loss': current_price * (1 - action.stop_loss/100),
                    'take_profit': current_price * (1 + action.take_profit/100),
                    'entry_bar': len(df) - 1,
                    'entry_features': self.market_features.copy()
                }
                reward = -0.0001  # Small cost for taking position
                
        elif action.action_type == 2:  # Sell
            if self.position is not None:
                # Calculate profit/loss
                pl_pct = ((current_price - self.position['entry_price']) / 
                         self.position['entry_price'] * 100)
                
                # Dynamic reward calculation based on market conditions
                reward = self._calculate_dynamic_reward(pl_pct)
                
                # Update balance and record trade
                self.balance += self.position['size'] * current_price
                self._record_trade(current_price, pl_pct, len(df))
                
                self.position = None
                done = True
                
        elif self.position is not None:
            # Dynamic stop-loss and take-profit management
            self._manage_position_limits(current_price, df)
            
            # Calculate holding reward/penalty
            reward = self._calculate_holding_reward(current_price, df)
            
        return reward, done
        
    def _calculate_dynamic_reward(self, pl_pct):
        """Calculate reward based on profit and market conditions"""
        if pl_pct > 0:
            reward = np.exp(pl_pct/10) - 1  # Exponential reward for profits
            
            # Bonus for selling in favorable conditions
            if self.market_features.get('rsi', 50) > 70:
                reward *= 1.2  # Overbought bonus
            if self.market_features.get('volume_trend', 0) > 20:
                reward *= 1.1  # High volume bonus
                
        else:
            reward = pl_pct  # Linear penalty for losses
            
            # Reduce penalty in poor market conditions
            if self.market_features.get('trend', 0) < -2:
                reward *= 0.8
            if self.market_features.get('volatility', 0) > 20:
                reward *= 0.9
                
        return reward
        
    def _manage_position_limits(self, current_price, df):
        """Dynamically manage position limits based on market conditions"""
        if self.position is None:
            return
            
        # Dynamic stop-loss adjustment
        base_stop = self.position['stop_loss']
        if current_price > self.position['entry_price']:
            # Trail stop loss if in profit
            new_stop = current_price * 0.98  # 2% trailing stop
            self.position['stop_loss'] = max(base_stop, new_stop)
            
        # Check market conditions for early exit
        rsi = self.market_features.get('rsi', 50)
        volume_trend = self.market_features.get('volume_trend', 0)
        volatility = self.market_features.get('volatility', 0)
        
        # Tighten stops in high-risk conditions
        if rsi < 30 or volatility > 30 or volume_trend < -20:
            self.position['stop_loss'] = max(base_stop, current_price * 0.99)
            
    def _calculate_holding_reward(self, current_price, df):
        """Calculate reward/penalty for holding position"""
        if self.position is None:
            return 0
            
        holding_time = len(df) - self.position['entry_bar']
        pl_pct = ((current_price - self.position['entry_price']) / 
                 self.position['entry_price'] * 100)
                 
        # Base holding reward/penalty
        reward = 0
        
        # Reward for holding winning positions with good momentum
        if pl_pct > 0 and self.market_features.get('momentum', 0) > 0:
            reward = 0.0001 * pl_pct
            
        # Penalty for holding too long
        if holding_time > 30:  # More than 30 bars
            reward -= 0.0001 * holding_time
            
        return reward
        
    def _record_trade(self, exit_price, pl_pct, current_bar):
        """Record trade details with enhanced analytics"""
        if self.position is None:
            return
            
        holding_time = current_bar - self.position['entry_bar']
        
        # Calculate market condition changes
        condition_changes = {
            key: self.market_features.get(key, 0) - 
                 self.position['entry_features'].get(key, 0)
            for key in self.market_features
        }
        
        self.trade_history.append({
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'pl_pct': pl_pct,
            'size': self.position['size'],
            'holding_time': holding_time,
            'entry_features': self.position['entry_features'],
            'exit_features': self.market_features.copy(),
            'condition_changes': condition_changes,
            'timestamp': datetime.now()
        })
        
    def get_position_info(self):
        """Get current position information"""
        if self.position is None:
            return None
            
        return {
            'entry_price': self.position['entry_price'],
            'size': self.position['size'],
            'stop_loss': self.position['stop_loss'],
            'take_profit': self.position['take_profit'],
            'holding_time': len(self.position.get('entry_features', {}))
        }
        
    def get_portfolio_status(self):
        """Get current portfolio status"""
        return {
            'balance': self.balance,
            'returns': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'position': self.get_position_info(),
            'market_features': self.market_features.copy()
        }

class EnhancedTradingAgent:
    """Enhanced RL agent using PyTorch"""
    def __init__(self, learning_rate=0.001, gamma=0.95, epsilon=1.0):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Action space parameters
        self.position_sizes = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.stop_losses = [0.5, 1.0, 1.5, 2.0]
        self.take_profits = [1.0, 1.5, 2.0, 3.0]
        
        # Calculate total number of actions
        self.n_actions = len(self.position_sizes) * len(self.stop_losses) * len(self.take_profits) * 3
        
        # Initialize neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(8, 64, self.n_actions).to(self.device)
        self.target_model = DQN(8, 64, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return self._get_random_action()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.to_array()).to(self.device)
            q_values = self.model(state_tensor)
            action_idx = q_values.argmax().item()
            return self._decode_action(action_idx)
        
    def _get_random_action(self):
        """Generate random action for exploration"""
        action_type = random.choice([0, 1, 2])
        if action_type == 0:  # Hold
            return TradingAction(0)
        
        position_size = random.choice(self.position_sizes)
        stop_loss = random.choice(self.stop_losses)
        take_profit = random.choice(self.take_profits)
        return TradingAction(action_type, position_size, stop_loss, take_profit)
        
    def train(self, state, action, reward, next_state, done):
        """Train the agent using experience replay"""
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) >= self.batch_size:
            self._train_batch()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _train_batch(self):
        """Train on a batch of experiences with optimized tensor creation"""
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert lists to numpy arrays first
        states_array = np.array([exp[0].to_array() for exp in batch])
        actions_array = np.array([self._encode_action(exp[1]) for exp in batch])
        rewards_array = np.array([exp[2] for exp in batch])
        next_states_array = np.array([exp[3].to_array() for exp in batch])
        dones_array = np.array([exp[4] for exp in batch])
        
        # Create tensors from numpy arrays
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.FloatTensor(dones_array).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):
        """Update target network weights"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def save(self, filename):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load(self, filename):
        """Load model weights"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
    def _encode_action(self, action):
        """Encode TradingAction to index"""
        if action.action_type == 0:
            return 0
            
        size_idx = self.position_sizes.index(action.position_size)
        sl_idx = self.stop_losses.index(action.stop_loss)
        tp_idx = self.take_profits.index(action.take_profit)
        
        base_idx = action.action_type * (len(self.position_sizes) * len(self.stop_losses) * len(self.take_profits))
        return base_idx + (size_idx * len(self.stop_losses) * len(self.take_profits) + sl_idx * len(self.take_profits) + tp_idx)
        
    def _decode_action(self, action_idx):
        """Decode index to TradingAction"""
        if action_idx == 0:
            return TradingAction(0)
            
        action_type = action_idx // (len(self.position_sizes) * len(self.stop_losses) * len(self.take_profits))
        remaining_idx = action_idx % (len(self.position_sizes) * len(self.stop_losses) * len(self.take_profits))
        
        size_idx = remaining_idx // (len(self.stop_losses) * len(self.take_profits))
        remaining_idx = remaining_idx % (len(self.stop_losses) * len(self.take_profits))
        
        sl_idx = remaining_idx // len(self.take_profits)
        tp_idx = remaining_idx % len(self.take_profits)
        
        return TradingAction(
            action_type,
            self.position_sizes[size_idx],
            self.stop_losses[sl_idx],
            self.take_profits[tp_idx]
        )

def integrate_rl_agent(intraday_df, ticker, agent, env):
    """Enhanced RL integration function"""
    current_state = None
    prev_state = None
    prev_action = None
    
    if len(intraday_df) > 1:
        # Initialize environment with first state
        current_state = env.update_state(intraday_df)
        prev_state = current_state
        prev_action = agent.get_action(current_state)
    
    for i in range(1, len(intraday_df)):
        # Get current slice of data
        df_slice = intraday_df.iloc[:i+1]
        
        # Update state with current data
        current_state = env.update_state(df_slice)
        
        # Get action and execute
        action = agent.get_action(current_state)
        reward, done = env.step(action, df_slice)
        
        # Train agent
        if prev_state is not None and prev_action is not None:
            agent.train(prev_state, prev_action, reward, current_state, done)
        
        prev_state = current_state
        prev_action = action
    
    return agent

def check_signals_with_rl(df, ticker, agent, env):
    """Enhanced signal checking using RL agent"""
    if df is None or len(df) < 2:
        return None
    
    # Update environment state
    current_state = env.update_state(df)
    
    # Get action from agent
    action = agent.get_action(current_state)
    
    # Convert action to signal
    if action.action_type == 1:  # Buy
        return 'buy'
    elif action.action_type == 2:  # Sell
        return 'sell'
    
    return None