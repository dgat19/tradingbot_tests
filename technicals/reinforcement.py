import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from collections import deque
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'trading_rl_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

class DQN(nn.Module):
    """Deep Q-Network optimized for trading"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Enhance network architecture with additional layer and dropout
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
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
        """Calculate market context scores with proper type handling"""
        trend_score = self.market_features.get('trend', 0) / 10  # Normalize to -1 to 1
        volatility_score = min(1.0, self.market_features.get('volatility', 0) / 20)
        
        # Use momentum_5 as primary momentum indicator
        momentum_score = self.market_features.get('momentum_5', 0) / 10
            
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
    def __init__(self, initial_balance=100000.0, logger=None):
        self.initial_balance = initial_balance
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
        
        # Add performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        
    def reset(self):
        """Reset the environment with performance metrics and logging flag"""
        self.balance = self.initial_balance
        self.position = None
        self.trade_history = []
        self.market_features = {}
        self.current_state = None
        self.position_entry_price = None
        self.peak_balance = self.initial_balance
        # Reset the flag for data warning each cycle
        self._data_warning_logged = False
        return self.current_state

    def get_position_entry_price(self):
        """Safe getter for position entry price"""
        if self.position and 'entry_price' in self.position:
            return self.position['entry_price']
        return None
        
    def calculate_market_features(self, df, lookback_window=20):
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.warning("Invalid DataFrame provided to calculate_market_features")
            return {}
                
        if len(df) < lookback_window:
            if not getattr(self, '_data_warning_logged', False):
                self.logger.log_info(f"Insufficient data for lookback window of {lookback_window}")
                self._data_warning_logged = True
            return {}
                
        try:
            current_price = float(df['close'].iloc[-1])
            lookback_prices = df['close'].tail(lookback_window)
            
            # Calculate returns and volatility
            returns = df['close'].pct_change()
            volatility = float(returns.std() * np.sqrt(252))  # Annualized volatility
            
            # Calculate momentum scores with safety checks
            momentum_5 = 0.0
            momentum_20 = 0.0
            
            if len(df) > 5:
                momentum_5 = float((df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100)
            if len(df) > 20:
                momentum_20 = float((df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100)
            
            # Volume analysis with safety checks
            volume_sma = df['volume'].rolling(window=lookback_window).mean()
            volume_ratio = 1.0
            if not volume_sma.empty and volume_sma.iloc[-1] > 0:
                volume_ratio = float(df['volume'].iloc[-1] / volume_sma.iloc[-1])
            
            # Price channel analysis
            upper_channel = df['high'].rolling(lookback_window).max()
            lower_channel = df['low'].rolling(lookback_window).min()
            channel_position = 0.5
            if not upper_channel.empty and not lower_channel.empty:
                channel_range = upper_channel.iloc[-1] - lower_channel.iloc[-1]
                if channel_range > 0:
                    channel_position = float((current_price - lower_channel.iloc[-1]) / channel_range)
            
            # Trend strength with safety check
            di_strength = 0.0
            if 'DI+' in df.columns and 'DI-' in df.columns:
                di_strength = float(abs(df['DI+'].iloc[-1] - df['DI-'].iloc[-1]))
            
            # Build features dictionary with safety
            features = {
                'trend': float((current_price - lookback_prices.mean()) / lookback_prices.mean() * 100),
                'volatility': float(volatility * 100),
                'momentum_5': momentum_5,
                'momentum_20': momentum_20,
                'volume_ratio': volume_ratio,
                'channel_position': channel_position,
                'di_strength': di_strength
            }
            
            # Calculate risk score with safety
            risk_components = [
                volatility / 0.4,  # Normalize volatility
                abs(features['trend']) / 20,  # Trend extremity
                (volume_ratio - 1) / 2,  # Volume spike
                abs(channel_position - 0.5) * 2  # Channel position
            ]
            
            features['risk_score'] = float(min(1.0, sum(risk_components) / len(risk_components)))
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating market features: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
        
    def update_state(self, df):
        """Update environment state with market data"""
        if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 2:
            return None
            
        try:
            current_price = float(df['close'].iloc[-1])
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
                di_plus=float(df['DI+'].iloc[-1]),
                di_minus=float(df['DI-'].iloc[-1]),
                price=current_price,
                market_features=self.market_features,
                position=position_info
            )

            return self.current_state
            
        except Exception as e:
            print(f"Error updating state: {e}")
            import traceback
            print(traceback.format_exc())
            return None
        
    def step(self, action, df):
        """Execute trading action and return reward"""
        if df.empty or len(df) < 2:
            return 0, False
            
        current_price = float(df['close'].iloc[-1])
        reward = 0
        done = False
        self.market_features = self.calculate_market_features(df)
        
        if action.action_type == 1:  # Buy
            if self.position is None:
                self.position = {
                    'entry_price': current_price,
                    'size': float(action.position_size),
                    'stop_loss': current_price * (1 - action.stop_loss/100),
                    'take_profit': current_price * (1 + action.take_profit/100),
                    'entry_bar': len(df) - 1,
                    'entry_features': self.market_features.copy()
                }
                self.position_entry_price = current_price  # Set entry price explicitly
                reward = -0.0001  # Small cost for taking position
                
        elif action.action_type == 2:  # Sell
            if self.position is not None:
                # Calculate profit/loss
                pl_pct = ((current_price - self.position['entry_price']) / 
                         self.position['entry_price'] * 100)
                
                # Dynamic reward calculation
                reward = self._calculate_dynamic_reward(pl_pct, self.market_features.get('risk_score', 0.5))
                
                # Update balance and record trade
                self.balance += self.position['size'] * current_price
                self._record_trade(current_price, pl_pct, len(df))
                
                self.position = None
                self.position_entry_price = None  # Clear entry price
                done = True
                
        return reward, done
        
    def _calculate_dynamic_reward(self, pl_pct, risk_score):
        """Calculate reward based on profit and market features"""
        if pl_pct > 0:
            # Exponential reward scaled by risk
            base_reward = np.exp(pl_pct/5) - 1
            risk_adjusted_reward = base_reward * (1 - risk_score)
            
            # Bonus for low-risk wins
            if risk_score < 0.3 and pl_pct > 1.0:
                risk_adjusted_reward *= 1.5
                
            return risk_adjusted_reward
        else:
            # Penalize losses more in high-risk situations
            risk_multiplier = 1 + risk_score
            return pl_pct * risk_multiplier
        
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
        """Enhanced trade recording with performance metrics"""
        if self.position is None:
            return
            
        holding_time = current_bar - self.position['entry_bar']
        trade_record = {
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'pl_pct': pl_pct,
            'size': self.position['size'],
            'holding_time': holding_time,
            'entry_features': self.position['entry_features'],
            'exit_features': self.market_features.copy(),
            'timestamp': datetime.now()
        }
        # Update performance metrics
        self.total_trades += 1
        if pl_pct > 0:
            self.successful_trades += 1
        self.total_profit += pl_pct
        
        # Update maximum drawdown
        self.balance = self.balance * (1 + pl_pct/100)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self.trade_history.append(trade_record)
        self.logger.log_info(f"Trade recorded: {trade_record}")

        
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
        
    def update_target_network(self):
        """Update target network weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def update_target_model(self):
        """Alias for update_target_network for backward compatibility"""
        self.update_target_network()
        
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

def integrate_rl_agent(intraday_df, ticker, agent, env, signal_generator, trading_plotter):
    logger = logging.getLogger(__name__)
    
    # Initialize environment with the full data slice
    current_state = env.update_state(intraday_df)
    if current_state is None:
        logger.info(f"Initial state is None for ticker {ticker}; skipping RL integration.")
        return agent

    prev_state = current_state
    prev_action = agent.get_action(current_state)
    
    # Loop over the data slice to integrate experience
    for i in range(1, len(intraday_df)):
        df_slice = intraday_df.iloc[:i+1]
        current_state = env.update_state(df_slice)
        if current_state is None:
            continue
        
        # Get a base signal using the shared signal generator.
        base_signal = signal_generator.check_signals(df_slice, ticker)
        # Optionally, obtain signal strength if needed.
        signal_strength = signal_generator.get_signal_strength(df_slice)
        
        # If a base signal is generated, record a virtual trade.
        if base_signal:
            virtual_trade = {
                'ticker': ticker,
                'side': base_signal,  # 'buy' or 'sell'
                'time': df_slice.index[-1],
                'price': df_slice['close'].iloc[-1],
                'quantity': 0,  # Virtual trade; quantity is 0 (for visualization only)
                'signal_strength': signal_strength,
                'virtual': True
            }
            trading_plotter.add_trade(virtual_trade)
            logger.info(f"{ticker}: Virtual RL signal recorded as '{base_signal}' at price {virtual_trade['price']:.2f}")
        
        # Get an action using the RL agent's policy (with epsilon-greedy exploration)
        is_training = random.random() < agent.epsilon
        action = agent.get_action(current_state, training=is_training)
        
        reward, done = env.step(action, df_slice)
        
        # Train the agent with experience replay
        if prev_state is not None and prev_action is not None:
            agent.train(prev_state, prev_action, reward, current_state, done)
            
            # Optionally update the target network periodically
            if i % 100 == 0:
                agent.update_target_model()
        
        prev_state = current_state
        prev_action = action

    logger.info(f"Completed training on {ticker} - Episodes: {len(intraday_df)}, Final Epsilon: {agent.epsilon:.3f}")
    return agent

def check_signals_with_rl(df, ticker, agent, env):
    """Enhanced signal checking with dynamic thresholds"""
    logger = logging.getLogger(__name__)
    
    if df is None or len(df) < 2:
        return None
    
    try:
        current_state = env.update_state(df)
        market_features = env.calculate_market_features(df)
        risk_score = market_features.get('risk_score', 0.5)
        
        # Get action with exploration disabled
        action = agent.get_action(current_state, training=False)
        
        # Dynamic risk thresholds based on market conditions
        volatility = market_features.get('volatility', 20)
        risk_threshold = 0.7 - (0.1 if volatility < 15 else 0)  # Lower threshold for low volatility
        
        if action.action_type == 1:  # Buy
            # Enhanced risk checks
            if (risk_score > risk_threshold or
                volatility > 40 or
                market_features.get('volume_ratio', 1) < 0.5):
                logger.info(f"{ticker}: Buy signal rejected due to risk factors")
                return None
                
            momentum_short = market_features.get('momentum_5', 0)
            momentum_long = market_features.get('momentum_20', 0)
            
            # Momentum confirmation
            if momentum_short >= 0 or momentum_long >= 0:
                logger.info(f"{ticker}: Buy signal generated with momentum confirmation")
                return 'buy'
            
        elif action.action_type == 2:  # Sell
            position_price = env.get_position_entry_price()
            if position_price:
                current_price = df['close'].iloc[-1]
                pl_pct = ((current_price - position_price) / position_price) * 100
                
                # Dynamic exit thresholds
                stop_loss = -2.0 * (1 + risk_score)
                take_profit = min(5.0, volatility)
                
                if pl_pct <= stop_loss or pl_pct >= take_profit:
                    logger.info(f"{ticker}: Sell signal generated due to price targets")
                    return 'sell'
            
            logger.info(f"{ticker}: Sell signal generated")
            return 'sell'
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking signals for {ticker}: {str(e)}")
        return None

def calculate_position_size(account_cash, current_price, risk_score, confidence):
    """Calculate position size with enhanced risk management"""
    # Base position size
    max_position_pct = 0.5  # Maximum 50% of account
    
    # Risk-adjusted position size
    risk_factor = 1 - risk_score
    volatility_factor = max(0.3, min(1.0, 1 - risk_score))
    
    # Calculate final position size
    position_pct = max_position_pct * risk_factor * confidence * volatility_factor
    
    # Additional safety caps
    if risk_score > 0.7:
        position_pct *= 0.5  # Halve position size for high risk
    if confidence < 0.4:
        position_pct *= 0.5  # Halve position size for low confidence
        
    # Calculate actual position
    position_dollars = account_cash * position_pct
    qty = position_dollars / current_price
    
    return round(qty, 3)

# Export necessary components
__all__ = [
    'DQN',
    'TradingAction',
    'EnhancedTradingState',
    'EnhancedTradingEnvironment',
    'EnhancedTradingAgent',
    'integrate_rl_agent',
    'check_signals_with_rl'
]