import numpy as np
from collections import deque
import random
import pickle
from datetime import datetime, timedelta

class TradingState:
    """
    Represents the state of the trading environment
    """
    def __init__(self, di_plus, di_minus, price_change, current_pl=0, position=0):
        self.di_plus = di_plus
        self.di_minus = di_minus
        self.price_change = price_change
        self.current_pl = current_pl  # Current profit/loss if in position
        self.position = position      # 1 for long, 0 for no position
        
    def to_array(self):
        """Convert state to array for NN input"""
        return np.array([
            self.di_plus,
            self.di_minus,
            self.price_change,
            self.current_pl,
            self.position
        ])
        
    def discretize(self):
        """Convert continuous state to discrete for Q-table"""
        di_plus_disc = int(self.di_plus / 10)  # Discretize to 10-point buckets
        di_minus_disc = int(self.di_minus / 10)
        price_change_disc = int((self.price_change + 5) / 0.5)  # 0.5% buckets from -5% to +5%
        pl_disc = int((self.current_pl + 5) / 0.5)  # 0.5% buckets from -5% to +5%
        
        return (di_plus_disc, di_minus_disc, price_change_disc, pl_disc, self.position)

class TradingEnvironment:
    """
    Trading environment that tracks state and calculates rewards
    """
    def __init__(self):
        self.reset()
        self.trade_history = []
        
    def reset(self):
        self.current_state = None
        self.position_entry_price = None
        return self.current_state
        
    def update_state(self, di_plus, di_minus, current_price, prev_price):
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Calculate P/L if in position
        current_pl = 0
        position = 0
        if self.position_entry_price is not None:
            position = 1
            current_pl = ((current_price - self.position_entry_price) / self.position_entry_price) * 100
            
        self.current_state = TradingState(
            di_plus=di_plus,
            di_minus=di_minus,
            price_change=price_change,
            current_pl=current_pl,
            position=position
        )
        return self.current_state
        
    def step(self, action, current_price):
        """
        Execute action and return reward
        action: 0 (hold), 1 (buy), 2 (sell)
        """
        reward = 0
        done = False
        
        if action == 1:  # Buy
            if self.position_entry_price is None:
                self.position_entry_price = current_price
                reward = -0.001  # Small penalty for trading cost
                
        elif action == 2:  # Sell
            if self.position_entry_price is not None:
                pl_pct = ((current_price - self.position_entry_price) / self.position_entry_price) * 100
                reward = pl_pct
                self.trade_history.append({
                    'entry_price': self.position_entry_price,
                    'exit_price': current_price,
                    'pl_pct': pl_pct,
                    'timestamp': datetime.now()
                })
                self.position_entry_price = None
                
        return reward, done
        
    def get_state(self):
        return self.current_state

class TradingAgent:
    """
    Q-Learning agent that learns optimal trading strategy
    """
    def __init__(self, action_size=3, learning_rate=0.001, gamma=0.95, epsilon=1.0):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize Q-table
        self.q_table = {}
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
            
        state_key = state.discretize()
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
            
        return np.argmax(self.q_table[state_key])
        
    def train(self, state, action, reward, next_state, done):
        """Train the agent using Q-learning"""
        state_key = state.discretize()
        next_state_key = next_state.discretize() if next_state else None
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
            
        if next_state_key and next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
            
        # Q-learning update
        if not done and next_state_key:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
        else:
            target = reward
            
        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filename):
        """Save the Q-table"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filename):
        """Load the Q-table"""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

# Function to integrate RL with existing trading system
def integrate_rl_agent(intraday_df, ticker, agent, env):
    """
    Use RL agent to enhance trading decisions
    """
    current_state = None
    
    for i in range(1, len(intraday_df)):
        prev_row = intraday_df.iloc[i-1]
        curr_row = intraday_df.iloc[i]
        
        # Update environment state
        current_state = env.update_state(
            di_plus=curr_row['DI+'],
            di_minus=curr_row['DI-'],
            current_price=curr_row['close'],
            prev_price=prev_row['close']
        )
        
        # Get action from agent
        action = agent.get_action(current_state)
        
        # Execute action and get reward
        reward, done = env.step(action, curr_row['close'])
        
        # Train agent
        if i > 1:  # Need previous state for training
            agent.train(prev_state, prev_action, reward, current_state, done)
            
        prev_state = current_state
        prev_action = action
        
    return agent

# Modified check_signals function to use RL agent
def check_signals_with_rl(df, ticker, agent, env):
    """
    Enhanced signal checking using RL agent
    """
    if df is None or len(df) < 2:
        return None
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Update environment state
    current_state = env.update_state(
        di_plus=curr['DI+'],
        di_minus=curr['DI-'],
        current_price=curr['close'],
        prev_price=prev['close']
    )
    
    # Get action from agent
    action = agent.get_action(current_state)
    
    # Convert action to signal
    if action == 1:  # Buy
        return 'buy'
    elif action == 2:  # Sell
        return 'sell'
    
    return None  # Hold