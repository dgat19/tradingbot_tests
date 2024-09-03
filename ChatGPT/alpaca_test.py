import yfinance as yf
import requests
import numpy as np
import ta
import gym
from gym import spaces
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from alpaca_trade_api.rest import REST
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from cvxopt import matrix, solvers
from datetime import datetime

# API keys and configuration
NEWSAPI_KEY = 'bcec8fa2304344b5892af472fab2a6b0'
ALPACA_API_KEY = 'PA3NRFGUO5AU'
ALPACA_SECRET_KEY = 'PKLLPAIZVPAFBTCF72XM'
BASE_URL = 'https://paper-api.alpaca.markets'

# Data collection functions
def get_news_data(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    return response.json()

def get_social_mentions(ticker):
    return {"ticker": ticker, "mentions": np.random.randint(50, 200)}

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    return sentiment['compound']

def collect_data(tickers):
    news_sentiments = {}
    for ticker in tickers:
        news_data = get_news_data(ticker)
        sentiment_scores = [analyze_sentiment(article['title']) for article in news_data['articles']]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        mentions = get_social_mentions(ticker)['mentions']
        news_sentiments[ticker] = (avg_sentiment, mentions)
    return news_sentiments

# Gym environment definition
class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.current_step = 0
        self.done = False
        
        self.action_space = spaces.Discrete(2)  # Buy or sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        self.done = False
        return self._next_observation()
    
    def _next_observation(self):
        return self.data.iloc[self.current_step].values
    
    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
        reward = self._calculate_reward(action)
        return self._next_observation(), reward, self.done, {}
    
    def _calculate_reward(self, action):
        return np.random.random()  # Placeholder reward calculation
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Prepare data for RL environment
def prepare_rl_data(ticker):
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2023-01-01', end=today)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD_diff'] = ta.trend.MACD(data['Close']).macd_diff()
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=14).sma_indicator()
    return data.dropna()

def optimize_portfolio(returns):
    n = len(returns.columns)
    cov_matrix = np.cov(returns.T)

    P = matrix(cov_matrix)
    q = matrix(np.zeros(n))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    return weights

def execute_trade(ticker, signal):
    alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)
    try:
        response = alpaca.submit_order(
            symbol=ticker,
            qty=10,
            side='buy' if signal == 1 else 'sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Order Response: {response}")
    except Exception as e:
        print(f"Error executing trade: {e}")

def main():
    tickers = ["AAPL", "TSLA", "MSFT", "AMZN"]
    collected_data = collect_data(tickers)
    selected_stocks = [ticker for ticker, sentiment in collected_data.items() if sentiment[0] > 0.5 and sentiment[1] > 100]

    for stock in selected_stocks:
        data = prepare_rl_data(stock)
        env = TradingEnvironment(data)
        vec_env = DummyVecEnv([lambda: env])
        model = A2C('MlpPolicy', vec_env, verbose=1)
        model.learn(total_timesteps=10000)
        
        weights = optimize_portfolio(data.pct_change().dropna())
        print(f"Optimized Portfolio Weights for {stock}: {weights}")

        last_signal = model.predict(env.reset())[0]
        execute_trade(stock, last_signal)

main()

