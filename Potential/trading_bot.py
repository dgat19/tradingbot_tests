# trading_bot.py

import asyncio
import aiohttp  # For asynchronous HTTP requests
import pandas as pd
import logging
import datetime
import os
import threading
import warnings
import requests
from dotenv import load_dotenv

# FastAPI for creating API endpoints
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# PySide6 for GUI
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                               QPushButton, QLineEdit, QDateEdit, QTextEdit,
                               QRadioButton, QHBoxLayout, QButtonGroup)
from PySide6.QtCore import QThread, Signal, QDate
from PySide6.QtGui import QTextCursor

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import xgboost as xgb

# NLTK for sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Additional libraries
import yfinance as yf
import backtrader as bt  # For advanced backtesting
from tradingview_ta import TA_Handler, Interval  # For tradingview-screener
from newsapi import NewsApiClient  # For news data
import re  # For regex
from collections import Counter  # For counting occurrences
from bs4 import BeautifulSoup  # For web scraping
#from stable_baselines3 import PPO  # For reinforcement learning
#from stable_baselines3.common.vec_env import DummyVecEnv
#import gym  # For custom environment

# Disable warnings
warnings.filterwarnings('ignore')

# Ensure that NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)

load_dotenv()
# Configurations
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Validate that the API keys are loaded
assert ALPACA_API_KEY is not None, "ALPACA_API_KEY not set."
assert ALPACA_SECRET_KEY is not None, "ALPACA_SECRET_KEY not set."
assert NEWS_API_KEY is not None, "NEWS_API_KEY not set."

USER_OPTIONS = {
    'trade_risk_management': {
        'max_position_size': 0.1,  # Max 10% of capital per trade
        'risk_per_trade': 0.02,    # Max 2% of capital at risk per trade
    },
    'trade_timeframes': ['1Day', '1Hour'],
}
GENERAL_SETTINGS = {
    'log_level': 'INFO',
}

# Set up logging
log_level_str = GENERAL_SETTINGS.get('log_level', 'INFO')
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
logging.basicConfig(
    filename='tradingbot.log',
    level=log_level,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Helper function to check if model exists
def check_model_exists(model_path):
    return os.path.exists(model_path)

# Helper function to set up logger
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Asynchronous Data Loader
class AsyncDataLoader:
    def __init__(self, alpaca_api_key, alpaca_secret_key, news_api_key):
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.news_api_key = news_api_key

    async def get_most_active_tickers(self, limit=10):
        try:
            url = "https://finance.yahoo.com/markets/stocks/most-active/"
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = 'utf-8'  # Set the encoding to handle special characters
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stock_list = []
            
            # Find the table containing the most active stocks
            table = soup.find('tbody', {'class': 'body yf-1dbt8wv'})
            if table:
                rows = table.find_all('tr')  # Get all rows
                for row in rows:
                    cols = row.find_all('td')
                    if cols:
                        symbol = cols[0].find('span', {'class': 'symbol'}).text.strip()
                        change_percent = cols[3].find('fin-streamer').text.strip()
                        volume = cols[4].find('fin-streamer').text.strip()
                        avg_volume = cols[5].text.strip()
                        stock_list.append({
                            'symbol': symbol,
                            'change_percent': change_percent,
                            'volume': volume,
                            'avg_volume': avg_volume
                        })
            return stock_list
        except Exception as e:
            print(f"Error fetching top active movers: {str(e)}")
            return []

    async def get_trending_tickers_from_reddit(self, limit=10):
        url = 'https://api.swaggystocks.com/wsb/sentiment/top'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            tickers = [item['ticker'] for item in data[:limit]]
            return tickers
        else:
            print(f"Error fetching data from SwaggyStocks: {response.status_code}")
            return []

    async def extract_tickers_from_articles(self, limit=10):
        newsapi = NewsApiClient(api_key=self.news_api_key)
        all_articles = newsapi.get_top_headlines(language='en')
        tickers = []
        for article in all_articles['articles']:
            content = article.get('title', '') + ' ' + article.get('description', '')
            # Simple regex to find words in uppercase (possible tickers)
            found_tickers = re.findall(r'\b[A-Z]{2,5}\b', content)
            tickers.extend(found_tickers)
        # Count occurrences and get the most frequent tickers
        ticker_counts = Counter(tickers)
        most_common_tickers = [ticker for ticker, count in ticker_counts.most_common(limit)]
        return most_common_tickers

    async def get_trending_tickers(self):
        yahoo_tickers = await self.get_most_active_tickers()
        reddit_tickers = await self.get_trending_tickers_from_reddit()
        newsapi_tickers = await self.extract_tickers_from_articles()
        
        # Ensure none of the ticker lists are None
        yahoo_tickers = yahoo_tickers or []
        reddit_tickers = reddit_tickers or []
        newsapi_tickers = newsapi_tickers or []
        
        # Remove any None values from the lists
        yahoo_tickers = [ticker for ticker in yahoo_tickers if ticker]
        reddit_tickers = [ticker for ticker in reddit_tickers if ticker]
        newsapi_tickers = [ticker for ticker in newsapi_tickers if ticker]
        
        combined_tickers = set(yahoo_tickers + reddit_tickers + newsapi_tickers)
        return list(combined_tickers)

    async def fetch_stock_data(self, session, symbol, start_date, end_date):
        """
        Asynchronously fetches historical stock data.
        """
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        headers = {
            'APCA-API-KEY-ID': self.alpaca_api_key,
            'APCA-API-SECRET-KEY': self.alpaca_secret_key
        }
        params = {
            'start': start_date,
            'end': end_date,
            'timeframe': '1Day'
        }
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'bars' in data:
                    df = pd.DataFrame(data['bars'])
                    if not df.empty:
                        df['t'] = pd.to_datetime(df['t'])
                        df.set_index('t', inplace=True)
                        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                    return df
                else:
                    print(f"No 'bars' data in response for {symbol}")
                    return pd.DataFrame()
            else:
                error_text = await response.text()
                print(f"HTTP Error {response.status}: {error_text}")
                return pd.DataFrame()

    async def get_stock_data(self, symbols, start_date, end_date):
        """
        Fetches stock data for multiple symbols concurrently.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_stock_data(session, symbol, start_date, end_date) for symbol in symbols
            ]
            results = await asyncio.gather(*tasks)
            stock_data = dict(zip(symbols, results))
            return stock_data

    async def fetch_news(self, session, ticker):
        """
        Asynchronously fetches news articles for a given ticker.
        """
        url = ('https://newsapi.org/v2/everything?'
               'q={}&'
               'language=en&'
               'sortBy=publishedAt&'
               'apiKey={}').format(ticker, self.news_api_key)
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                articles = data.get('articles', [])
                return articles
            else:
                error_text = await response.text()
                print(f"HTTP Error {response.status}: {error_text}")
                return []

    async def get_news_data(self, tickers):
        """
        Fetches news data for multiple tickers concurrently.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_news(session, ticker) for ticker in tickers
            ]
            results = await asyncio.gather(*tasks)
            news_data = dict(zip(tickers, results))
            return news_data

# Indicator Calculator
class IndicatorCalculator:
    def calculate(self, data):
        """
        Computes technical indicators and adds them to the data DataFrame.
        """
        data = data.copy()
        data['SMA'] = self.calculate_sma(data['Close'], window=20)
        data['EMA'] = self.calculate_ema(data['Close'], span=20)
        data['RSI'] = self.calculate_rsi(data['Close'])
        macd_df = self.calculate_macd(data['Close'])
        data = pd.concat([data, macd_df], axis=1)
        bollinger_df = self.calculate_bollinger_bands(data['Close'])
        data = pd.concat([data, bollinger_df], axis=1)
        data.dropna(inplace=True)
        return data

    def calculate_sma(self, data, window):
        return data.rolling(window=window).mean()

    def calculate_ema(self, data, span):
        return data.ewm(span=span, adjust=False).mean()

    def calculate_rsi(self, data, period=14):
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        df = pd.DataFrame({
            'MACD_Line': macd_line,
            'Signal_Line': signal_line,
            'MACD_Histogram': macd_histogram
        })
        return df

    def calculate_bollinger_bands(self, data, window=20, num_std_dev=2):
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)
        df = pd.DataFrame({
            'Upper_Band': upper_band,
            'Middle_Band': middle_band,
            'Lower_Band': lower_band
        })
        return df

    def fetch_tradingview_indicators(self, symbol):
        """
        Fetches technical indicators from TradingView using tradingview-ta.
        """
        handler = TA_Handler(
            symbol=symbol,
            exchange="NASDAQ",
            screener="america",
            interval=Interval.INTERVAL_1_DAY,
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators
        return indicators

# Sentiment Analyzer
class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.sia = SentimentIntensityAnalyzer()
        self.newsapi = NewsApiClient(api_key=self.api_key)

    async def analyze_sentiment(self, tickers):
        sentiment_data = []
        for ticker in tickers:
            sentiments = []
            # Analyze sentiment from NewsAPI articles
            try:
                response = self.newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt')
                articles = response.get('articles', [])
                for article in articles:
                    title = article.get('title') or ''
                    description = article.get('description') or ''
                    content = title + ' ' + description
                    sentiment_score = self.sia.polarity_scores(content)['compound']
                    sentiments.append(sentiment_score)
            except Exception as e:
                print(f"Error fetching articles for {ticker}: {e}")
            # Analyze sentiment from Reddit (SwaggyStocks)
            reddit_sentiment = self.get_reddit_sentiment(ticker)
            sentiments.append(reddit_sentiment)
            # Calculate average sentiment
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
            else:
                avg_sentiment = 0
            sentiment_data.append({
                'Ticker': ticker,
                'Sentiment': avg_sentiment
            })
        df = pd.DataFrame(sentiment_data)
        return df

    def get_reddit_sentiment(self, ticker):
        # Fetch sentiment data from SwaggyStocks or other Reddit APIs
        url = f'https://api.swaggystocks.com/wsb/sentiment/{ticker}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                sentiment_score = data.get('sentiment_score', 0)
            else:
                sentiment_score = 0
            return sentiment_score
        else:
            return 0


# Feature Engineer
class FeatureEngineer:
    def create_features(self, data, indicators, sentiments, tradingview_indicators):
        """
        Combines data, indicators, sentiments, and TradingView indicators into a feature set.
        """
        data = data.copy()
        # Merge indicators
        data = pd.concat([data, indicators], axis=1)

        # Merge TradingView indicators
        tv_indicators_df = pd.DataFrame(tradingview_indicators, index=[data.index[-1]])
        data = pd.concat([data, tv_indicators_df], axis=1)

        # Merge sentiments
        if sentiments is not None:
            data = data.merge(sentiments, left_on='symbol', right_on='Ticker', how='left')
            data['Sentiment'].fillna(0, inplace=True)
            data.drop('Ticker', axis=1, inplace=True)
        else:
            data['Sentiment'] = 0

        # Fill missing values
        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)

        # Define target variable
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        # Extract features and labels
        features = data.drop(columns=['Target', 'symbol'])
        labels = data['Target']

        return features, labels

# Model Trainer
class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train(self, features, labels):
        # Scale features
        features = self.scaler.fit_transform(features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        # Fit model on training data
        self.model.fit(X_train, y_train)

        # Evaluate model on test data
        y_pred = self.model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return self.model

    def save_model(self, model, model_path):
        dump((model, self.scaler), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        self.model, self.scaler = load(model_path)
        print(f"Model loaded from {model_path}")
        return self.model

# Trading Logic
class TradingLogic:
    def __init__(self, alpaca_api_key, alpaca_secret_key):
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.base_url = 'https://paper-api.alpaca.markets'  # For paper trading
        # Initialize Alpaca REST API
        import alpaca_trade_api as tradeapi
        self.api = tradeapi.REST(
            self.alpaca_api_key,
            self.alpaca_secret_key,
            self.base_url
        )

    def calculate_position_size(self, win_probability, win_loss_ratio, capital):
        """
        Calculates position size using the Kelly Criterion and adjusts based on account balance.
        """
        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)
        position_size = capital * kelly_fraction * USER_OPTIONS['trade_risk_management']['max_position_size']
        # Ensure position size does not exceed available cash
        position_size = min(position_size, capital)
        return max(0, position_size)  # Ensure non-negative

    def get_account_capital(self):
        account = self.api.get_account()
        return float(account.cash)

    def select_best_option(self, symbol, expiration_date, option_type):
        """
        Selects the best option contract based on criteria such as open interest and implied volatility.
        """
        stock = yf.Ticker(symbol)
        try:
            option_chain = stock.option_chain(expiration_date)
            options_df = option_chain.calls if option_type.lower() == 'call' else option_chain.puts
            # Filter options based on criteria
            options_df = options_df[options_df['inTheMoney'] == False]
            # Sort by highest open interest
            options_df = options_df.sort_values(by='openInterest', ascending=False)
            if not options_df.empty:
                best_option = options_df.iloc[0]
                return best_option
            else:
                print("No suitable option contracts found.")
                return None
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return None

    def execute_trade(self, order_details, is_backtest=False):
        """
        Execute the trade using Alpaca's API.
        """
        if is_backtest:
            # Simulate trade execution
            print(f"Simulated trade execution: {order_details}")
            return
        else:
            symbol = order_details['symbol']
            side = order_details['side']
            expiration_date = order_details['expiration_date']
            strike_price = order_details['strike_price']
            option_type = order_details['option_type']
            win_probability = order_details.get('win_probability', 0.5)
            win_loss_ratio = order_details.get('win_loss_ratio', 1)
            capital = self.get_account_capital()
            position_size = self.calculate_position_size(win_probability, win_loss_ratio, capital)

            # Get the best option contract
            best_option = self.select_best_option(symbol, expiration_date, option_type)
        if best_option is not None:
            option_symbol = best_option['contractSymbol']
            option_price = best_option['lastPrice']
            # Options are traded in contracts of 100 shares
            quantity = int(position_size // (option_price * 100))
            if quantity > 0:
                try:
                    order = self.api.submit_order(
                        symbol=option_symbol,
                        qty=quantity,
                        side=side,
                        type='market',
                        time_in_force='gtc',
                    )
                    print(f"Order submitted: {order}")
                    return order
                except Exception as e:
                    print(f"Error submitting order: {e}")
                    return None
            else:
                print("Calculated quantity is zero. Trade not executed.")
                return None
        else:
            print("No suitable option contract found. Trade not executed.")
            return None

    def construct_option_symbol(self, symbol, expiration_date, option_type, strike_price):
        """
        Constructs the OCC option symbol.
        """
        if None in [symbol, expiration_date, option_type, strike_price]:
            print("Error: One of the parameters to construct_option_symbol is None")
            return None
        try:
            # Convert expiration_date to YYMMDD
            expiration_datetime = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
            expiration_str = expiration_datetime.strftime('%y%m%d')

            # Option type
            option_type_char = 'C' if option_type.lower() == 'call' else 'P'

            # Strike price * 1000, padded to 8 digits
            strike_price_int = int(float(strike_price) * 1000)
            strike_price_str = f"{strike_price_int:08d}"

            option_symbol = f"{symbol}{expiration_str}{option_type_char}{strike_price_str}"

            return option_symbol
        except Exception as e:
            print(f"Error constructing option symbol: {e}")
            return None

    def get_option_quote(self, symbol, expiration_date, strike_price, option_type):
        """
        Fetches the option quote for the specified option contract using yfinance.
        """
        stock = yf.Ticker(symbol)
        try:
            option_chain = stock.option_chain(expiration_date)
            if option_type.lower() == 'call':
                options_df = option_chain.calls
            else:
                options_df = option_chain.puts
            # Filter for the specific strike price
            option = options_df[options_df['strike'] == float(strike_price)]
            if not option.empty:
                return option.iloc[0]
            else:
                print("Option contract not found.")
                return None
        except Exception as e:
            print(f"Error fetching option quote: {e}")
            return None

    def get_model_predictions(self):
        """
        Placeholder function to get model predictions.
        Replace with actual model integration.
        """
        # Example prediction
        return {
            'symbol': 'AAPL',
            'prediction': 'buy',
            'confidence': 0.8,
            'expiration_date': '2023-12-17',
            'strike_price': 150,
            'option_type': 'call'
        }

    def determine_trade_entry(self, model_predictions):
        """
        Determine whether to enter a trade based on model predictions and account balance.
        Returns order details if a trade should be executed.
        """
        confidence_threshold = 0.7
        capital = self.get_account_capital()
        min_required_capital = 500  # Minimum capital required to place a trade

        if (model_predictions['confidence'] >= confidence_threshold) and (capital >= min_required_capital):
            order_details = {
                'symbol': model_predictions['symbol'],
                'side': model_predictions['prediction'],  # 'buy' or 'sell'
                'expiration_date': model_predictions['expiration_date'],
                'strike_price': model_predictions['strike_price'],
                'option_type': model_predictions['option_type'],
                'win_probability': model_predictions['confidence'],
                'win_loss_ratio': 1  # Adjust based on strategy
            }
            return order_details
        else:
            if capital < min_required_capital:
                print("Insufficient capital to execute trade.")
            return None

    def monitor_positions(self):
        """
        Monitors active positions and applies adaptive stop-loss and take-profit.
        """
        positions = self.api.list_positions()

        for position in positions:
            if position.asset_class == 'option':
                symbol = position.symbol
                qty = int(float(position.qty))
                current_price = float(position.current_price)
                entry_price = float(position.avg_entry_price)
                volatility = self.calculate_volatility(symbol)
                sentiment_score = self.get_sentiment_score(symbol)

                stop_loss, take_profit = self.determine_stop_loss_take_profit(volatility, sentiment_score)

                # Calculate P&L percentages
                pnl_pct = (current_price - entry_price) / entry_price

                if pnl_pct >= take_profit or pnl_pct <= -stop_loss:
                    self.exit_trade(symbol, qty)

    def calculate_volatility(self, symbol):
        # Placeholder for volatility calculation
        return 0.2  # Example volatility

    def get_sentiment_score(self, symbol):
        # Placeholder for sentiment score
        return 0.1  # Example sentiment score

    def determine_stop_loss_take_profit(self, volatility, sentiment_score):
        """
        Adjusts stop-loss and take-profit levels based on market volatility and sentiment.
        """
        base_stop_loss = 0.02  # 2%
        base_take_profit = 0.05  # 5%

        # Adjust based on volatility and sentiment
        adjusted_stop_loss = base_stop_loss * (1 + volatility)
        adjusted_take_profit = base_take_profit * (1 + sentiment_score)

        return adjusted_stop_loss, adjusted_take_profit

    def exit_trade(self, symbol, qty):
        """
        Close the position by submitting a market order in the opposite direction.
        """
        side = 'sell' if qty > 0 else 'buy'
        qty = abs(qty)

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc',
            )
            print(f"Exit order submitted: {order}")
            return order
        except Exception as e:
            print(f"Error submitting exit order: {e}")
            return None

# Backtrader Strategy
class SentimentStrategy(bt.Strategy):
    params = (
        ('stop_loss', 0.03),  # 3% stop loss
        ('take_profit', 0.05),  # 5% take profit
    )

    def __init__(self):
        self.order = None  # To keep track of pending orders
        self.buy_price = None
        self.rsi = bt.indicators.RSI(self.data.close)
        self.sma = bt.indicators.SMA(self.data.close, period=20)

    def log(self, text, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {text}")

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price}")
                self.buy_price = order.executed.price
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price}")

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
            self.order = None

    def next(self):
        if self.order:
            return  # If an order is pending, do nothing

        if not self.position:
            # Entry condition: RSI below 30 and price above SMA
            if self.rsi < 30 and self.data.close[0] > self.sma[0]:
                self.order = self.buy()
        else:
            # Exit conditions
            current_price = self.data.close[0]
            if (current_price >= self.buy_price * (1 + self.params.take_profit) or
                current_price <= self.buy_price * (1 - self.params.stop_loss)):
                self.order = self.sell()

# FastAPI Application
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
async def predict(symbol: str):
    """
    API endpoint to get the latest prediction for a given symbol.
    """
    try:
        # Load the model
        model_path = f'model_{symbol}.joblib'
        model_trainer = ModelTrainer()
        if check_model_exists(model_path):
            model_trainer.load_model(model_path)
        else:
            return {"error": "Model not found. Please train the model first."}

        # Fetch data
        data_loader = AsyncDataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY, NEWS_API_KEY)
        stock_data = await data_loader.get_stock_data([symbol], '2023-01-01', '2023-12-31')
        data = stock_data[symbol]
        if data.empty:
            return {"error": f"No data fetched for {symbol}."}

        # Compute indicators
        indicator_calculator = IndicatorCalculator()
        indicators = indicator_calculator.calculate(data)

        # Fetch TradingView indicators
        tradingview_indicators = indicator_calculator.fetch_tradingview_indicators(symbol)

        # Analyze sentiments
        sentiment_analyzer = SentimentAnalyzer(NEWS_API_KEY)
        sentiments_df = await sentiment_analyzer.analyze_sentiment([symbol])
        sentiments_df.set_index('Ticker', inplace=True)
        sentiments = sentiments_df[['Sentiment']]

        # Create features
        feature_engineer = FeatureEngineer()
        features, _ = feature_engineer.create_features(
            data, indicators, sentiments, tradingview_indicators
        )
        # Scale features
        features_scaled = model_trainer.scaler.transform(features.tail(1))
        # Predict
        prediction = model_trainer.model.predict(features_scaled)
        confidence = model_trainer.model.predict_proba(features_scaled)[0][prediction[0]]

        return {
            "symbol": symbol,
            "prediction": int(prediction[0]),
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return {"error": str(e)}

# PySide6 GUI Application
class TradingBotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Initialize components
        self.data_loader = AsyncDataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY, NEWS_API_KEY)
        self.indicator_calculator = IndicatorCalculator()
        self.sentiment_analyzer = SentimentAnalyzer(NEWS_API_KEY)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.trading_logic = TradingLogic(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    def init_ui(self):
        self.setWindowTitle('Automated Trading Bot')

        self.layout = QVBoxLayout()

        # Trading Mode Selection
        self.mode_label = QLabel('Select Trading Mode:')
        self.live_mode_radio = QRadioButton('Live Trading')
        self.backtest_mode_radio = QRadioButton('Backtesting')
        self.live_mode_radio.setChecked(True)  # Default to Live Trading

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.live_mode_radio)
        self.mode_group.addButton(self.backtest_mode_radio)

        self.mode_layout = QHBoxLayout()
        self.mode_layout.addWidget(self.live_mode_radio)
        self.mode_layout.addWidget(self.backtest_mode_radio)

        self.layout.addWidget(self.mode_label)
        self.layout.addLayout(self.mode_layout)

        # Start date input (only for Backtesting)
        self.start_date_label = QLabel('Start Date:')
        self.start_date_input = QDateEdit(QDate.currentDate().addYears(-1))
        self.layout.addWidget(self.start_date_label)
        self.layout.addWidget(self.start_date_input)

        # End date input (only for Backtesting)
        self.end_date_label = QLabel('End Date:')
        self.end_date_input = QDateEdit(QDate.currentDate())
        self.layout.addWidget(self.end_date_label)
        self.layout.addWidget(self.end_date_input)

        # Polling interval input
        self.polling_interval_label = QLabel('Polling Interval (seconds):')
        self.polling_interval_input = QLineEdit('60')
        self.layout.addWidget(self.polling_interval_label)
        self.layout.addWidget(self.polling_interval_input)

        # Start and Stop buttons
        self.start_button = QPushButton('Start Trading')
        self.start_button.clicked.connect(self.start_trading)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Trading')
        self.stop_button.clicked.connect(self.stop_trading)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        # Output console
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.layout.addWidget(self.output_console)

        self.setLayout(self.layout)

        # Connect mode selection change to update UI
        self.live_mode_radio.toggled.connect(self.update_ui_for_mode)

        # Initialize UI based on default mode
        self.update_ui_for_mode()

    def update_ui_for_mode(self):
        if self.live_mode_radio.isChecked():
            # Hide date inputs for Live Trading
            self.start_date_label.hide()
            self.start_date_input.hide()
            self.end_date_label.hide()
            self.end_date_input.hide()
        else:
            # Show date inputs for Backtesting
            self.start_date_label.show()
            self.start_date_input.show()
            self.end_date_label.show()
            self.end_date_input.show()

    def start_trading(self):
        mode = 'live' if self.live_mode_radio.isChecked() else 'backtest'
        polling_interval = int(self.polling_interval_input.text())

        if mode == 'backtest':
            start_date = self.start_date_input.date().toString('yyyy-MM-dd')
            end_date = self.end_date_input.date().toString('yyyy-MM-dd')
        else:
            start_date = None
            end_date = None

        self.thread = TradingThread(
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            polling_interval=polling_interval,
            data_loader=self.data_loader,
            indicator_calculator=self.indicator_calculator,
            sentiment_analyzer=self.sentiment_analyzer,
            feature_engineer=self.feature_engineer,
            model_trainer=self.model_trainer,
            trading_logic=self.trading_logic
        )
        self.thread.log_signal.connect(self.update_console)
        self.thread.finished.connect(self.trading_finished)
        self.thread.start()

        # Update button states
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_console("Trading started.")

    def stop_trading(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            # Stop the trading thread
            self.thread.stop()
            self.thread.wait()
            self.update_console("Trading stopped.")

            # Update button states
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        else:
            self.update_console("No trading thread is running.")

    def trading_finished(self):
        # Called when the trading thread finishes naturally
        self.update_console("Trading thread has finished.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_console(self, message):
        self.output_console.append(message)
        self.output_console.moveCursor(QTextCursor.End)

# Trading Thread
class TradingThread(QThread):
    log_signal = Signal(str)

    def __init__(self, mode, start_date, end_date, polling_interval,
                 data_loader, indicator_calculator, sentiment_analyzer,
                 feature_engineer, model_trainer, trading_logic):
        super().__init__()
        self.mode = mode  # 'live' or 'backtest'
        self.symbols = []  # Will be updated with trending tickers
        self.start_date = start_date
        self.end_date = end_date
        self.polling_interval = polling_interval
        self.data_loader = data_loader
        self.indicator_calculator = indicator_calculator
        self.sentiment_analyzer = sentiment_analyzer
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.trading_logic = trading_logic
        self.running = True

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.trading_loop())

    async def trading_loop(self):
        if self.mode == 'backtest':
            await self.run_backtest()
        else:
            await self.run_live_trading()

    async def run_live_trading(self):
        # Set a confidence threshold for executing trades
        confidence_threshold = 0.8

        while self.running:
            try:
                # Fetch trending tickers
                trending_tickers = await self.data_loader.get_trending_tickers()
                self.symbols = trending_tickers or []
                if not self.symbols:
                    self.log_signal.emit("No trending tickers found.")
                    await asyncio.sleep(self.polling_interval)
                    continue

                # Safely convert tickers to strings for logging
                symbols_str = [symbol if symbol is not None else 'Unknown' for symbol in self.symbols]
                self.log_signal.emit(f"Trending Tickers: {', '.join(symbols_str)}")

                for symbol in self.symbols:
                    symbol_str = symbol if symbol else "Unknown"

                    if symbol is None:
                        self.log_signal.emit("Symbol is None. Skipping.")
                        continue

                    self.log_signal.emit(f"Processing symbol: {symbol_str}")

                    # Use historical data up to today for training
                    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

                    model_path = f'model_{symbol_str}.joblib'

                    if check_model_exists(model_path):
                        self.model_trainer.load_model(model_path)
                        self.log_signal.emit(f"Loaded existing model for {symbol_str}.")
                    else:
                        self.log_signal.emit(f"Training new model for {symbol_str}.")
                        stock_data = await self.data_loader.get_stock_data([symbol], start_date, end_date)
                        data = stock_data.get(symbol)
                        if data is None or data.empty:
                            self.log_signal.emit(f"No data fetched for {symbol_str}. Skipping.")
                            continue

                        # Train model
                        await self.train_model_for_symbol(symbol, data)
                        if self.model_trainer.model is None:
                            self.log_signal.emit(f"Model training failed for {symbol_str}.")
                            continue
                        self.model_trainer.save_model(self.model_trainer.model, model_path)
                        self.log_signal.emit(f"Model for {symbol_str} trained and saved.")

                    # Fetch real-time data for prediction
                    data = await self.fetch_real_time_data(symbol)
                    if data is None or data.empty:
                        self.log_signal.emit(f"No data fetched for {symbol_str}. Skipping.")
                        continue

                    # Make predictions and execute trades (only if above confidence threshold)
                    await self.make_predictions_and_trade(symbol, data, confidence_threshold=confidence_threshold)

                self.trading_logic.monitor_positions()
                await asyncio.sleep(self.polling_interval)

            except Exception as e:
                # Safely log the error
                error_msg = str(e) if e else "Unknown error"
                self.log_signal.emit(f"An error occurred: {error_msg}")
                logger.error(f"An unexpected error occurred: {error_msg}")

    async def run_backtest(self):
        # Set a confidence threshold for backtesting as well
        confidence_threshold = 0.8

        try:
            trending_tickers = await self.data_loader.get_trending_tickers()
            # Filter out None tickers
            self.symbols = [symbol for symbol in trending_tickers if symbol is not None]
            if not self.symbols:
                self.log_signal.emit("No trending tickers found.")
                return

            symbols_str = [symbol if symbol else 'Unknown' for symbol in self.symbols]
            self.log_signal.emit(f"Backtesting on Tickers: {', '.join(symbols_str)}")

            for symbol in self.symbols:
                symbol_str = symbol if symbol else "Unknown"
                if symbol is None:
                    self.log_signal.emit("Symbol is None. Skipping.")
                    continue

                self.log_signal.emit(f"Backtesting symbol: {symbol_str}")

                stock_data = await self.data_loader.get_stock_data([symbol], self.start_date, self.end_date)
                data = stock_data.get(symbol)
                if data is None or data.empty:
                    self.log_signal.emit(f"No data fetched for {symbol_str}. Skipping.")
                    continue

                await self.train_model_for_symbol(symbol, data)
                if self.model_trainer.model is None:
                    self.log_signal.emit(f"Model training failed for {symbol_str}.")
                    continue

                # In backtesting, simulate trades
                await self.make_predictions_and_trade(symbol, data, is_backtest=True, confidence_threshold=confidence_threshold)

            self.log_signal.emit("Backtesting completed.")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            self.log_signal.emit(f"An error occurred during backtesting: {error_msg}")
            logger.error(f"An unexpected error occurred during backtesting: {error_msg}")

    async def train_model_for_symbol(self, symbol, data):
        if symbol is None:
            self.log_signal.emit("Symbol is None. Skipping model training.")
            return
        if data is None or data.empty:
            self.log_signal.emit(f"No data available for symbol: {symbol}. Skipping model training.")
            return

        # Compute indicators
        indicators = self.indicator_calculator.calculate(data)

        # Fetch TradingView indicators
        tradingview_indicators = self.indicator_calculator.fetch_tradingview_indicators(symbol)

        # Analyze sentiments
        sentiments_df = await self.sentiment_analyzer.analyze_sentiment([symbol])
        sentiments_df.set_index('Ticker', inplace=True)
        sentiments = sentiments_df[['Sentiment']]

        # Validate data before proceeding
        if sentiments is None or indicators is None or tradingview_indicators is None:
            self.log_signal.emit(f"Missing sentiment or indicators for {symbol}. Skipping training.")
            return

        # Create features and labels
        features, labels = self.feature_engineer.create_features(data, indicators, sentiments, tradingview_indicators)
        if features.empty or labels.empty:
            self.log_signal.emit(f"Insufficient data to train model for {symbol}. Skipping model training.")
            return

        # Train model
        self.model_trainer.train(features, labels)

    async def fetch_real_time_data(self, symbol):
        if symbol is None:
            self.log_signal.emit("Symbol is None. Cannot fetch real-time data.")
            return pd.DataFrame()
        # Fetch the latest data point
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = end_date  # Only fetch data for today
        stock_data = await self.data_loader.get_stock_data([symbol], start_date, end_date)
        data = stock_data.get(symbol)
        if data is None or data.empty:
            self.log_signal.emit(f"No real-time data available for symbol: {symbol}.")
        return data or pd.DataFrame()

    async def make_predictions_and_trade(self, symbol, data, is_backtest=False, confidence_threshold=0.8):
        symbol_str = symbol if symbol else "Unknown"

        if symbol is None:
            self.log_signal.emit("Symbol is None.")
            return
        if data is None or data.empty:
            self.log_signal.emit(f"No data available for symbol: {symbol_str}")
            return

        # Compute real-time indicators
        indicators = self.indicator_calculator.calculate(data)

        # Fetch TradingView indicators
        tradingview_indicators = self.indicator_calculator.fetch_tradingview_indicators(symbol)

        # Analyze real-time sentiments
        sentiments_df = await self.sentiment_analyzer.analyze_sentiment([symbol])
        sentiments_df.set_index('Ticker', inplace=True)
        sentiments = sentiments_df[['Sentiment']]

        # Validate data before making predictions
        if sentiments is None or indicators is None or tradingview_indicators is None:
            self.log_signal.emit(f"Missing sentiment or indicators for {symbol_str}. Skipping.")
            return

        # Create real-time features
        features, _ = self.feature_engineer.create_features(data, indicators, sentiments, tradingview_indicators)
        if features.empty:
            self.log_signal.emit(f"Insufficient data to make predictions for {symbol_str}.")
            return

        # Scale features and make predictions
        features_scaled = self.model_trainer.scaler.transform(features.tail(1))
        predictions = self.model_trainer.model.predict(features_scaled)
        confidence_scores = self.model_trainer.model.predict_proba(features_scaled)[:, 1]

        # Check if confidence meets the threshold
        current_confidence = confidence_scores[0]
        if current_confidence < confidence_threshold:
            self.log_signal.emit(f"Confidence {current_confidence:.2f} too low for {symbol_str}, skipping trade.")
            return

        # Trading signal
        model_predictions = {
            'symbol': symbol_str,
            'prediction': 'buy' if predictions[0] == 1 else 'sell',
            'confidence': current_confidence,
            'expiration_date': (datetime.datetime.now() + datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
            'strike_price': float(data['Close'].iloc[-1]) if not data.empty else None,
            'option_type': 'call' if predictions[0] == 1 else 'put'
        }

        order_details = self.trading_logic.determine_trade_entry(model_predictions)
        if order_details:
            if is_backtest:
                self.log_signal.emit(f"Simulated trade for {symbol_str}: {order_details}")
            else:
                self.trading_logic.execute_trade(order_details)
                self.log_signal.emit(f"Trade executed for {symbol_str}: {order_details}")
        else:
            self.log_signal.emit(f"No trade executed for {symbol_str} based on current predictions.")

    def stop(self):
        self.running = False


# Main function to run the application
def main():
    # Start the FastAPI server in a separate thread
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    app_qt = QApplication([])
    trading_bot_app = TradingBotApp()
    trading_bot_app.show()
    app_qt.exec()

if __name__ == "__main__":
    main()
