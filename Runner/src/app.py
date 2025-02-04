"""Acts as the main driver of the program.

Decides whether to run backtesting or live trading.
Integrates:
1. Sentiment analysis from `news_scraper.py`.
2. Technical indicators from `indicators.py`.
3. RL agent actions from `rl_agent.py`.
4. Regression model predictions from `regression_model.py`.
5. Trade execution logic from `trading_logic.py`.
Depending on command-line arguments or config settings, 
decides whether to run backtesting or live trading.

Calls functions from news_scraper.py to get trending tickers, 
indicators.py for technical calculations, and uses trading_logic.py to execute trades.

Loads pre-trained regression models from data/models/ and, if using RL, 
loads the RL agent policy from the same directory.

Backtest: app.py is run with a --backtest flag. 
It loads historical data and simulates trades using 
both regression predictions and RL actions, logging results to data/logs.

Live Trading: app.py run in live mode fetches real-time data, 
uses news_scraper.py for dynamic tickers, 
gets predictions from regression, 
decisions from RL agent, and executes trades in real-time.

Run for:
Backtesting- python app.py --mode backtest --backtest-days 180
Paper Trading- python app.py --mode paper
Live Trading- python app.py --mode live"""


import os
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Local imports
from data.news_scraper import NewsScraper
from indicators.tech_indicators import analyze_indicators
from models.regression_model import RegressionModelHandler
from trading.trading_logic import TradingSystem, OptionStrategy
from models.rl_agent import MarketSimulator, EnhancedDQNAgent

# Setup logging
LOG_DIR = Path('data/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingManager:
    """Manages the trading system."""

    def __init__(self, config_path='config.yaml'):
        load_dotenv()
        self.config_path = config_path
        self.sentiment_analyzer = NewsScraper()
        self.indicators = analyze_indicators
        self.regression_model = RegressionModelHandler()
        self.trading_system = None

    def initialize_trading_system(self, mode='paper'):
        """Initialize trading system based on mode."""
        try:
            self.trading_system = TradingSystem(mode=mode)
            logger.info(f"Trading system initialized in {mode} mode.")
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            raise

    def fetch_analysis(self, symbols):
        """Fetch sentiment, indicators, and model predictions for symbols."""
        analysis_results = {}
        for symbol in symbols:
            try:
                # Sentiment Analysis
                articles = self.sentiment_analyzer.fetch_stock_news(symbol)
                sentiment = self.sentiment_analyzer.analyze_sentiment(articles)

                # Indicators
                indicators = self.indicators(symbol)

                # Regression Prediction
                prediction = self.regression_model.predict(symbol)

                analysis_results[symbol] = {
                    'sentiment': sentiment,
                    'indicators': indicators,
                    'prediction': prediction
                }

                logger.info(f"Analysis for {symbol}: {analysis_results[symbol]}")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        return analysis_results

    def execute_option_strategy(self, symbol, strategy, signals):
        """Execute a specific option strategy for the given symbol."""
        try:
            logger.info(f"Executing {strategy} for {symbol}")
            if strategy == OptionStrategy.LONG_CALL:
                self.trading_system.place_option_trade(symbol, 'CALL', signals)
            elif strategy == OptionStrategy.LONG_PUT:
                self.trading_system.place_option_trade(symbol, 'PUT', signals)
            elif strategy == OptionStrategy.BULL_CALL_SPREAD:
                # Implement logic for bull call spread
                self.trading_system.execute_option_strategy(symbol, OptionStrategy.BULL_CALL_SPREAD, signals)
            elif strategy == OptionStrategy.BEAR_PUT_SPREAD:
                # Implement logic for bear put spread
                self.trading_system.execute_option_strategy(symbol, OptionStrategy.BEAR_PUT_SPREAD, signals)
            elif strategy == OptionStrategy.IRON_CONDOR:
                # Implement logic for iron condor
                self.trading_system.execute_option_strategy(symbol, OptionStrategy.IRON_CONDOR, signals)
            elif strategy == OptionStrategy.BUTTERFLY:
                # Implement logic for butterfly
                self.trading_system.execute_option_strategy(symbol, OptionStrategy.BUTTERFLY, signals)
            else:
                logger.warning(f"Unsupported strategy: {strategy}")
        except Exception as e:
            logger.error(f"Error executing {strategy} for {symbol}: {e}")

    def run_live_trading(self):
        """Execute live trading."""
        try:
            self.initialize_trading_system(mode='live')

            # Fetch trending stocks
            yahoo_tickers = self.sentiment_analyzer.fetch_yahoo_trending()
            swaggy_tickers = self.sentiment_analyzer.fetch_swaggy_sentiment()
            symbols = yahoo_tickers.union(swaggy_tickers)

            # Perform analysis
            analysis = self.fetch_analysis(symbols)

            # Execute trades
            for symbol, data in analysis.items():
                try:
                    signals = {
                        'regression_prediction': data['prediction']['prediction'],
                        'regression_confidence': data['prediction']['confidence'],
                        'rl_action': 'CALL' if data['sentiment']['trading_signal'] == 'CALL' else 'PUT',
                        'rl_confidence': data['sentiment']['sentiment_score'],
                    }

                    # Select strategy based on signals
                    strategy = OptionStrategy.LONG_CALL if signals['regression_prediction'] > 0 else OptionStrategy.LONG_PUT

                    self.execute_option_strategy(symbol, strategy, signals)

                except Exception as e:
                    logger.error(f"Error trading {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error in live trading: {e}")
    
    def run_paper_trading(self):
        """Execute paper trading."""
        try:
            self.initialize_trading_system(mode='paper')

            # Fetch trending stocks (for example)
            yahoo_tickers = asyncio.run(self.sentiment_analyzer.fetch_yahoo_trending())
            swaggy_tickers = asyncio.run(self.sentiment_analyzer.fetch_swaggy_sentiment())
            symbols = yahoo_tickers.union(swaggy_tickers)

            # Perform analysis
            analysis = self.fetch_analysis(symbols)

            # Execute trades
            for symbol, data in analysis.items():
                try:
                    signals = {
                        'regression_prediction': data['prediction']['prediction'],
                        'regression_confidence': data['prediction']['confidence'],
                        'rl_action': 'CALL' if data['sentiment']['trading_signal'] == 'CALL' else 'PUT',
                        'rl_confidence': data['sentiment']['sentiment_score'],
                    }

                    # Select strategy based on signals
                    strategy = OptionStrategy.LONG_CALL if signals['regression_prediction'] > 0 else OptionStrategy.LONG_PUT

                    self.execute_option_strategy(symbol, strategy, signals)

                except Exception as e:
                    logger.error(f"Error trading {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error in paper trading: {e}")

    def run_backtest(self, start_date, end_date):
        """Run backtesting on historical data."""
        try:
            self.initialize_trading_system(mode='backtest')

            # Fetch historical symbols (e.g., from config)
            symbols = self.sentiment_analyzer.fetch_yahoo_trending()

            for symbol in symbols:
                try:
                    # Simulate trades
                    simulator = MarketSimulator(symbol, start_date, end_date)
                    rl_agent = EnhancedDQNAgent(state_size=12, action_size=3)

                    for step in simulator.data.iterrows():
                        obs = simulator._get_observation()
                        action = rl_agent.act(obs, evaluate=True)
                        simulator.step(action)

                    logger.info(f"Backtesting complete for {symbol}")

                except Exception as e:
                    logger.error(f"Error in backtesting {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error in backtest setup: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading System Controller')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], required=True, help='Trading mode')
    parser.add_argument('--start_date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Backtest end date (YYYY-MM-DD)')
    args = parser.parse_args()

    manager = TradingManager()

    if args.mode == 'live':
        manager.run_live_trading()
    elif args.mode == 'paper':
        manager.run_paper_trading()
    elif args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            logger.error("Start and end dates are required for backtesting.")
            return

        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        manager.run_backtest(start_date, end_date)
    else:
        logger.error("Invalid mode.")


if __name__ == "__main__":
    main()
