from config import MONGO_DB_USER, MONGO_DB_PASS
from pymongo import MongoClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime
import logging
import yfinance as yf
from urllib.request import urlopen
import json
from contextlib import contextmanager
from urllib.parse import quote_plus
from strategies.trading_strategies_v2 import (  
   rsi_strategy, bollinger_bands_strategy, momentum_strategy, mean_reversion_strategy,  
   triple_moving_average_strategy, volume_price_trend_strategy, keltner_channel_strategy,  
   dual_thrust_strategy, adaptive_momentum_strategy, hull_moving_average_strategy,  
   ultimate_oscillator_strategy, stochastic_momentum_strategy, volume_weighted_macd_strategy, 
   volatility_breakout_strategy, momentum_divergence_strategy, wavelet_decomposition_strategy  
)
from strategies.trading_strategies_v3 import (
    pairs_trading_strategy, kalman_filter_strategy, regime_switching_strategy, adaptive_momentum_filter_strategy,
    topological_data_analysis_strategy, levy_distribution_strategy,
    wavelet_momentum_strategy, complex_network_strategy, quantum_oscillator_strategy, simple_trend_reversal_strategy
)


beta_strategies = [pairs_trading_strategy, kalman_filter_strategy, regime_switching_strategy, adaptive_momentum_filter_strategy,
                   topological_data_analysis_strategy, levy_distribution_strategy,
                   wavelet_momentum_strategy, complex_network_strategy, quantum_oscillator_strategy, simple_trend_reversal_strategy]

strategies = [rsi_strategy, bollinger_bands_strategy, momentum_strategy, mean_reversion_strategy,  
   triple_moving_average_strategy, volume_price_trend_strategy, keltner_channel_strategy,  
   dual_thrust_strategy, adaptive_momentum_strategy, hull_moving_average_strategy,  
   ultimate_oscillator_strategy, stochastic_momentum_strategy, volume_weighted_macd_strategy, 
   volatility_breakout_strategy, momentum_divergence_strategy,  
   wavelet_decomposition_strategy] + beta_strategies

escaped_username = quote_plus(MONGO_DB_USER)
escaped_password = quote_plus(MONGO_DB_PASS)

# MongoDB connection string
mongo_url = f"mongodb+srv://{escaped_username}:{escaped_password}@cluster0.0qoxq.mongodb.net"

# MongoDB connection helper
@contextmanager
def mongo_client():
    client = MongoClient(mongo_url)
    try:
        yield client
    finally:
        client.close()

# Helper to place an order
def place_order(trading_client, symbol, side, qty, mongo_url):
    """
    Place a market order and log the order to MongoDB.

    :param trading_client: The Alpaca trading client instance
    :param symbol: The stock symbol to trade
    :param side: Order side (OrderSide.BUY or OrderSide.SELL)
    :param qty: Quantity to trade
    :param mongo_url: MongoDB connection URL
    :return: Order result from Alpaca API
    """
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    order = trading_client.submit_order(market_order_data)
    qty = round(qty, 3)

    # Log trade details to MongoDB
    with mongo_client() as client:
        db = client.trades
        db.paper.insert_one({
            'symbol': symbol,
            'qty': qty,
            'side': side.name,
            'time_in_force': TimeInForce.DAY.name,
            'time': datetime.now()
        })
        assets = db.assets_quantities
        if side == OrderSide.BUY:
            assets.update_one({'symbol': symbol}, {'$inc': {'quantity': qty}}, upsert=True)
        elif side == OrderSide.SELL:
            assets.update_one({'symbol': symbol}, {'$inc': {'quantity': -qty}}, upsert=True)
            if assets.find_one({'symbol': symbol})['quantity'] == 0:
                assets.delete_one({'symbol': symbol})
    return order

# Helper to retrieve NASDAQ-100 tickers from MongoDB
def get_ndaq_tickers(mongo_url, FINANCIAL_PREP_API_KEY):
    """
    Connects to MongoDB and retrieves NASDAQ-100 tickers.

    :param mongo_url: MongoDB connection URL
    :return: List of NASDAQ-100 ticker symbols.
    """
    def call_ndaq_100():
        """
        Fetches the list of NASDAQ 100 tickers using the Financial Modeling Prep API and stores it in a MongoDB collection.
        The MongoDB collection is cleared before inserting the updated list of tickers.
        """
        logging.info("Calling NASDAQ 100 to retrieve tickers.")

        def get_jsonparsed_data(url):
            """
            Parses the JSON response from the provided URL.
            
            :param url: The API endpoint to retrieve data from.
            :return: Parsed JSON data as a dictionary.
            """
            response = urlopen(url)
            data = response.read().decode("utf-8")
            return json.loads(data)

        try:
            # API URL for fetching NASDAQ 100 tickers
            ndaq_url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={FINANCIAL_PREP_API_KEY}"
            ndaq_stocks = get_jsonparsed_data(ndaq_url)
            logging.info("Successfully retrieved NASDAQ 100 tickers.")
        except Exception as e:
            logging.error(f"Error fetching NASDAQ 100 tickers: {e}")
            return

        try:
            # MongoDB connection details
            mongo_client = MongoClient(mongo_url)
            db = mongo_client.stock_list
            ndaq100_tickers = db.ndaq100_tickers

            ndaq100_tickers.delete_many({})  # Clear existing data
            ndaq100_tickers.insert_many(ndaq_stocks)  # Insert new data
            logging.info("Successfully inserted NASDAQ 100 tickers into MongoDB.")
        except Exception as e:
            logging.error(f"Error inserting tickers into MongoDB: {e}")
        finally:
            mongo_client.close()
            logging.info("MongoDB connection closed.")
    call_ndaq_100()
    mongo_client = MongoClient(mongo_url) 
    tickers = [stock['symbol'] for stock in mongo_client.stock_list.ndaq100_tickers.find()]
    mongo_client.close()
    return tickers

# Market status checker helper
def market_status(polygon_client):
    """
    Check market status using the Polygon API.

    :param polygon_client: An instance of the Polygon RESTClient
    :return: Current market status ('open', 'early_hours', 'closed')
    """
    try:
        status = polygon_client.get_market_status()
        if status.exchanges.nasdaq == "open" and status.exchanges.nyse == "open":
            return "open"
        elif status.early_hours:
            return "early_hours"
        else:
            return "closed"
    except Exception as e:
        logging.error(f"Error retrieving market status: {e}")
        return "error"

# Helper to get latest price
def get_latest_price(ticker):  
   """  
   Fetch the latest price for a given stock ticker using yfinance.  
  
   :param ticker: The stock ticker symbol  
   :return: The latest price of the stock  
   """  
   try:  
      ticker_yahoo = yf.Ticker(ticker)  
      data = ticker_yahoo.history()  
      return round(data['Close'].iloc[-1], 2)  
   except Exception as e:  
      logging.error(f"Error fetching latest price for {ticker}: {e}")  
      return None