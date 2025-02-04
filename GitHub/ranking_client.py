from config import FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET, POLYGON_API_KEY
from pymongo import MongoClient
import time
from datetime import datetime
import strategies.trading_strategies_v3 as trading_strategies
import math
import logging
from collections import Counter
from helper_files.client_helper import strategies, get_latest_price, get_ndaq_tickers
import heapq
from alpaca.data.historical import StockHistoricalDataClient
from urllib.parse import quote_plus
from polygon import RESTClient
from helper_files.client_helper import market_status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('rank_system.log'),  # Log messages to a file
        logging.StreamHandler()             # Log messages to the console
    ]
)

# Initialize Polygon client
polygon_client = RESTClient(POLYGON_API_KEY)

escaped_username = quote_plus(MONGO_DB_USER)
escaped_password = quote_plus(MONGO_DB_PASS)

# MongoDB connection string
mongo_url = f"mongodb+srv://{escaped_username}:{escaped_password}@cluster0.0qoxq.mongodb.net"

def with_mongo_client(func):
    """Decorator to manage MongoDB connection."""
    def wrapper(*args, **kwargs):
        client = MongoClient(mongo_url)
        try:
            return func(client, *args, **kwargs)
        finally:
            client.close()
    return wrapper

def find_nans_within_rank_holding():
    # Using Counter here
    mongo_client = MongoClient(mongo_url)
    db = mongo_client.trading_simulator
    collections = db.algorithm_holdings
    for strategy in strategies:
        strategy_doc = collections.find_one({"strategy": strategy.__name__})
        holdings_doc = strategy_doc.get("holdings", {})
        for ticker in holdings_doc:
            if holdings_doc[ticker]['quantity'] == 0:
                print(f"{ticker} : {strategy.__name__}")

    # Count zero-quantity tickers to provide some stats (utilizing Counter)
    tickers_count = Counter([ticker for holdings in collections.find() for ticker in holdings.get('holdings', {}) if holdings['holdings'][ticker]['quantity'] == 0])
    print(f"Tickers with zero quantity: {tickers_count}")

def insert_rank_to_coefficient(i):
    client = MongoClient(mongo_url)
    db = client.trading_simulator
    collections = db.rank_to_coefficient
    collections.delete_many({})
    for j in range(1, i + 1):
        e = math.e
        rate = (e**e)/(e**2) - 1
        coefficient = rate**(2 * j)
        collections.insert_one({
            "rank": j,
            "coefficient": coefficient
        })
    client.close()


@with_mongo_client
def initialize_rank():
    client = MongoClient(mongo_url)
    db = client.trading_simulator
    collections = db.algorithm_holdings
    initialization_date = datetime.now()

    for strategy in strategies:
        strategy_name = strategy.__name__
        if not collections.find_one({"strategy": strategy_name}):
            collections.insert_one({
                "strategy": strategy_name,
                "holdings": {},
                "amount_cash": 50000,
                "initialized_date": initialization_date,
                "total_trades": 0,
                "successful_trades": 0,
                "neutral_trades": 0,
                "failed_trades": 0,
                "last_updated": initialization_date,
                "portfolio_value": 50000
            })

            points_collection = db.points_tally
            points_collection.insert_one({
                "strategy": strategy_name,
                "total_points": 0,
                "initialized_date": initialization_date,
                "last_updated": initialization_date
            })
    client.close()

def simulate_trade(ticker, strategy, historical_data, current_price, account_cash, portfolio_qty, total_portfolio_value, mongo_url):
   """
   Simulates a trade based on the given strategy and updates MongoDB.
   """
    
   # Simulate trading action from strategy
   print(f"Simulating trade for {ticker} with strategy {strategy.__name__} and quantity of {portfolio_qty}")
   action, quantity, _ = strategy(ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value)
   
   # MongoDB setup
   client = MongoClient(mongo_url)
   db = client.trading_simulator
   holdings_collection = db.algorithm_holdings
   points_collection = db.points_tally
   
   # Find the strategy document in MongoDB
   strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
   holdings_doc = strategy_doc.get("holdings", {})
   time_delta = db.time_delta.find_one({})['time_delta']
   
   
   # Update holdings and cash based on trade action
   if action in ["buy", "strong buy"] and strategy_doc["amount_cash"] - quantity * current_price > 15000 and quantity > 0:
      logging.info(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}")
      # Calculate average price if already holding some shares of the ticker
      if ticker in holdings_doc:
         current_qty = holdings_doc[ticker]["quantity"]
         new_qty = current_qty + quantity
         average_price = (holdings_doc[ticker]["price"] * current_qty + current_price * quantity) / new_qty
      else:
         new_qty = quantity
         average_price = current_price

      # Update the holdings document for the ticker. 
      holdings_doc[ticker] = {
            "quantity": new_qty,
            "price": average_price
      }

      # Deduct the cash used for buying and increment total trades
      holdings_collection.update_one(
         {"strategy": strategy.__name__},
         {
            "$set": {
                  "holdings": holdings_doc,
                  "amount_cash": strategy_doc["amount_cash"] - quantity * current_price,
                  "last_updated": datetime.now()
            },
            "$inc": {"total_trades": 1}
         },
         upsert=True
      )
      

   elif action in ["sell", "strong sell"] and str(ticker) in holdings_doc and holdings_doc[str(ticker)]["quantity"] > 0:
      
      logging.info(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}")
      current_qty = holdings_doc[ticker]["quantity"]
        
      # Ensure we do not sell more than we have
      sell_qty = min(quantity, current_qty)
      holdings_doc[ticker]["quantity"] = current_qty - sell_qty
      
      price_change_ratio = current_price / holdings_doc[ticker]["price"] if ticker in holdings_doc else 1
      
      

      if current_price > holdings_doc[ticker]["price"]:
         #increment successful trades
         holdings_collection.update_one(
            {"strategy": strategy.__name__},
            {"$inc": {"successful_trades": 1}},
            upsert=True
         )
         
         # Calculate points to add if the current price is higher than the purchase price
         if price_change_ratio < 1.05:
            points = time_delta * 1
         elif price_change_ratio < 1.1:
            points = time_delta * 1.5
         else:
            points = time_delta * 2
         
      else:
         # Calculate points to deduct if the current price is lower than the purchase price
         if holdings_doc[ticker]["price"] == current_price:
            holdings_collection.update_one(
               {"strategy": strategy.__name__},
               {"$inc": {"neutral_trades": 1}}
            )
            
         else:   
            
            holdings_collection.update_one(
               {"strategy": strategy.__name__},
               {"$inc": {"failed_trades": 1}},
               upsert=True
            )
         
         if price_change_ratio > 0.975:
            points = -time_delta * 1
         elif price_change_ratio > 0.95:
            points = -time_delta * 1.5
         else:
            points = -time_delta * 2
         
      # Update the points tally
      points_collection.update_one(
         {"strategy": strategy.__name__},
         {
            "$set" : {
               "last_updated": datetime.now()
            },
            "$inc": {"total_points": points}
         },
         upsert=True
      )
      if holdings_doc[ticker]["quantity"] == 0:      
         del holdings_doc[ticker]
      # Update cash after selling
      holdings_collection.update_one(
         {"strategy": strategy.__name__},
         {
            "$set": {
               "holdings": holdings_doc,
               "amount_cash": strategy_doc["amount_cash"] + sell_qty * current_price,
               "last_updated": datetime.now()
            },
            "$inc": {"total_trades": 1}
         },
         upsert=True
      )

        
      # Remove the ticker if quantity reaches zero
      if holdings_doc[ticker]["quantity"] == 0:      
         del holdings_doc[ticker]
        
   else:
      logging.info(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}")
   print(f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}")
   # Close the MongoDB connection
   client.close()

def update_portfolio_values():
   """
   still need to implement.
   we go through each strategy and update portfolio value buy cash + summation(holding * current price)
   """
   client = MongoClient(mongo_url)  
   db = client.trading_simulator  
   holdings_collection = db.algorithm_holdings
   # Update portfolio values
   for strategy_doc in holdings_collection.find({}):
      # Calculate the portfolio value for the strategy
      portfolio_value = strategy_doc["amount_cash"]
      
      for ticker, holding in strategy_doc["holdings"].items():
          
          # Get the current price of the ticker from the Polygon API
          current_price = None
          while current_price is None:
            try:
               current_price = get_latest_price(ticker)
            except:
               print(f"Error fetching price for {ticker}. Retrying...")
          print(f"Current price of {ticker}: {current_price}")
          # Calculate the value of the holding
          holding_value = holding["quantity"] * current_price
          # Add the holding value to the portfolio value
          portfolio_value += holding_value
          
      # Update the portfolio value in the strategy document
      holdings_collection.update_one({"strategy": strategy_doc["strategy"]}, {"$set": {"portfolio_value": portfolio_value}}, upsert=True)

   # Update MongoDB with the modified strategy documents
   client.close()

@with_mongo_client
def update_ranks():
    client = MongoClient(mongo_url)
    db = client.trading_simulator
    points_collection = db.points_tally
    rank_collection = db.rank
    algo_holdings = db.algorithm_holdings
    rank_collection.delete_many({})
    q = []

    for strategy_doc in algo_holdings.find({}):
        strategy_name = strategy_doc["strategy"]
        if strategy_name == "test" or strategy_name == "test_strategy":
            continue
        points = points_collection.find_one({"strategy": strategy_name})["total_points"] / 10
        portfolio_value = strategy_doc["portfolio_value"] / 50000
        successful_trades = strategy_doc["successful_trades"] - strategy_doc["failed_trades"]
        heapq.heappush(q, (points + portfolio_value, successful_trades, strategy_doc["strategy"]))
    
    rank = 1
    while q:
        _, _, strategy_name = heapq.heappop(q)
        rank_collection.insert_one({"strategy": strategy_name, "rank": rank})
        rank += 1
    client.close()

def main():
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_market_hour_first_iteration = True
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    mongo_client = MongoClient(mongo_url)
    db = mongo_client.trading_simulator
    holdings_collection = db.algorithm_holdings

    while True:
        if market_status == "open":
            if not ndaq_tickers:
                ndaq_tickers = get_ndaq_tickers(mongo_url, FINANCIAL_PREP_API_KEY)
            for strategy in strategies:
                strategy_doc = holdings_collection.find_one({"strategy": strategy.__name__})
                if not strategy_doc:
                    continue
                account_cash = strategy_doc["amount_cash"]
                total_portfolio_value = strategy_doc["portfolio_value"]
                for ticker in ndaq_tickers:
                    current_price = get_latest_price(ticker)
                    if current_price is None:
                        continue
                    historical_data = trading_strategies.get_historical_data(ticker, data_client)
                    portfolio_qty = strategy_doc["holdings"].get(ticker, {}).get("quantity", 0)
                    simulate_trade(ticker, strategy, historical_data, current_price, account_cash, portfolio_qty, total_portfolio_value, mongo_url)
            update_portfolio_values()
            time.sleep(30)  # Update every 30 seconds (can be adjusted)
        elif market_status == "closed":
            if post_market_hour_first_iteration:
                post_market_hour_first_iteration = False
                update_ranks()
            time.sleep(60)
        else:
            time.sleep(60)

if __name__ == "__main__":
    main()