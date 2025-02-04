from config import FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET
from pymongo import MongoClient
import time
from threading import Thread
from helper_files.client_helper import place_order, get_ndaq_tickers, market_status, strategies, get_latest_price
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.enums import OrderSide
from strategies.trading_strategies_v2 import get_historical_data
from statistics import median
import logging
import heapq
from urllib.parse import quote_plus
from ranking_client import market_status

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('system.log'),  # Log messages to a file
        logging.StreamHandler()             # Log messages to the console
    ]
)

market_status = market_status(polygon_client)

escaped_username = quote_plus(MONGO_DB_USER)
escaped_password = quote_plus(MONGO_DB_PASS)

# MongoDB connection string
mongo_url = f"mongodb+srv://{escaped_username}:{escaped_password}@cluster0.0qoxq.mongodb.net"
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Adjustable parameters
TRADE_INTERVAL = 60  # in seconds
MIN_SPENDING_BALANCE = 1000  # Minimum spending balance
LIQUIDITY_RATIO = 0.30  # Maintain 30% liquidity

# Establish MongoDB client
mongo_client = MongoClient(mongo_url)
db = mongo_client.trading_simulator


def weighted_majority_decision_and_median_quantity(decisions_and_quantities):
    """Determines the majority decision (buy, sell, or hold) and returns the weighted median quantity for the chosen action."""
    buy_decisions = ['buy', 'strong buy']
    sell_decisions = ['sell', 'strong sell']

    weighted_buy_quantities = []
    weighted_sell_quantities = []
    buy_weight = sell_weight = hold_weight = 0

    # Process decisions with weights
    for decision, quantity, weight in decisions_and_quantities:
        if decision in buy_decisions:
            weighted_buy_quantities.extend([quantity])
            buy_weight += weight
        elif decision in sell_decisions:
            weighted_sell_quantities.extend([quantity])
            sell_weight += weight
        elif decision == 'hold':
            hold_weight += weight

    # Determine the majority decision based on the highest accumulated weight
    if buy_weight > sell_weight and buy_weight > hold_weight:
        return 'buy', median(weighted_buy_quantities) if weighted_buy_quantities else 0
    elif sell_weight > buy_weight and sell_weight > hold_weight:
        return 'sell', median(weighted_sell_quantities) if weighted_sell_quantities else 0
    else:
        return 'hold', 0

def initialize_strategy_coefficients():
    """Initialize the coefficient for each strategy based on the ranking."""
    rank_collection = db.rank
    r_t_c_collection = db.rank_to_coefficient
    strategy_to_coefficient = {}

    for strategy in strategies:
        rank = rank_collection.find_one({'strategy': strategy.__name__})['rank']
        coefficient = r_t_c_collection.find_one({'rank': rank})['coefficient']
        strategy_to_coefficient[strategy.__name__] = coefficient

    return strategy_to_coefficient

def execute_trades():
    """Main function to execute trades based on market status."""
    ndaq_tickers = []
    strategy_to_coefficient = initialize_strategy_coefficients()
    buy_heap = []

    while True:
        # Check market status
        status = market_status(polygon_client)
        db.market_data.market_status.update_one({}, {"$set": {"market_status": status}})
        
        if status == "open":
            logging.info("Market is open. Executing trading strategies.")

            # Fetch NASDAQ tickers if not already fetched
            if not ndaq_tickers:
                ndaq_tickers = get_ndaq_tickers(mongo_url, FINANCIAL_PREP_API_KEY)

            # Get account information
            account = trading_client.get_account()
            buying_power = float(account.cash)
            portfolio_value = float(account.portfolio_value)

            # Execute trades for each ticker
            for ticker in ndaq_tickers:
                decisions_and_quantities = []
                
                try:
                    # Get current price and historical data
                    current_price = get_latest_price(ticker)
                    if current_price is None:
                        continue
                    historical_data = get_historical_data(ticker, data_client)

                    # Determine strategy decisions and quantities
                    asset_info = db.assets_quantities.find_one({'symbol': ticker})
                    portfolio_qty = asset_info['quantity'] if asset_info else 0.0

                    for strategy in strategies:
                        decision, quantity, _ = strategy(
                            ticker, current_price, historical_data,
                            buying_power, portfolio_qty, portfolio_value
                        )
                        weight = strategy_to_coefficient[strategy.__name__]
                        decisions_and_quantities.append((decision, quantity, weight))

                    decision, quantity = weighted_majority_decision_and_median_quantity(decisions_and_quantities)
                    logging.info(f"Ticker: {ticker}, Decision: {decision}, Quantity: {quantity}")

                    # Buy logic with heap to prioritize highest weighted buys
                    if decision == "buy" and buying_power > MIN_SPENDING_BALANCE and ((quantity + portfolio_qty) * current_price / portfolio_value) < 0.1:
                        heapq.heappush(buy_heap, (-(strategy_to_coefficient[strategy.__name__]), quantity, ticker))
                    elif decision == "sell" and portfolio_qty > 0:
                        order = place_order(trading_client, ticker, OrderSide.SELL, qty=quantity, mongo_url=mongo_url)
                        logging.info(f"Executed SELL order for {ticker}: {order}")
                    else:
                        logging.info(f"Holding for {ticker}, no action taken.")

                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")

            # Process buys from the heap
            while buy_heap and buying_power > MIN_SPENDING_BALANCE:
                try:
                    _, quantity, ticker = heapq.heappop(buy_heap)
                    order = place_order(trading_client, ticker, OrderSide.BUY, qty=quantity, mongo_url=mongo_url)
                    logging.info(f"Executed BUY order for {ticker}: {order}")
                    # Update buying power after each buy
                    account = trading_client.get_account()
                    buying_power = float(account.cash)

                except Exception as e:
                    logging.error(f"Error occurred while executing buy order for {ticker}: {e}")

            # Wait for the next cycle
            logging.info(f"Sleeping for {TRADE_INTERVAL} seconds...")
            time.sleep(TRADE_INTERVAL)

        elif status == "closed":
            logging.info("Market is closed. Performing post-market operations.")
            ndaq_tickers = []  # Reset tickers for next trading day
            time.sleep(TRADE_INTERVAL)

        else:
            logging.warning("Market is in early hours or after hours. Waiting...")
            time.sleep(TRADE_INTERVAL)

def execute_trade_for_ticker(ticker, strategies):
    try:
        # Get account details from Alpaca
        account = trading_client.get_account()
        account_cash = float(account.cash)
        total_portfolio_value = float(account.cash) + float(account.portfolio_value)

        # Get historical data for the ticker
        historical_data = get_historical_data(ticker, data_client)

        # Retrieve portfolio quantity for the ticker from MongoDB
        with MongoClient(mongo_url) as client:
            db = client.trading_simulator
            asset_info = db.assets_quantities.find_one({'symbol': ticker})
            portfolio_qty = asset_info['quantity'] if asset_info else 0.0

        # Execute strategy for the ticker
        for strategy in strategies:
            action, quantity, ticker = strategy(ticker, current_price=historical_data['close'].iloc[-1],
                                                historical_data=historical_data,
                                                account_cash=account_cash,
                                                portfolio_qty=portfolio_qty,
                                                total_portfolio_value=total_portfolio_value)
            logging.info(f"Trade decision for {ticker}: Action: {action}, Quantity: {quantity}")

            # Place the trade order if a buy or sell action is recommended
            if action == 'buy' and quantity > 0:
                order = place_order(trading_client, ticker, 'buy', quantity)
                logging.info(f"Placed BUY order for {ticker}: {order}")

            elif action == 'sell' and quantity > 0:
                order = place_order(trading_client, ticker, 'sell', quantity)
                logging.info(f"Placed SELL order for {ticker}: {order}")

    except Exception as e:
        logging.error(f"Error executing trade for ticker {ticker}: {e}")

def main():
    ndaq_tickers = get_ndaq_tickers(mongo_url, FINANCIAL_PREP_API_KEY)
    threads = []

    # Run trading strategies for each ticker in parallel
    for ticker in ndaq_tickers:
        thread = Thread(target=execute_trade_for_ticker, args=(ticker, strategies))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()