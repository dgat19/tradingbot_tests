import os
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# -----------------------
# CONFIGURATION VARIABLES
# -----------------------
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
# For paper trading use:
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

# Trading instruments
TECH_STOCK_SYMBOL = "NVDA"      # AI/tech company proxy
BENCHMARK_SYMBOL = "QQQ"        # Tech-heavy ETF

# Trading parameters
POSITION_SIZE = 100             # Number of shares per leg
MAX_LOSS_PERCENT = 0.20         # 20% stop-loss threshold
DIVERGENCE_THRESHOLD = 0.05     # 5% divergence threshold

# Historical multiplier to estimate NVDA's fair value from QQQ.
HISTORICAL_MULTIPLIER = 8.0

# -----------------------
# INITIALIZE ALPACA CLIENTS
# -----------------------
# Trading client for order management.
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Data client for market data retrieval.
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ------------------------------------
# HELPER FUNCTIONS FOR DATA RETRIEVAL
# ------------------------------------
def get_latest_price(symbol: str) -> float:
    """
    Retrieve the latest trade price (ask/bid mid-quote) for the given symbol.
    Uses the latest quote from Alpaca's data API.
    """
    try:
        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        latest_quote = data_client.get_stock_latest_quote(request_params)
        # Calculate the mid price from the latest bid and ask
        bid = latest_quote[symbol].bid_price
        ask = latest_quote[symbol].ask_price
        if bid is None or ask is None:
            raise ValueError("Bid or Ask price not available.")
        mid_price = (bid + ask) / 2
        return mid_price
    except Exception as e:
        print(f"Error retrieving latest price for {symbol}: {e}")
        return None

# ------------------------------------
# TRADE EXECUTION FUNCTIONS
# ------------------------------------
def place_order(symbol: str, qty: int, side: OrderSide, stop_loss: float = None):
    """
    Place a market order via Alpaca's TradingClient.
    In practice, you may want to use bracket orders to embed stop-losses.
    Here we submit a simple market order and log the intended stop-loss.
    """
    try:
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        print(f"Placed {side.value} order for {qty} shares of {symbol}.")
        if stop_loss:
            print(f"Intended stop-loss for {symbol} at: {stop_loss:.2f}")
        return order
    except Exception as e:
        print(f"Error placing order for {symbol}: {e}")
        return None

def execute_trade_pair(stock_side: OrderSide, benchmark_side: OrderSide, stock_price: float):
    """
    Execute a paired trade between the tech stock and the benchmark ETF.
    The stock_side parameter indicates the side for the tech stock (NVDA)
    and benchmark_side is the opposite side for QQQ.
    """
    # Calculate stop-loss prices for risk management
    if stock_side == OrderSide.BUY:
        stock_stop_loss = stock_price * (1 - MAX_LOSS_PERCENT)
    else:
        stock_stop_loss = stock_price * (1 + MAX_LOSS_PERCENT)
    
    benchmark_price = get_latest_price(BENCHMARK_SYMBOL)
    if benchmark_price is None:
        print("Could not retrieve benchmark price. Aborting trade.")
        return

    if benchmark_side == OrderSide.BUY:
        benchmark_stop_loss = benchmark_price * (1 - MAX_LOSS_PERCENT)
    else:
        benchmark_stop_loss = benchmark_price * (1 + MAX_LOSS_PERCENT)
    
    # Submit orders using alpaca-py's TradingClient
    stock_order = place_order(TECH_STOCK_SYMBOL, POSITION_SIZE, stock_side, stop_loss=stock_stop_loss)
    benchmark_order = place_order(BENCHMARK_SYMBOL, POSITION_SIZE, benchmark_side, stop_loss=benchmark_stop_loss)
    
    print("Paired trade orders submitted.")
    return stock_order, benchmark_order

# ------------------------------------
# CORE TRADING LOGIC
# ------------------------------------
def evaluate_trade():
    """
    Evaluate the divergence between the tech stock (NVDA) and its fair value as implied by QQQ.
    Fair value is estimated as: fair_NVDA = HISTORICAL_MULTIPLIER * QQQ_price.
    A significant divergence triggers a paired trade.
    """
    tech_stock_price = get_latest_price(TECH_STOCK_SYMBOL)
    benchmark_price = get_latest_price(BENCHMARK_SYMBOL)

    if tech_stock_price is None or benchmark_price is None:
        print("Error retrieving market data. Aborting trade evaluation.")
        return

    # Calculate fair value and divergence
    fair_value = HISTORICAL_MULTIPLIER * benchmark_price
    divergence = (tech_stock_price - fair_value) / fair_value

    print(f"{TECH_STOCK_SYMBOL} Price: {tech_stock_price:.2f}, {BENCHMARK_SYMBOL} Price: {benchmark_price:.2f}")
    print(f"Computed Fair Value for {TECH_STOCK_SYMBOL}: {fair_value:.2f}")
    print(f"Divergence: {divergence:.2%}")

    # Decision logic for trade execution
    if divergence > DIVERGENCE_THRESHOLD:
        # NVDA appears overpriced relative to the fair value implied by QQQ.
        print(f"{TECH_STOCK_SYMBOL} is overpriced. Initiating short {TECH_STOCK_SYMBOL} and long {BENCHMARK_SYMBOL}.")
        execute_trade_pair(OrderSide.SELL, OrderSide.BUY, tech_stock_price)
    elif divergence < -DIVERGENCE_THRESHOLD:
        # NVDA appears undervalued.
        print(f"{TECH_STOCK_SYMBOL} is undervalued. Initiating long {TECH_STOCK_SYMBOL} and short {BENCHMARK_SYMBOL}.")
        execute_trade_pair(OrderSide.BUY, OrderSide.SELL, tech_stock_price)
    else:
        print("No significant divergence detected. No trade executed.")

# ------------------------------------
# MAIN LOOP OR SCHEDULER
# ------------------------------------
if __name__ == "__main__":
    try:
        while True:
            print("Evaluating market inefficiencies for AI/Tech companies...")
            evaluate_trade()
            print("Sleeping for 60 seconds...\n")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Trading bot terminated by user.")
