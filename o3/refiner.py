import os
import time
import alpaca_trade_api as tradeapi
import requests

# -----------------------
# CONFIGURATION VARIABLES
# -----------------------
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # or live endpoint as appropriate

# Define trading parameters
REFINER_STOCK_SYMBOL = "XYZ"  # Replace with your chosen refiner stock symbol
POSITION_SIZE = 100  # Number of shares to trade
MAX_LOSS_PERCENT = 0.20

# For futures (crack spread), we assume a broker API endpoint
SCHWAB_API_ENDPOINT = "https://api.yourbroker.com/v1"  # placeholder URL
SCHWAB_API_KEY = os.getenv('SCHWAB_API_KEY')  # Placeholder

# Historical average divergence threshold (set based on backtest)
DIVERGENCE_THRESHOLD = 0.05  # Example: 5% divergence

# ------------------------------------
# INITIALIZE ALPACA API FOR EQUITY TRADING
# ------------------------------------
alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# ------------------------------------
# HELPER FUNCTIONS FOR DATA RETRIEVAL
# ------------------------------------
def get_refiner_stock_price(symbol):
    """
    Retrieve latest trade price for the refiner stock using Alpaca market data.
    """
    try:
        barset = alpaca.get_barset(symbol, 'minute', limit=1)
        price = barset[symbol][0].c
        return price
    except Exception as e:
        print(f"Error retrieving price for {symbol}: {e}")
        return None

def get_crack_spread_price():
    """
    Retrieve a proxy price for the 3:2:1 crack spread. In practice this might be computed
    from futures prices of crude oil, gasoline, and distillate.
    This function is a placeholder.
    """
    # For demonstration, we use dummy data.
    # In practice, you’d call your futures data provider or broker API.
    # Example: crude, gasoline, distillate = get_futures_prices(...)
    # crack_spread = (2 * gasoline + 1 * distillate) - 3 * crude
    try:
        # Dummy data simulation:
        crude = 70.0  # USD per barrel
        gasoline = 75.0  # USD per gallon equivalent, scaled as needed
        distillate = 72.0
        crack_spread = (2 * gasoline + distillate) - (3 * crude)
        return crack_spread
    except Exception as e:
        print(f"Error retrieving crack spread: {e}")
        return None

def get_schwab_futures_position():
    """
    Retrieve current position info for the crack spread futures from the Schwab API.
    This is a placeholder function.
    """
    # Example request (you'd need to adapt based on your broker's API):
    headers = {"Authorization": f"Bearer {SCHWAB_API_KEY}"}
    try:
        response = requests.get(f"{SCHWAB_API_ENDPOINT}/futures/crack_spread/position", headers=headers)
        if response.status_code == 200:
            return response.json()  # Expected to include position size, entry price, etc.
        else:
            print("Error retrieving futures position:", response.text)
            return None
    except Exception as e:
        print("Exception retrieving futures position:", e)
        return None

# ------------------------------------
# TRADE EXECUTION FUNCTIONS
# ------------------------------------
def place_alpaca_order(symbol, qty, side, order_type="market", time_in_force="day", stop_loss=None):
    """
    Place an order via Alpaca API.
    Optionally attach a stop-loss order.
    """
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        print(f"Placed {side} order for {qty} shares of {symbol}.")
        if stop_loss:
            # For stop-loss, you could use Alpaca’s bracket orders or submit a separate stop order.
            # Here we simply print the intended stop-loss for illustration.
            print(f"Stop-loss set at: {stop_loss}")
        return order
    except Exception as e:
        print(f"Error placing order for {symbol}: {e}")
        return None

def place_schwab_futures_order(direction, quantity, stop_loss=None):
    """
    Place a futures order for the crack spread via the Schwab API (placeholder).
    'direction' can be 'buy' or 'sell'.
    """
    headers = {"Authorization": f"Bearer {SCHWAB_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "instrument": "crack_spread",
        "direction": direction,
        "quantity": quantity,
        "order_type": "market"
    }
    if stop_loss:
        payload["stop_loss"] = stop_loss
    try:
        response = requests.post(f"{SCHWAB_API_ENDPOINT}/futures/orders", json=payload, headers=headers)
        if response.status_code == 200:
            print(f"Placed {direction} futures order for crack spread, quantity: {quantity}.")
            return response.json()
        else:
            print("Error placing futures order:", response.text)
            return None
    except Exception as e:
        print("Exception placing futures order:", e)
        return None

# ------------------------------------
# CORE TRADING LOGIC
# ------------------------------------
def evaluate_trade():
    """
    Evaluate the divergence between the crack spread and the refiner stock.
    If divergence exceeds the threshold, execute the paired trade.
    """
    refiner_price = get_refiner_stock_price(REFINER_STOCK_SYMBOL)
    crack_spread = get_crack_spread_price()

    if refiner_price is None or crack_spread is None:
        print("Error retrieving market data. Aborting trade evaluation.")
        return

    # For the purpose of this example, assume a simplistic model:
    # The "fair value" of the refiner stock might be modeled as a function of the crack spread.
    # For instance, assume: fair_stock_price = k * crack_spread, with k determined historically.
    # Here we use an arbitrary constant for demonstration.
    k = 2.0
    fair_stock_price = k * crack_spread

    divergence = (refiner_price - fair_stock_price) / fair_stock_price
    print(f"Refiner Stock Price: {refiner_price}, Crack Spread: {crack_spread}")
    print(f"Computed fair value: {fair_stock_price} and divergence: {divergence:.2%}")

    # Check divergence criteria (example: if divergence exceeds +/- DIVERGENCE_THRESHOLD)
    if divergence > DIVERGENCE_THRESHOLD:
        # The stock is expensive relative to the crack spread margin
        print("Divergence positive: stock overvalued relative to crack spread. Initiate short stock / long futures trade.")
        execute_trade_pair(stock_side="sell", futures_side="buy", refiner_price=refiner_price)
    elif divergence < -DIVERGENCE_THRESHOLD:
        # The stock is cheap relative to the crack spread margin
        print("Divergence negative: stock undervalued relative to crack spread. Initiate long stock / short futures trade.")
        execute_trade_pair(stock_side="buy", futures_side="sell", refiner_price=refiner_price)
    else:
        print("No significant divergence. No trade executed.")

def execute_trade_pair(stock_side, futures_side, refiner_price):
    """
    Execute a paired trade with position sizing and stop-loss orders.
    """
    # Calculate stop-loss price for the stock position (20% adverse move)
    if stock_side == "buy":
        stock_stop_loss = refiner_price * (1 - MAX_LOSS_PERCENT)
    else:
        stock_stop_loss = refiner_price * (1 + MAX_LOSS_PERCENT)
    
    # Place order on refiner stock via Alpaca
    stock_order = place_alpaca_order(
        symbol=REFINER_STOCK_SYMBOL,
        qty=POSITION_SIZE,
        side=stock_side,
        order_type="market",
        time_in_force="day",
        stop_loss=stock_stop_loss  # for illustration; actual implementation may require bracket orders
    )

    # For futures, we define an equivalent position quantity.
    # This could be based on a risk-adjusted sizing formula. Here we use a dummy quantity.
    futures_quantity = 1  # placeholder: in practice, calculate contract size and exposure
    # Assume similar stop-loss logic applies to futures (value determined by crack spread sensitivity)
    futures_stop_loss = None  # Implement as needed

    # Place order on futures via Schwab (or your futures broker)
    futures_order = place_schwab_futures_order(
        direction=futures_side,
        quantity=futures_quantity,
        stop_loss=futures_stop_loss
    )

    # Log orders (in production, add robust error checking and monitoring)
    print("Paired trade orders submitted.")
    return stock_order, futures_order

# ------------------------------------
# MAIN LOOP OR SCHEDULER
# ------------------------------------
if __name__ == "__main__":
    # This loop runs continuously, checking every minute.
    # In a real trading system, you might use event-driven architecture instead.
    try:
        while True:
            print("Evaluating trade conditions...")
            evaluate_trade()
            print("Sleeping for 60 seconds...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Trading bot terminated by user.")
