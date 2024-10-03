
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import time
import sys
from news_scraper import get_top_active_movers
from indicators import get_trend_indicator, get_volume_indicator

# Alpaca API keys - ensure to replace with your actual keys or environment variables
ALPACA_API_KEY = "PKLSUU1T53HAXFDKFQMY"
ALPACA_API_SECRET = "M46BGIZBuunwXIgDU1ttxnNj0nURPZfxt1IjLkdr"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)  # Paper trading mode

# Initialize a trade tracking dictionary
placed_trades = {}

# Cooldown period in seconds
TRADE_COOLDOWN = 600  # 10 minutes

# Hardcoded list of stocks that can be traded based on 1% price movement
HARDCODED_STOCKS = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']

def get_options_chain(symbol, expiration=None):
    stock = yf.Ticker(symbol)
    options_dates = stock.options
    if not expiration:
        expiration = options_dates[0]  # Choose the nearest expiration by default

    if expiration in options_dates:
        options_chain = stock.option_chain(expiration)
        return options_chain
    return None

def place_option_trade(contract_symbol, qty=1, option_type='call'):
    # Use Alpaca API to place an order
    try:
        print(f"Placing {option_type.upper()} order for {contract_symbol}, Quantity: {qty}")
        order = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY if option_type == 'call' else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"{option_type.capitalize()} order placed for {contract_symbol}")
        return True
    except Exception as e:
        print(f"Error placing order for {contract_symbol}: {e}")
        return False

def check_and_close_trade(entry_price, contract_symbol, qty=1):
    # Simulate checking and closing the trade.
    print(f"Checking trade for {contract_symbol}, Entry price: {entry_price}")

def check_hardcoded_stocks():
    # Fetch data for hardcoded stocks and check for price movements greater than 1%
    hardcoded_trades = []
    
    for stock_symbol in HARDCODED_STOCKS:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="1d")
        open_price = stock_data["Open"].iloc[-1]
        current_price = stock_data["Close"].iloc[-1]
        
        # Calculate price movement percentage
        price_movement = ((current_price - open_price) / open_price) * 100
        
        if abs(price_movement) > 1:  # If movement is greater than 1%
            hardcoded_trades.append((stock_symbol, price_movement, current_price))
    
    return hardcoded_trades

def automated_trading(stock_symbol, qty=1):
    print(f"Fetching data for {stock_symbol}")
    
    stock = yf.Ticker(stock_symbol)
    stock_price = stock.history(period="1d")["Close"].iloc[-1]
    current_price = stock_price

    # Check if a trade for this stock has been placed within the cooldown period
    current_time = time.time()
    if stock_symbol in placed_trades and (current_time - placed_trades[stock_symbol]) < TRADE_COOLDOWN:
        print(f"{stock_symbol} - Trade already placed recently, skipping for now.")
        return

    # Get trend and volume indicators
    trend = get_trend_indicator(stock_symbol)
    volume_signal = get_volume_indicator(stock_symbol)

    # Print the indicators for debugging and visibility purposes
    print(f"--- Indicators for {stock_symbol} ---")
    print(f"Current Price: {current_price}")
    print(f"Trend: {'Upward' if trend > 0 else 'Downward' if trend < 0 else 'Neutral'}")
    print(f"Volume Signal: {'Above Average' if volume_signal else 'Normal'}")

    if trend > 0 and volume_signal:  # Upward trend, buy calls
        print(f"{stock_symbol} - Positive trend identified, buying calls")
        options_chain = get_options_chain(stock_symbol)
        if options_chain:
            target_strike = current_price * 1.10  # 10% above current price
            suitable_options = options_chain.calls[options_chain.calls['strike'] > target_strike]
            
            if not suitable_options.empty:
                option_contract = suitable_options.iloc[0]  # Select the first suitable option
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - Trading conditions met. Attempting to place order for {contract_symbol}")
                order = place_option_trade(contract_symbol, qty=qty, option_type='call')

                if order:
                    entry_price = option_contract['lastPrice']
                    check_and_close_trade(entry_price, contract_symbol, qty)
                    placed_trades[stock_symbol] = current_time  # Record the trade time
            else:
                print(f"{stock_symbol} - No suitable options found for call purchase")
    
    elif trend < 0 and volume_signal:  # Downward trend, buy puts
        print(f"{stock_symbol} - Negative trend identified, buying puts")
        options_chain = get_options_chain(stock_symbol)
        if options_chain:
            target_strike = current_price * 0.90  # 10% below current price
            suitable_options = options_chain.puts[options_chain.puts['strike'] < target_strike]
            
            if not suitable_options.empty:
                option_contract = suitable_options.iloc[0]  # Select the first suitable option
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - Trading conditions met. Attempting to place order for {contract_symbol}")
                order = place_option_trade(contract_symbol, qty=qty, option_type='put')

                if order:
                    entry_price = option_contract['lastPrice']
                    check_and_close_trade(entry_price, contract_symbol, qty)
                    placed_trades[stock_symbol] = current_time  # Record the trade time
            else:
                print(f"{stock_symbol} - No suitable options found for put purchase")
    else:
        print(f"{stock_symbol} - No significant action - indicators not met")

def continuous_trading(qty=1, interval=180):
    while True:
        print("\\n--- Starting new trading cycle ---")
        
        # Check and trade on hardcoded stocks
        hardcoded_trades = check_hardcoded_stocks()
        for stock_symbol, price_movement, current_price in hardcoded_trades:
            print(f"{stock_symbol} - Hardcoded stock with price movement of {price_movement:.2f}% from open.")
            if price_movement > 0:
                print(f"{stock_symbol} - Upward trend, buying calls.")
                options_chain = get_options_chain(stock_symbol)
                if options_chain:
                    target_strike = current_price * 1.10  # 10% above current price
                    suitable_options = options_chain.calls[options_chain.calls['strike'] > target_strike]
                    if not suitable_options.empty:
                        option_contract = suitable_options.iloc[0]
                        contract_symbol = option_contract['contractSymbol']
                        place_option_trade(contract_symbol, qty=qty, option_type='call')
                        placed_trades[stock_symbol] = time.time()  # Track trade time
            else:
                print(f"{stock_symbol} - Downward trend, buying puts.")
                options_chain = get_options_chain(stock_symbol)
                if options_chain:
                    target_strike = current_price * 0.90  # 10% below current price
                    suitable_options = options_chain.puts[options_chain.puts['strike'] < target_strike]
                    if not suitable_options.empty:
                        option_contract = suitable_options.iloc[0]
                        contract_symbol = option_contract['contractSymbol']
                        place_option_trade(contract_symbol, qty=qty, option_type='put')
                        placed_trades[stock_symbol] = time.time()  # Track trade time

        # Check and trade on dynamically fetched active stocks
        stock_list = get_top_active_movers()
        for stock in stock_list:
            try:
                print(f"Symbol: {stock['symbol']}, Change %: {stock['change_percent']}, Volume: {stock['volume']}, Avg Volume: {stock['avg_volume']}")
                automated_trading(stock['symbol'], qty)
                print("----------------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"Error trading {stock['symbol']}: {e}")

        # Trade hardcoded stocks based on 1% price movement
        for stock_symbol in HARDCODED_STOCKS:
            try:
                trade_hardcoded_stocks(stock_symbol, qty)
            except Exception as e:
                print(f"Error trading {stock_symbol}: {e}")
        
        print(f"\nWaiting {interval} seconds before next cycle...")
        for remaining in range(interval, 0, -1):
            sys.stdout.write(f"\rNext refresh in {remaining} seconds...   ")
            sys.stdout.flush()
            time.sleep(1)
        
        sys.stdout.write("\rNext refresh in 0 seconds...    \n")
        sys.stdout.flush()

if __name__ == "__main__":
    continuous_trading(qty=1, interval=180)
