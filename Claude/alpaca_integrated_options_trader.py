
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import time
import sys
from indicators import get_trend_indicator, get_volume_indicator
from datetime import datetime, timedelta

# Alpaca API keys - ensure to replace with your actual keys or environment variables
ALPACA_API_KEY = "your_alpaca_api_key"
ALPACA_API_SECRET = "your_alpaca_api_secret"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)  # Paper trading mode

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

def get_top_active_movers():
    # Placeholder for fetching the top stock movers
    return [
        {"symbol": "AAPL", "change_percent": 2.5, "volume": 5000000, "avg_volume": 3000000},
        {"symbol": "MSFT", "change_percent": -1.8, "volume": 6000000, "avg_volume": 4000000}
    ]

def automated_trading(stock_symbol, qty=1):
    print(f"Fetching data for {stock_symbol}")
    
    stock = yf.Ticker(stock_symbol)
    stock_price = stock.history(period="1d")["Close"].iloc[-1]
    current_price = stock_price

    trend = get_trend_indicator(stock_symbol)
    volume_signal = get_volume_indicator(stock_symbol)

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
            else:
                print(f"{stock_symbol} - No suitable options found for put purchase")
    else:
        print(f"{stock_symbol} - No significant action - indicators not met")

def continuous_trading(qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")
        stock_list = get_top_active_movers()
        for stock in stock_list:
            try:
                print(f"Symbol: {stock['symbol']}, Change %: {stock['change_percent']}, Volume: {stock['volume']}, Avg Volume: {stock['avg_volume']}")
                automated_trading(stock['symbol'], qty)
                print("----------------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"Error trading {stock['symbol']}: {e}")
        
        print(f"\nWaiting {interval} seconds before next cycle...")
        for remaining in range(interval, 0, -1):
            sys.stdout.write(f"\rNext refresh in {remaining} seconds...   ")
            sys.stdout.flush()
            time.sleep(1)
        
        sys.stdout.write("\rNext refresh in 0 seconds...    \n")
        sys.stdout.flush()

if __name__ == "__main__":
    continuous_trading(qty=1, interval=180)
