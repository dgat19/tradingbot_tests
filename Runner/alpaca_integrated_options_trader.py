
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import time
import sys
import logging
from news_scraper import get_top_active_movers
from indicators import get_trend_indicator, get_volume_indicator
from ml_trade_performance_evaluation import load_trade_data, train_model, evaluate_trade, adjust_strategy_based_on_model


# Alpaca API keys - ensure to replace with your actual keys or environment variables
ALPACA_API_KEY = "PKLSUU1T53HAXFDKFQMY"
ALPACA_API_SECRET = "M46BGIZBuunwXIgDU1ttxnNj0nURPZfxt1IjLkdr"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)  # Paper trading mode

# Hardcoded list of stocks that can be traded based on 1% price movement
hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']

# Track open trades to avoid duplicate trades
open_trades = {}

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

# Define the place_trade function
def place_trade(stock_symbol, qty, option_type='call'):
    options_chain = get_options_chain(stock_symbol)
    if options_chain:
        if option_type == 'call':
            target_strike = 1.10 * yf.Ticker(stock_symbol).history(period="1d")["Close"].iloc[-1]  # 10% above current price
            suitable_options = options_chain.calls[options_chain.calls['strike'] > target_strike]
        elif option_type == 'put':
            target_strike = 0.90 * yf.Ticker(stock_symbol).history(period="1d")["Close"].iloc[-1]  # 10% below current price
            suitable_options = options_chain.puts[options_chain.puts['strike'] < target_strike]
        
        if not suitable_options.empty:
            option_contract = suitable_options.iloc[0]  # Select the best option
            contract_symbol = option_contract['contractSymbol']
            print(f"Placing {option_type} order for {contract_symbol}")
            place_option_trade(contract_symbol, qty, option_type)

def trade_hardcoded_stocks(qty=1):
    # Check for price movements in hardcoded stocks
    for stock_symbol in hardcoded_stocks:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="1d")
        open_price = stock_data["Open"].iloc[-1]
        current_price = stock_data["Close"].iloc[-1]
        price_change = (current_price - open_price) / open_price * 100

        # Only trade if price moves more than 1% and avoid duplicate trades
        if abs(price_change) > 1 and stock_symbol not in open_trades:
            print(f"{stock_symbol} moved {price_change:.2f}% from the open price. Trading now.")
            trend = 1 if price_change > 1 else -1  # Upward or downward trend
            volume_signal = True  # Assume volume is sufficient for hardcoded stocks
            place_trade_for_hardcoded(stock_symbol, trend, volume_signal, qty)

def place_trade_for_hardcoded(stock_symbol, trend, volume_signal, qty=1):
    if trend > 0 and volume_signal:  # Upward trend, buy calls
        place_trade(stock_symbol, qty, option_type='call')
    elif trend < 0 and volume_signal:  # Downward trend, buy puts
        place_trade(stock_symbol, qty, option_type='put')

def automated_trading(stock_symbol, qty=1):
    print(f"Fetching data for {stock_symbol}")
    
    stock = yf.Ticker(stock_symbol)
    stock_price = stock.history(period="1d")["Close"].iloc[-1]

    # Get trend and volume indicators
    trend = get_trend_indicator(stock_symbol)
    volume_signal = get_volume_indicator(stock_symbol)

    # Evaluate the trade using the machine learning model before proceeding
    predicted_outcome = evaluate_trade(model, stock_symbol, price_at_trade=stock_price, volatility=0.35, volume_signal=volume_signal, market_sentiment=trend)

    # If the model predicts a loss, adjust the strategy
    if predicted_outcome == 0:
        print(f"Model predicts a loss for {stock_symbol}. Adjusting strategy.")
        adjust_strategy_based_on_model(model, stock_symbol, price_at_trade=stock_price, volatility=0.35, volume_signal=volume_signal, market_sentiment=trend)
        return  # Skip the trade if a loss is predicted

    # If model predicts profit, proceed with the trade
    if trend > 0 and volume_signal:  # Buy calls for upward trend
        print(f"{stock_symbol} - Positive trend identified, buying calls")
        place_trade(stock_symbol, qty, option_type='call')

    elif trend < 0 and volume_signal:  # Buy puts for downward trend
        print(f"{stock_symbol} - Negative trend identified, buying puts")
        place_trade(stock_symbol, qty, option_type='put')

def continuous_trading(qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")
        stock_list = get_top_active_movers()  # Using get_top_active_movers from news_scraper
        trade_hardcoded_stocks(qty)  # Trade hardcoded stocks
        
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

# Main function to load data, train model, and start trading
if __name__ == "__main__":
    # Load trade data and train the model
    trade_data = load_trade_data()
    model = train_model(trade_data)
    
    continuous_trading(qty=1, interval=180)
