import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType #, OrderClass, OrderStatus
import time
import sys
import contextlib
import os
from datetime import datetime, timedelta
from news_scraper import get_news_sentiment, get_top_active_movers
from indicators import analyze_indicators, get_stock_volatility

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKLSUU1T53HAXFDKFQMY"
ALPACA_API_SECRET = "M46BGIZBuunwXIgDU1ttxnNj0nURPZfxt1IjLkdr"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
    print(f"Account cash withdrawal: {account.options_buying_power}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_current_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        with suppress_stdout():
            hist = stock.history(period="1d")
        if hist.empty:
            print(f"No data available for {symbol}")
            return None, None, None
        current_price = hist['Close'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        volatility = get_stock_volatility(symbol)
        return current_price, current_volume, volatility
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None, None, None

def get_options_chain(symbol):
    try:
        stock = yf.Ticker(symbol)
        options_dates = stock.options
        
        if not options_dates:
            print(f"No options available for {symbol}")
            return None
        
        # Choose an expiration date within a month
        today = datetime.now()
        one_month_later = today + timedelta(days=30)
        valid_dates = [date for date in options_dates if datetime.strptime(date, "%Y-%m-%d") <= one_month_later]
        
        if not valid_dates:
            print(f"No valid options dates within a month for {symbol}")
            return None
        
        latest_valid_date = max(valid_dates)
        options_chain = stock.option_chain(latest_valid_date)
        return options_chain
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {str(e)}")
        return None

def place_option_trade(contract_symbol, qty, option_type='call'):
    try:
        order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        order = trading_client.submit_order(order_data)
        print(f"Order placed for {contract_symbol}: {order}")
        return order
    except Exception as e:
        print(f"Error placing order for {contract_symbol}: {str(e)}")
        return None

def check_and_close_trade(entry_price, contract_symbol, qty):
    # Define your logic to check and close trades here
    print(f"Checking and closing trades for: {contract_symbol} with entry price: {entry_price} and quantity: {qty}")


def automated_trading(stock_symbol, qty=1):
    # Get current stock data
    current_price, current_volume, volatility = get_current_stock_data(stock_symbol)
    if current_price is None:
        print(f"Skipping {stock_symbol} due to data retrieval error")
        return

    # Get top active movers data
    stock_list = get_top_active_movers()
    stock_data = next((stock for stock in stock_list if stock['symbol'] == stock_symbol), None)
    if not stock_data:
        print(f"No active mover data found for {stock_symbol}")
        return

    change_percent = stock_data['change_percent']
    volume = stock_data['volume']
    avg_volume = stock_data['avg_volume']

    print(f"\n{stock_symbol} - Current Price: ${current_price:.2f}, Change %: {change_percent} \nVolume: {volume}, Avg Volume: {avg_volume} \nVolatility: {volatility:.4f}")

    # Get news sentiment, indicators, options chain
    sentiment_score = get_news_sentiment(stock_symbol)
    indicators = analyze_indicators(stock_symbol)
    options_chain = get_options_chain(stock_symbol)

    print(f"{stock_symbol} - Sentiment Score: {sentiment_score:.4f}")
    print(f"{stock_symbol} - Monthly Performance: {indicators['monthly_performance']:.4f}")
    print(f"{stock_symbol} - High Volume: {'Yes' if indicators['high_volume'] else 'No'}")
    print(f"{stock_symbol} - Positive Trend: {'Yes' if indicators['positive_trend'] else 'No'}")
    
    if options_chain is None:
        print(f"Skipping {stock_symbol} due to options data retrieval error")
        return

    # Decision logic based on volume and trend analysis
    should_trade = (
        indicators['high_volume'] and
        indicators['positive_trend'] and
        volatility > 1
    )

    if should_trade:
        # Choose a strike price (for simplicity, choosing the first in-the-money option)
        itm_options = options_chain.calls[options_chain.calls['strike'] < current_price]
        
        if not itm_options.empty:
            option_contract = itm_options.iloc[-1]  # Choose the closest in-the-money option
            contract_symbol = option_contract['contractSymbol']
            
            print(f"{stock_symbol} - Trading conditions met. Attempting to place order for {contract_symbol}")
            order = place_option_trade(contract_symbol, qty=qty, option_type='call')
            
            if order:
                entry_price = option_contract['lastPrice']
                check_and_close_trade(entry_price, contract_symbol, qty)
        else:
            print(f"{stock_symbol} - No suitable in-the-money options found")
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