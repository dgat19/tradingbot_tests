import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, OrderStatus
import time
import sys
from datetime import datetime, timedelta
from news_scraper import get_news_sentiment
from indicators import analyze_indicators, get_stock_volatility

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PK9RIB7H3DVU9FMHROR7"
ALPACA_API_SECRET = "dvwSlk4p1ZKBqPsLGJbehu1dAcd82MSwLJ5BgHVh"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

def get_current_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1d")
    current_price = hist['Close'].iloc[-1]
    current_volume = hist['Volume'].iloc[-1]
    volatility = get_stock_volatility(symbol)
    return current_price, current_volume, volatility

def get_options_chain(symbol):
    try:
        stock = yf.Ticker(symbol)
        options_dates = stock.options
        
        # Choose an expiration date within a month
        today = datetime.now()
        one_month_later = today + timedelta(days=30)
        valid_dates = [date for date in options_dates if datetime.strptime(date, "%Y-%m-%d") <= one_month_later]
        
        if not valid_dates:
            return None
        
        latest_valid_date = max(valid_dates)
        options_chain = stock.option_chain(latest_valid_date)
        return options_chain
    except Exception as e:
        print(f"Error fetching options chain for {symbol}: {e}")
        return None

def place_option_trade(contract_symbol, qty, option_type='call'):
    try:
        side = OrderSide.BUY if option_type == 'call' else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE
        )
        
        order = trading_client.submit_order(order_data=market_order_data)
        
        if order.status == OrderStatus.FILLED:
            print(f"Option order for {contract_symbol} ({side}) has been filled.")
            return order
        else:
            print(f"Option order for {contract_symbol} ({side}) is {order.status}.")
            return None
    except Exception as e:
        print(f"Error placing options trade for {contract_symbol}: {e}")
        return None

def check_and_close_trade(entry_price, contract_symbol, qty):
    try:
        option_data = yf.Ticker(contract_symbol).history(period="1d", interval="1m")
        current_price = option_data['Close'].iloc[-1]
        
        gain_percentage = ((current_price - entry_price) / entry_price) * 100
        
        if gain_percentage >= 75:
            print(f"Closing trade for {contract_symbol} at a gain of {gain_percentage:.2f}%")
            place_option_trade(contract_symbol, qty, option_type='sell')
        else:
            print(f"Current gain for {contract_symbol} is {gain_percentage:.2f}%, waiting to close.")
    except Exception as e:
        print(f"Error checking trade for {contract_symbol}: {e}")

def automated_trading(stock_symbol, qty=1):
    # Get current stock data
    current_price, current_volume, volatility = get_current_stock_data(stock_symbol)
    print(f"\n{stock_symbol} - Current Price: ${current_price:.2f}, Volume: {current_volume:,}, Volatility: {volatility:.4f}")

    # Get news sentiment
    sentiment_score = get_news_sentiment(stock_symbol)
    print(f"{stock_symbol} - Sentiment Score: {sentiment_score:.4f}")
    
    # Get indicators
    indicators = analyze_indicators(stock_symbol)
    print(f"{stock_symbol} - Monthly Performance: {indicators['monthly_performance']:.4f}")
    print(f"{stock_symbol} - High Volume: {'Yes' if indicators['high_volume'] else 'No'}")
    print(f"{stock_symbol} - Positive Trend: {'Yes' if indicators['positive_trend'] else 'No'}")
    
    # Get options chain
    options_chain = get_options_chain(stock_symbol)
    if options_chain is None:
        return

    # Decision logic based on sentiment, indicators, and options
    should_trade = (
        sentiment_score > 0.05 and
        indicators['monthly_performance'] > 0 and
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

def continuous_trading(stock_list, qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")
        for stock_symbol in stock_list:
            try:
                automated_trading(stock_symbol, qty)
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
    stock_list = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NVDA", "AVGO", "INTC", "ASTS", "LUNR"]
    continuous_trading(stock_list, qty=1, interval=180)