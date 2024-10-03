import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import time
import sys
import contextlib
import os
import pytz
from datetime import datetime, timedelta
from news_scraper import get_news_sentiment, get_top_active_movers, get_trending_stocks
from indicators import analyze_indicators, get_stock_volatility
from ml_trade_performance_evaluation import load_trade_data, train_model, evaluate_trade, adjust_strategy_based_on_model

# Set up your Alpaca API keys (Replace with your own)
ALPACA_API_KEY = "PKOS59NZIX9P0L8VNJK8"
ALPACA_API_SECRET = "7eFxPPP15H3eYW3V0B5i1UdV21sCbS9oW1L4WcoA"
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Verify Alpaca connection
try:
    account = trading_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Account balance: {account.cash}")
    print(f"Options buying power: {account.options_buying_power}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")

# Load trade data and train the model
trade_data = load_trade_data()
model = train_model(trade_data)

hardcoded_stocks = ['NVDA', 'AAPL', 'MSFT', 'INTC', 'AVGO', 'LUNR', 'ASTS', 'PLTR']

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Function to get the correct open price from 8:30 AM UTC (which corresponds to the market open at 9:30 AM EST)
def get_open_price_at_market_open(symbol):
    try:
        stock = yf.Ticker(symbol)
        ny_tz = pytz.timezone('America/New_York')  # U.S. Eastern Timezone
        current_time = datetime.now(ny_tz).replace(hour=9, minute=30, second=0, microsecond=0)
        history = stock.history(period="1d", interval="1m")
        open_price = history.loc[history.index == current_time]['Open'].values[0]
        return open_price
    except Exception as e:
        print(f"Error retrieving open price at market open for {symbol}: {e}")
        return None

def get_current_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        with suppress_stdout():
            hist = stock.history(period="1d")
        if hist.empty:
            print(f"No data available for {symbol}")
            return None, None, None, None  # Ensure 4 values are returned
        
        current_price = hist['Close'].iloc[-1]
        open_price = get_open_price_at_market_open(symbol) # Get the 8:30 AM UTC open price
        current_volume = hist['Volume'].iloc[-1]
        volatility = get_stock_volatility(symbol)
        
        # Ensure valid prices before returning
        if current_price is None or open_price is None:
            raise ValueError(f"Invalid data for {symbol}: current_price or open_price is None")
        
        return current_price, open_price, current_volume, volatility
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None, None, None, None  # Ensure 4 values are returned even on error

def get_options_chain(symbol):
    try:
        stock = yf.Ticker(symbol)
        options_dates = stock.options
        if not options_dates:
            print(f"No options available for {symbol}")
            return None
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

# Modify the place_option_trade to return the entry price and track it
def place_option_trade(contract_symbol, qty, option_type='call'):
    try:
        # Check if the contract is valid
        if contract_symbol is None:
            raise ValueError(f"Invalid contract symbol for {contract_symbol}")
        
        order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        print(f"Order placed for {contract_symbol}")
        
        return order
    except Exception as e:
        print(f"Error placing order for {contract_symbol}: {str(e)}")
        return None

# Check stop-loss and take-profit conditions and exit the trade
def check_exit_conditions(contract_symbol, entry_price, qty):
    try:
        stock = yf.Ticker(contract_symbol[:4])
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1]

        print(f"Checking exit conditions for {contract_symbol}: Current Price = {current_price}, Entry Price = {entry_price}")

        # Stop-loss: Exit if current price drops below 25% of entry price
        if current_price <= entry_price * 0.75:
            print(f"{contract_symbol} - Stop-loss triggered. Exiting trade.")
            exit_trade(contract_symbol, qty)
            return

        # Take-profit: Exit if current price increases by 75% from entry price
        if current_price >= entry_price * 1.75:
            print(f"{contract_symbol} - Take-profit triggered. Exiting trade.")
            exit_trade(contract_symbol, qty)
            return
    except Exception as e:
        print(f"Error checking exit conditions for {contract_symbol}: {str(e)}")

# Function to exit a trade (sell the option)
def exit_trade(contract_symbol, qty):
    try:
        order_data = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        print(f"Exited trade for {contract_symbol}")
        return order
    except Exception as e:
        print(f"Error exiting trade for {contract_symbol}: {str(e)}")
        return None

# Monitor stop-loss and take-profit conditions
def check_and_close_trade(entry_price, contract_symbol, qty):
    print(f"Monitoring trade for {contract_symbol} with entry price: {entry_price} and quantity: {qty}")
    while True:
        check_exit_conditions(contract_symbol, entry_price, qty)
        time.sleep(60)  # Check every 60 seconds

# Modify trade_hardcoded_stocks to skip invalid stocks and ensure correct option selection
def trade_hardcoded_stocks(stock_symbol, qty=1):
    current_price, open_price, current_volume, volatility = get_current_stock_data(stock_symbol)
    
    if current_price is None or open_price is None:
        print(f"Skipping {stock_symbol} due to data retrieval error")
        return
    
    price_change = ((current_price - open_price) / open_price) * 100

    if abs(price_change) >= 3:
        options_chain = get_options_chain(stock_symbol)
        if options_chain is None:
            print(f"No options data for {stock_symbol}")
            return

        if price_change > 3:
            target_call_strike = current_price * 1.1
            itm_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
            if not itm_calls.empty:
                option_contract = itm_calls.iloc[0]
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - +3% price movement detected. Placing CALL order for {contract_symbol}")
                order = place_option_trade(contract_symbol, qty, option_type='call')
                if order:
                    print(f"Trade placed for {stock_symbol}: {order}")
            else:
                print(f"{stock_symbol} - No suitable ITM CALL options found")
        elif price_change < -3:
            target_put_strike = current_price * 0.9
            itm_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
            if not itm_puts.empty:
                option_contract = itm_puts.iloc[0]
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - -3% price movement detected. Placing PUT order for {contract_symbol}")
                order = place_option_trade(contract_symbol, qty, option_type='put')
                if order:
                    print(f"Trade placed for {stock_symbol}: {order}")
            else:
                print(f"{stock_symbol} - No suitable ITM PUT options found")
    else:
        print(f"{stock_symbol} - No significant price movement (±3%)")

def automated_trading(stock_symbol, qty=1):
    # Get current stock data
    current_price, open_price, current_volume, volatility = get_current_stock_data(stock_symbol)
    if current_price is None:
        print(f"Skipping {stock_symbol} due to data retrieval error")
        return

    # Get news sentiment and indicators
    sentiment_score = get_news_sentiment(stock_symbol)  # Now used
    indicators = analyze_indicators(stock_symbol)  # Now used

    print(f"{stock_symbol} - Sentiment Score: {sentiment_score:.4f}")
    print(f"{stock_symbol} - Monthly Performance: {indicators['monthly_performance']:.4f}")
    print(f"{stock_symbol} - High Volume: {'Yes' if indicators['high_volume'] else 'No'}")
    print(f"{stock_symbol} - Positive Trend: {'Yes' if indicators['positive_trend'] else 'No'}")

    # Check if options data is available
    options_chain = get_options_chain(stock_symbol)
    if options_chain is None:
        print(f"Skipping {stock_symbol} due to options data retrieval error")
        return

    # Decision logic based on volume and trend analysis
    if indicators['high_volume'] and volatility > 1:
        if indicators['positive_trend']:
            # Buy CALL options for positive trend
            target_call_strike = current_price * 1.1
            itm_calls = options_chain.calls[options_chain.calls['strike'] >= target_call_strike]
            if not itm_calls.empty:
                option_contract = itm_calls.iloc[0]
                contract_symbol = option_contract['contractSymbol']
                print(f"{stock_symbol} - Positive trend detected. Evaluating CALL trade for {contract_symbol}")

                # Evaluate the trade using the machine learning model
                predicted_outcome = evaluate_trade(model, stock_symbol, current_price, volatility, current_volume, sentiment_score)
                if predicted_outcome == 0:
                    print(f"{stock_symbol} - Model predicts a loss for CALL. Adjusting strategy.")
                    adjust_strategy_based_on_model(model, stock_symbol, current_price, volatility, current_volume, sentiment_score)
                    # Flip to PUT
                    target_put_strike = current_price * 0.9
                    itm_puts = options_chain.puts[options_chain.puts['strike'] <= target_put_strike]
                    if not itm_puts.empty:
                        option_contract = itm_puts.iloc[0]
                        contract_symbol = option_contract['contractSymbol']
                        print(f"{stock_symbol} - Placing PUT order for {contract_symbol}")
                        order = place_option_trade(contract_symbol, qty=qty, option_type='put')  # Now using `order`
                else:
                    print(f"{stock_symbol} - Model predicts profit for CALL. Placing order.")
                    order = place_option_trade(contract_symbol, qty=qty, option_type='call')  # Now using `order`
            else:
                print(f"{stock_symbol} - No suitable CALL options found.")


def continuous_trading(qty=1, interval=180):
    while True:
        print("\n--- Starting new trading cycle ---")
        stock_list = get_top_active_movers() + get_trending_stocks()
        for stock in stock_list:
            try:
                print(f"Symbol: {stock['symbol']}, Change %: {stock['change_percent']}, Volume: {stock['volume']}, Avg Volume: {stock['avg_volume']}")
                trade_hardcoded_stocks(stock['symbol'], qty)
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
    trade_data = load_trade_data()
    model = train_model(trade_data)
    
    continuous_trading(qty=1, interval=180)
