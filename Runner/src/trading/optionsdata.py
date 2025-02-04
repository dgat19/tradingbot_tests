from ib_insync import *
util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=12)

from ib_insync import IB, Stock, MarketOrder, Forex



# Define a callback to monitor order status
def onOrderStatus(trade):
    print("Order ID")

# Attach the callback to the orderStatusEvent
ib.orderStatusEvent += onOrderStatus

# Define a stock contract
contract = Forex("EURUSD")

# Place a market order
order = MarketOrder('BUY', 10)
trade = ib.placeOrder(contract, order)

# Wait to receive updates
ib.sleep(5)

nvda = Stock("NVDA", "SMART", "USD")
ib.qualifyContracts(nvda)
[ticker] = ib.reqTickers(nvda)
nvdaValue = ticker.marketPrice()

chains = ib.reqSecDefOptParams(nvda.symbol, "", nvda.secType, nvda.conId)
chain = next(c for c in chains if c.tradingClass == "NVDA" and c.exchange == "SMART")

strikes = [
    strike for strike in chain.strikes
    if strike % 5 == 0
    and nvdaValue - 20 < strike < nvdaValue + 20
]

expirations = sorted(exp for exp in chain.expirations)[:3]
rights = ["C", "P"]

contracts = [
    Option("NVDA", expiration, strike, right, "SMART", tradingClass="NVDA")
    for right in rights
    for expiration in expirations
    for strike in strikes
]

contracts = ib.qualifyContracts(*contracts)
tickers = ib.reqTickers(*contracts)

contractData = [
    (
        t.contract.lastTradeDateOrContractMonth, 
        t.contract.strike, 
        t.contract.right,
        t.time,
        t.close,
        nvdaValue,
    )
    for t
    in tickers
]

fields = [
    "expiration", 
    "strike", 
    "right", 
    "time", 
    "undPrice",
    "close", 
]

util.df([dict(zip(fields, t)) for t in contractData])