from options_trading import OptionsTrader
from swing_trading import SwingTrader
from performance_tracker import PerformanceTracker
from potential_movers import PotentialMovers
from backtester import Backtester
import visualization
import live_visualization

def main():
    # Initialize the trading strategies
    options_trader = OptionsTrader()
    swing_trader = SwingTrader()
    potential_movers = PotentialMovers()
    performance_tracker = PerformanceTracker()
    backtester = Backtester()

    # Backtest the trading strategies
    backtester.backtest_strategy(options_trader)
    backtester.backtest_strategy(swing_trader)
    backtester.backtest_strategy(potential_movers)

    # Execute options trading
    options_trades = options_trader.execute_strategy()
    performance_tracker.evaluate_trades(options_trades)

    # Visualize trades and decisions
    visualization.visualize_trades(options_trades, "Options Trading")

    # Execute swing trading
    swing_trades = swing_trader.execute_strategy()
    performance_tracker.evaluate_trades(swing_trades)

    # Visualize trades and decisions
    visualization.visualize_trades(swing_trades, "Swing Trading")

    # Execute trades based on top potential movers
    movers_trades = potential_movers.execute_strategy()
    performance_tracker.evaluate_trades(movers_trades)

    # Visualize trades and decisions
    visualization.visualize_trades(movers_trades, "Potential Movers Trading")

    # Live visualization of ongoing trades
    live_visualization.live_strategy_visualization(performance_tracker.trade_history)

    # Review and learn from trades
    performance_tracker.learn_from_past_trades()

if __name__ == "__main__":
    main()