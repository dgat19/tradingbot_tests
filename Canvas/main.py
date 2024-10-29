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

    # Train PPO model if necessary
    performance_tracker.train_ppo_model()

    # Backtest the trading strategies
    backtester.backtest_strategy(options_trader)
    backtester.backtest_strategy(swing_trader)
    backtester.backtest_strategy(potential_movers)

    # Execute options trading with PPO filtering
    options_trades = options_trader.execute_strategy()
    options_trades_filtered = [
        trade for trade in options_trades if performance_tracker.predict_trade_success(trade) > 0.5
    ]
    performance_tracker.evaluate_trades(options_trades_filtered)

    # Visualize trades and decisions
    visualization.visualize_trades(options_trades_filtered, "Options Trading")

    # Execute swing trading with PPO filtering
    swing_trades = swing_trader.execute_strategy()
    swing_trades_filtered = [
        trade for trade in swing_trades if performance_tracker.predict_trade_success(trade) > 0.5
    ]
    performance_tracker.evaluate_trades(swing_trades_filtered)

    # Visualize trades and decisions
    visualization.visualize_trades(swing_trades_filtered, "Swing Trading")

    # Execute trades based on top potential movers with PPO filtering
    movers_trades = potential_movers.execute_strategy()
    movers_trades_filtered = [
        trade for trade in movers_trades if performance_tracker.predict_trade_success(trade) > 0.5
    ]
    performance_tracker.evaluate_trades(movers_trades_filtered)

    # Visualize trades and decisions
    visualization.visualize_trades(movers_trades_filtered, "Potential Movers Trading")

    # Live visualization of ongoing trades
    live_visualization.live_strategy_visualization(performance_tracker.trade_history)

    # Review and learn from trades
    performance_tracker.learn_from_past_trades()

if __name__ == "__main__":
    main()

