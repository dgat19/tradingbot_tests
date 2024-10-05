import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Backtester:
    def __init__(self):
        pass

    def backtest_strategy(self, strategy):
        # Define backtesting period
        start_date = datetime.now() - timedelta(days=365)  # 1-year backtest
        end_date = datetime.now()

        # Generate historical data and simulate trading
        historical_trades = strategy.backtest(start_date, end_date)
        performance = self.calculate_performance(historical_trades)
        
        print(f"Backtest Performance for {strategy.__class__.__name__}: {performance}")

    def calculate_performance(self, trades):
        if not trades:
            return "No trades executed."

        returns = [trade['return'] for trade in trades]
        total_return = np.sum(returns)
        average_return = np.mean(returns) if returns else 0
        max_drawdown = self.calculate_max_drawdown(returns)

        performance_summary = {
            'Total Return': total_return,
            'Average Return': average_return,
            'Max Drawdown': max_drawdown,
            'Number of Trades': len(trades)
        }

        return performance_summary

    def calculate_max_drawdown(self, returns):
        cumulative_returns = np.cumsum(returns)
        peak = cumulative_returns[0]
        max_drawdown = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown