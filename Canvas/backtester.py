import numpy as np
from datetime import datetime, timedelta

class Backtester:
    def __init__(self):
        pass

    def backtest_strategy(self, strategy):
        # Define backtesting period
        start_date = datetime.now() - timedelta(days=365 * 3)  # 3-year backtest for robustness
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
            'Number of Trades': len(trades),
            'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
            'Sortino Ratio': self.calculate_sortino_ratio(returns)
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

    def calculate_sharpe_ratio(self, returns):
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        risk_free_rate = 0.01  # Assuming 1% annual risk-free rate
        sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0
        return sharpe_ratio

    def calculate_sortino_ratio(self, returns):
        mean_return = np.mean(returns)
        downside_std = np.std([r for r in returns if r < 0])
        risk_free_rate = 0.01
        sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std != 0 else 0
        return sortino_ratio
