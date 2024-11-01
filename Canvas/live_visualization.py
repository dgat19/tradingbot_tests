import matplotlib.pyplot as plt
import matplotlib.animation as animation
from alpaca_trade_api.rest import REST

def live_strategy_visualization(api_key, secret_key, base_url, trade_history):
    if not trade_history:
        print("No trades to visualize in real-time.")
        return

    # Initialize Alpaca API
    api = REST(api_key, secret_key, base_url)
    
    # Set up figure for live plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Live Strategy Visualization - Trade Performance")
    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Return")

    def update(frame):
        # Refresh trade data for live updates
        returns = []
        for trade in trade_history:
            symbol = trade['symbol']
            position = api.get_position(symbol)
            if position:
                current_price = float(position.current_price)
                entry_price = trade['entry_price']
                trade_return = (current_price - entry_price) / entry_price
                returns.append(trade_return)
            else:
                returns.append(0)  # Default if no position open

        ax.clear()
        ax.plot(range(len(returns)), returns, marker='o', linestyle='-', color='b', label='Return')
        ax.set_title("Live Strategy Visualization - Trade Performance")
        ax.set_xlabel("Trade Index")
        ax.set_ylabel("Return")
        ax.legend()
        plt.tight_layout()

    animation.FuncAnimation(fig, update, frames=len(trade_history), repeat=False, interval=1000)
    plt.show()