import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

def live_strategy_visualization(trade_history):
    if not trade_history:
        print("No trades to visualize in real-time.")
        return

    # Convert trade history to DataFrame for easier visualization
    trades_df = pd.DataFrame(trade_history)

    # Set up the figure and axis for live plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Live Strategy Visualization - Trade Performance")
    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Return")

    # Function to update the plot
    def update(frame):
        ax.clear()
        ax.plot(trades_df.index[:frame], trades_df['return'][:frame], marker='o', linestyle='-', color='b', label='Return')
        ax.set_title("Live Strategy Visualization - Trade Performance")
        ax.set_xlabel("Trade Index")
        ax.set_ylabel("Return")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(trades_df), repeat=False, interval=1000)
    plt.tight_layout()
    plt.show()