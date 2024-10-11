# visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def visualize_trades(trades, strategy_name):
    if not trades:
        print(f"No trades to visualize for {strategy_name}.")
        return

    # Convert trades to DataFrame for easier visualization
    trades_df = pd.DataFrame(trades)

    # Plot trades entry price and decision criteria
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot entry prices
    ax[0].plot(trades_df.index, trades_df['entry_price'], marker='o', linestyle='-', color='b', label='Entry Price')
    ax[0].set_title(f'{strategy_name} - Entry Prices')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Plot volatility
    ax[1].plot(trades_df.index, trades_df['volatility'], marker='o', linestyle='-', color='r', label='Volatility')
    ax[1].set_title(f'{strategy_name} - Volatility')
    ax[1].set_ylabel('Volatility')
    ax[1].legend()

    # Plot news sentiment
    if 'news_sentiment' in trades_df.columns:
        ax[2].plot(trades_df.index, trades_df['news_sentiment'], marker='o', linestyle='-', color='g', label='News Sentiment')
    else:
        ax[2].plot(trades_df.index, [0]*len(trades_df), marker='o', linestyle='-', color='g', label='News Sentiment')
    ax[2].set_title(f'{strategy_name} - News Sentiment')
    ax[2].set_ylabel('Sentiment Score')
    ax[2].legend()

    plt.xlabel('Trade Index')
    plt.tight_layout()
    plt.show()
