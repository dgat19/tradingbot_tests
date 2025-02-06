import pandas as pd

class DisplayHandler:
    def __init__(self, logger):
        self.logger = logger

    def display_ticker_info(self, ticker, intraday_df, daily_df=None):
        """Display comprehensive ticker information with all indicators"""
        try:
            # Get daily data info
            yesterday_close = float('nan')
            today_open = float('nan')
            
            if daily_df is not None and len(daily_df) >= 2:
                yesterday_close = daily_df['close'].iloc[-2]
                today_open = daily_df['open'].iloc[-1]

            # Check for intraday data
            if intraday_df is None or intraday_df.empty:
                print(f"\nTicker: {ticker}")
                print(f"  Yesterday's Close: {yesterday_close:.2f}")
                print(f"  Today's Open: {today_open:.2f}")
                self.logger.log_warning(f"No intraday data available for {ticker}")
                return

            # Get current values
            current_price = intraday_df['close'].iloc[-1]
            current_di_plus = intraday_df['DI+'].iloc[-1]
            current_di_minus = intraday_df['DI-'].iloc[-1]
            current_adx = intraday_df['ADX'].iloc[-1]
            current_macd = intraday_df['MACD'].iloc[-1]
            current_macd_signal = intraday_df['Signal_Line'].iloc[-1]
            current_macd_hist = intraday_df['MACD_Hist'].iloc[-1]

            # Get last 5 values for each indicator
            last_5_di_plus = intraday_df['DI+'].tail(5).values
            last_5_di_minus = intraday_df['DI-'].tail(5).values
            last_5_adx = intraday_df['ADX'].tail(5).values
            last_5_macd_hist = intraday_df['MACD_Hist'].tail(5).values

            # Calculate price change
            price_change = ((current_price - today_open) / today_open) * 100 if not pd.isna(today_open) else 0

            # Display information
            print(f"\n{'='*50}")
            print(f"Ticker: {ticker}")
            print(f"{'='*50}")
            
            # Price Information
            print("\nPrice Information:")
            print(f"  Yesterday's Close: {yesterday_close:.2f}")
            print(f"  Today's Open: {today_open:.2f}")
            print(f"  Current Price: {current_price:.2f} ({price_change:+.2f}%)")
            
            # Directional System
            print("\nDirectional System:")
            print(f"  Current DI+: {current_di_plus:.2f}")
            print(f"  Current DI-: {current_di_minus:.2f}")
            print(f"  Current ADX: {current_adx:.2f}")
            print(f"  Trend Strength: {'Strong' if current_adx > 25 else 'Weak'}")
            print(f"  Last 5 DI+ values: {', '.join([f'{x:.2f}' for x in last_5_di_plus])}")
            print(f"  Last 5 DI- values: {', '.join([f'{x:.2f}' for x in last_5_di_minus])}")
            print(f"  Last 5 ADX values: {', '.join([f'{x:.2f}' for x in last_5_adx])}")
            
            # MACD System
            print("\nMACD System:")
            print(f"  MACD: {current_macd:.4f}")
            print(f"  Signal Line: {current_macd_signal:.4f}")
            print(f"  Histogram: {current_macd_hist:.4f}")
            print(f"  Last 5 Histogram values: {', '.join([f'{x:.4f}' for x in last_5_macd_hist])}")
            
            # Trend Analysis
            trend_direction = self._analyze_trend(current_di_plus, current_di_minus, current_adx)
            macd_signal = self._analyze_macd(current_macd_hist, last_5_macd_hist)
            
            print("\nTrend Analysis:")
            print(f"  Direction: {trend_direction}")
            print(f"  MACD Signal: {macd_signal}")
            print(f"{'='*50}")

            # Log additional information
            self.logger.log_market_data(ticker, {
                'current_price': current_price,
                'price_change': price_change,
                'di_plus': current_di_plus,
                'di_minus': current_di_minus,
                'adx': current_adx,
                'macd_hist': current_macd_hist
            })

        except Exception as e:
            self.logger.log_error(f"Error displaying ticker info for {ticker}", e)

    def _analyze_trend(self, di_plus, di_minus, adx):
        """Analyze trend direction and strength"""
        if adx < 25:
            return "No Clear Trend (Weak ADX)"
        elif di_plus > di_minus:
            return f"Uptrend (Strong ADX: {adx:.1f})"
        else:
            return f"Downtrend (Strong ADX: {adx:.1f})"

    def _analyze_macd(self, current_hist, hist_values):
        """Analyze MACD histogram for signals"""
        if len(hist_values) < 2:
            return "Insufficient Data"
        
        prev_hist = hist_values[-2]
        if current_hist > 0 and prev_hist < 0:
            return "Bullish Crossover"
        elif current_hist < 0 and prev_hist > 0:
            return "Bearish Crossover"
        elif current_hist > 0:
            return "Above Signal Line"
        else:
            return "Below Signal Line"

    def display_signal(self, ticker, signal_type, signal_strength=None):
        """Display trading signals with strength"""
        if signal_type is None:
            print(f"{ticker}: No signal generated.")
        else:
            strength_info = f" (Strength: {signal_strength:.2%})" if signal_strength is not None else ""
            print(f"{ticker}: {signal_type.capitalize()} signal generated{strength_info}")

    def display_trade_execution(self, ticker, trade_type, quantity, price, signal_strength):
        """Display trade execution information"""
        print(f"\nTrade Execution - {ticker}:")
        print(f"  Type: {trade_type.upper()}")
        print(f"  Quantity: {quantity:.2f}")
        print(f"  Price: ${price:.2f}")
        print(f"  Signal Strength: {signal_strength:.2%}")
        print(f"  Total Value: ${(quantity * price):.2f}")

    def display_account_info(self, balance):
        """Display account information"""
        print(f"\n{'='*50}")
        print("Account Information:")
        print(f"{'='*50}")
        print(f"Current Balance: ${balance:,.2f}")
        print(f"{'='*50}\n")