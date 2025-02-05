from datetime import datetime, time, timedelta
import pytz

class MarketHours:
    """Handles market hours checking and scheduling."""
    
    def __init__(self):
        self.est_tz = pytz.timezone('US/Eastern')
        self.market_open = time(9, 30)  # 9:30 AM EST
        self.market_close = time(16, 30)  # 4:30 PM EST

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        # Get current time in EST
        current_time = datetime.now(self.est_tz)
        
        # Check if it's a weekday
        if current_time.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
            
        # Get just the time part
        current_time = current_time.time()
        
        # Check if within market hours
        return self.market_open <= current_time <= self.market_close

    def time_until_market_open(self) -> float:
        """Calculate seconds until market opens."""
        current_time = datetime.now(self.est_tz)
        
        # If it's weekend, calculate time until Monday
        if current_time.weekday() >= 5:
            days_until_monday = (7 - current_time.weekday()) % 7
            next_market_day = current_time + timedelta(days=days_until_monday)
            market_open = datetime.combine(next_market_day.date(), self.market_open)
            market_open = self.est_tz.localize(market_open)
        else:
            # If it's before market hours today
            if current_time.time() < self.market_open:
                market_open = datetime.combine(current_time.date(), self.market_open)
                market_open = self.est_tz.localize(market_open)
            else:
                # Calculate for next business day
                next_day = current_time + timedelta(days=1)
                if next_day.weekday() >= 5:  # If next day is weekend
                    days_until_monday = (7 - next_day.weekday()) % 7
                    next_day = next_day + timedelta(days=days_until_monday)
                market_open = datetime.combine(next_day.date(), self.market_open)
                market_open = self.est_tz.localize(market_open)
        
        return (market_open - current_time).total_seconds()

    def time_until_market_close(self) -> float:
        """Calculate seconds until market closes."""
        current_time = datetime.now(self.est_tz)
        market_close = datetime.combine(current_time.date(), self.market_close)
        market_close = self.est_tz.localize(market_close)
        return (market_close - current_time).total_seconds()