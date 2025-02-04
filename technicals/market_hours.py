from datetime import datetime, time, timedelta
import pytz
import pandas as pd
import pandas_market_calendars as mcal

nyse = mcal.get_calendar('NYSE')

class MarketHours:
    """Handles market hours checking and scheduling."""
    
    def __init__(self):
        self.est_tz = pytz.timezone('US/Eastern')
        self.market_open = time(9, 30)  # 9:30 AM EST
        self.market_close = time(16, 30)  # 4:30 PM EST

    def is_market_open_now(self) -> bool:
        now_utc = pd.Timestamp.utcnow()
        schedule = nyse.schedule(start_date=now_utc.date(), end_date=now_utc.date())
        
        # If schedule is empty (e.g., weekend, holiday), clearly market is closed
        if schedule.empty:
            return False
        
        # Otherwise, wrap `open_at_time` in a try/except
        try:
            return nyse.open_at_time(schedule, now_utc)
        except ValueError:
            # The timestamp isn't covered by today's schedule => closed
            return False


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