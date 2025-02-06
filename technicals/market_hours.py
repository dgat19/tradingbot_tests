import datetime
import pytz

class MarketHours:
    def __init__(self):
        self.eastern = pytz.timezone('US/Eastern')
        self.market_open = datetime.time(9, 30)
        self.market_close = datetime.time(16, 0)
        self.extended_open = datetime.time(4, 0)
        self.extended_close = datetime.time(20, 0)

    def is_market_open(self):
        """Check if regular market is open"""
        now = datetime.datetime.now(self.eastern)
        
        if now.weekday() >= 5:  # Weekend
            return False
            
        return self.market_open <= now.time() <= self.market_close

    def is_extended_hours(self):
        """Check if in extended hours trading"""
        now = datetime.datetime.now(self.eastern)
        
        if now.weekday() >= 5:  # Weekend
            return False
            
        current_time = now.time()
        
        # Pre-market (4:00 AM - 9:30 AM ET)
        if self.extended_open <= current_time < self.market_open:
            return True
            
        # After-hours (4:00 PM - 8:00 PM ET)
        if self.market_close < current_time <= self.extended_close:
            return True
            
        return False

    def get_next_market_open(self):
        """Get next market open time"""
        now = datetime.datetime.now(self.eastern)
        current_date = now.date()
        
        # If weekend, move to Monday
        while current_date.weekday() >= 5:
            current_date += datetime.timedelta(days=1)
            
        next_open = datetime.datetime.combine(current_date, self.market_open)
        return self.eastern.localize(next_open)

    def get_current_market_session(self):
        """Get current market session status"""
        if self.is_market_open():
            return "Regular Hours"
        elif self.is_extended_hours():
            now = datetime.datetime.now(self.eastern)
            if now.time() < self.market_open:
                return "Pre-Market"
            else:
                return "After-Hours"
        else:
            return "Closed"

    def get_trading_hours(self):
        """Get trading hours information"""
        return {
            'regular_open': self.market_open,
            'regular_close': self.market_close,
            'extended_open': self.extended_open,
            'extended_close': self.extended_close
        }