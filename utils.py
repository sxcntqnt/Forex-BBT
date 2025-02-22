from datetime import datetime, timezone
from typing import Union
import time

from config import Config
from portfolio_manager import PortfolioManager


class TimestampUtils:
    """Utility class for timestamp conversions."""

    @staticmethod
    def to_seconds(dt: datetime) -> int:
        """Convert datetime to Unix timestamp in seconds."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    def to_milliseconds(dt: datetime) -> int:
        """Convert datetime to Unix timestamp in milliseconds."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def from_seconds(ts: Union[int, float]) -> datetime:
        """Convert Unix timestamp in seconds to UTC datetime."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    @staticmethod
    def from_milliseconds(ts: Union[int, float]) -> datetime:
        """Convert Unix timestamp in milliseconds to UTC datetime."""
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


class PerformanceMonitor:
    def __init__(self, PortfolioManager):
        # Track when the monitor starts
        self.start_time = time.time()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.portfolio_manager = PortfolioManager

    async def update(self, portfolio_manager):
        # Periodically called to update performance metrics
        elapsed_time = time.time() - self.start_time
        open_positions = portfolio_manager.get_open_positions()
        total_exposure = portfolio_manager.get_total_exposure()

        # Print the performance statistics
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Open positions: {open_positions}")
        print(f"Total exposure: {total_exposure}")
        print(f"Total trades: {self.total_trades}")
        print(
            f"Win rate: {self.winning_trades / self.total_trades if self.total_trades > 0 else 0:.2%}"
        )
        print(f"Total profit: ${self.total_profit:.2f}")

    def record_trade(self, profit):
        """This method is called whenever a trade is recorded."""
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        self.total_profit += profit

    def get_start_time(self):
        """Returns the start time of the monitor in seconds."""
        return self.start_time

    def get_start_time_in_readable_format(self):
        """Returns the start time as a readable datetime."""
        return TimestampUtils.from_seconds(
            self.start_time
        )  # Convert to readable datetime
