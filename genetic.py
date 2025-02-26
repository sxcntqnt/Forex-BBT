# genetic.py
# genetic.py
from stratestic.backtesting import VectorizedBacktester
from stratestic.strategies._mixin import StrategyMixin
from collections import OrderedDict
import pandas as pd
import numpy as np
import asyncio
from typing import Tuple, Dict
from data_manager import DataManager  # Your provided DataManager
from datetime import datetime, timedelta

class PairsTradingStrategy(StrategyMixin):
    """
    A pairs trading strategy based on the spread between two assets.

    Parameters
    ----------
    symbol_pair : tuple
        A tuple of two symbol strings (e.g., ('BTCUSDT', 'ETHUSDT')).
    window : int
        The rolling window size for calculating spread mean and std.
    entry_threshold : float
        Z-score threshold for entering a trade.
    exit_threshold : float
        Z-score threshold for exiting a trade.
    """

    def __init__(
        self,
        symbol_pair: Tuple[str, str],
        window: int = 50,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        data=None,
        **kwargs
    ):
        self._symbol_pair = symbol_pair
        self._window = window
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold

        self.params = OrderedDict(
            window=lambda x: int(x),
            entry_threshold=lambda x: float(x),
            exit_threshold=lambda x: float(x)
        )

        StrategyMixin.__init__(self, data, **kwargs)

    def update_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Updates data with spread, mean, std, and z-score."""
        super().update_data(data)

        symbol1, symbol2 = self._symbol_pair
        # Extract close prices for both symbols from multi-index DataFrame
        spread = data.loc[symbol1, "close"] - data.loc[symbol2, "close"]
        data["spread"] = spread.reindex(data.index, level=1)  # Align with multi-index

        data["spread_mean"] = data["spread"].rolling(window=self._window).mean()
        data["spread_std"] = data["spread"].rolling(window=self._window).std()
        data["z_score"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]

        return data

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates positions based on z-score thresholds."""
        data["side"] = 0
        data.loc[data["z_score"] < -self._entry_threshold, "side"] = 1  # Long pair
        data.loc[data["z_score"] > self._entry_threshold, "side"] = -1  # Short pair
        data.loc[abs(data["z_score"]) < self._exit_threshold, "side"] = 0  # Exit
        data["position"] = data["side"].replace(0, method="ffill")
        return data

    def get_signal(self, row=None) -> int:
        """Returns signal for the current row."""
        if row is None or "z_score" not in row:
            return 0
        z_score = row["z_score"]
        if z_score < -self._entry_threshold:
            return 1  # Long pair
        elif z_score > self._entry_threshold:
            return -1  # Short pair
        elif abs(z_score) < self._exit_threshold:
            return 0  # Exit
        return row.get("position", 0)

class PairsTradingOptimizer:
    def __init__(self, data_manager: DataManager, symbol_pair: Tuple[str, str], initial_amount: float = 1000.0):
        self.data_manager = data_manager
        self.symbol_pair = symbol_pair
        self.initial_amount = initial_amount
        self.trading_costs = 0.1  # 0.1% trading costs
        self.optimized_params = None

    async def load_pairs_data(self, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
        """Load historical OHLC data for both symbols using DataManager."""
        symbol1, symbol2 = self.symbol_pair
        
        # Fetch historical data using DataManager's grab_historical_data
        tasks = [
            self.data_manager.grab_historical_data(start_timestamp, end_timestamp, symbol1, "minute"),
            self.data_manager.grab_historical_data(start_timestamp, end_timestamp, symbol2, "minute")
        ]
        df1, df2 = await asyncio.gather(*tasks)

        # Add symbol column and create multi-index
        df1["symbol"] = symbol1
        df2["symbol"] = symbol2
        combined = pd.concat([df1, df2]).set_index(["symbol", "timestamp"])
        return combined

    async def optimize_strategy(self, start_timestamp: int, end_timestamp: int) -> Dict[str, float]:
        """Optimize the pairs trading strategy using genetic algorithm."""
        strategy = PairsTradingStrategy(self.symbol_pair)
        backtester = VectorizedBacktester(
            strategy,
            symbol=self.symbol_pair[0],  # Primary symbol for logging
            amount=self.initial_amount,
            trading_costs=self.trading_costs
        )

        # Load historical data using DataManager
        pairs_data = await self.load_pairs_data(start_timestamp, end_timestamp)
        backtester.load_data(data=pairs_data)

        # Define optimization parameters
        opt_params = {
            "window": (20, 100, 10),  # Window: 20 to 100, step 10
            "entry_threshold": (1.0, 3.0, 0.5),  # Entry: 1.0 to 3.0, step 0.5
            "exit_threshold": (0.1, 1.0, 0.1)  # Exit: 0.1 to 1.0, step 0.1
        }

        # Run genetic optimization
        best_params, best_fitness = backtester.optimize(
            opt_params,
            optimizer="gen_alg",
            optimization_metric="Sharpe Ratio",
            pop_size=20,
            max_gen=30,
            mutation_rate=0.1,
            selection_rate=0.6,
            selection_strategy="roulette_wheel",
            fitness_tolerance=(1e-5, 10),
            verbose=True,
            plot_results=True
        )

        self.optimized_params = best_params
        return {"params": best_params, "sharpe_ratio": best_fitness}

    async def should_enter_trade(self) -> bool:
        """
        Check if a trade should be entered based on optimized strategy and latest data.

        Returns:
            bool: True if trade should be entered, False otherwise.
        """
        # Define a time range for the latest data (e.g., last 30 days)
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=30)).timestamp())

        if self.optimized_params is None:
            # Optimize if not already done
            await self.optimize_strategy(start_time, end_time)

        # Use optimized parameters
        strategy = PairsTradingStrategy(
            self.symbol_pair,
            window=self.optimized_params["window"],
            entry_threshold=self.optimized_params["entry_threshold"],
            exit_threshold=self.optimized_params["exit_threshold"]
        )

        # Load latest historical data
        pairs_data = await self.load_pairs_data(start_time, end_time)
        backtester = VectorizedBacktester(
            strategy,
            symbol=self.symbol_pair[0],
            amount=self.initial_amount,
            trading_costs=self.trading_costs
        )
        backtester.load_data(data=pairs_data)
        backtester.run()

        # Get the latest signal
        latest_row = backtester.data.tail(1).iloc[0]
        signal = strategy.get_signal(latest_row)
        return signal != 0  # True for long (1) or short (-1)

async def main():
    # Example usage
    from config import Config  # Placeholder; assumes Config exists
    import logging
    logger = logging.getLogger("example")
    
    config = Config()  # Placeholder
    api = None  # Placeholder; assumes API connection exists
    dm = DataManager(config, api, logger)
    
    optimizer = PairsTradingOptimizer(dm, symbol_pair=("BTCUSDT", "ETHUSDT"))
    
    # Optimize strategy
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=30)).timestamp())
    result = await optimizer.optimize_strategy(start_time, end_time)
    print(f"Optimized Parameters: {result['params']}")
    print(f"Best Sharpe Ratio: {result['sharpe_ratio']}")
    
    # Check trade signal
    trade_signal = await optimizer.should_enter_trade()
    print(f"Should enter trade: {trade_signal}")

if __name__ == "__main__":
    asyncio.run(main())
