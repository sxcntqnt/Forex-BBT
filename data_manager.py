import pandas as pd
import asyncio
from config import Config
from deriv_api import DerivAPI
from typing import List, Dict, Union
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import RollingGroupby


class DataManager:
    def __init__(self, config: Config, api, logger, data=None, max_data_points: int = 100000) -> None:
        """Initializes the Stock Data Manager with real-time data from Deriv API.

        Args:
            config: Configuration object with settings like symbols.
            api: Connection object for the Deriv API.
            logger: Logger object for logging.
            data: Optional initial data (defaults to None).
            max_data_points: Max number of data points to store for each symbol.
        """
        self.config = config
        self.api = api
        self.symbols = config.SYMBOLS
        self.max_data_points = max_data_points
        self.data_frames = {
            symbol: pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "bid", "ask", "quote"])
            for symbol in self.symbols
        }
        self.logger = logger
        self.data = data if data is not None else {}
        self.subscriptions = {}
        self.last_data = {}
        self._symbol_groups = None  # Initialize appropriately
        self._frame = pd.DataFrame()  # Initialize your main dataframe

    @property
    def frame(self):
        return self._frame

    def create_subs_cb(self, symbol: str):
        """Generates a callback function for real-time data updates.

        Args:
            symbol: The symbol to track.

        Returns:
            Callbackfunction to handle real-time tick data.
        """
        count = 0

        def cb(data):
            nonlocal count
            count += 1
            self.last_data[symbol] = data
            self.logger.debug(f"Received data for symbol {symbol}: {data} (Count: {count})")
            self.update(symbol, data)

        return cb

    async def subscribe_to_ticks(self, symbol):
        """Subscribe to tick stream for a given symbol."""
        try:
            if not isinstance(symbol, str) or not symbol:
                self.logger.error(f"Invalid symbol: {symbol}")
                return

            tick_stream = await self.api.subscribe({"ticks": symbol, "subscribe": 1})
            self.subscriptions[symbol] = tick_stream
            tick_stream.subscribe(self.create_subs_cb(symbol))
            self.logger.info(f"Subscribed to {symbol} tick stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")

    async def start_subscriptions(self) -> None:
        """Starts WebSocket subscriptions for all symbols."""
        if not self.symbols:
            self.logger.warning("No symbols to subscribe to.")
            return

        tasks = [self.subscribe_to_ticks(symbol) for symbol in self.symbols]

        try:
            await asyncio.gather(*tasks)
            self.logger.info("All subscriptions started successfully.")
        except Exception as e:
            self.logger.error(f"Error during subscription: {e}")

    async def update(self, symbol: str, tick: Dict[str, Union[int, float]]) -> None:
        """Updates the DataFrame for the specified symbol with new tick data.

        Args:
            symbol: The symbol being updated.
            tick: The tick data (a dictionary containing price data).
        """
        try:
            if "tick" not in tick:
                self.logger.error(f"Invalid tick data for symbol {symbol}: 'tick' key missing.")
                return

            epoch = tick["tick"].get("epoch")
            ask = tick["tick"].get("ask")
            bid = tick["tick"].get("bid")
            quote = tick["tick"].get("quote")

            if None in (epoch, ask, bid, quote):
                self.logger.error(f"Incomplete tick data for symbol {symbol}: {tick}")
                return

            new_data = pd.DataFrame({
                "symbol": [symbol],
                "timestamp": [epoch],
                "open": [bid],
                "high": [ask],
                "low": [bid],
                "close": [bid],
                "bid": [bid],
                "ask": [ask],
                "quote": [quote],
            })

            if symbol in self.data_frames:
                self.data_frames[symbol] = pd.concat([self.data_frames[symbol], new_data]).tail(self.max_data_points)
            else:
                self.logger.warning(f"Symbol {symbol} not found in managed symbols.")
        except KeyError as e:
            self.logger.error(f"KeyError: Missing expected key in tick data - {e}")
        except Exception as e:
            self.logger.error(f"Error updating symbol {symbol}: {e}")

    async def stop_subscriptions(self):
        """Proper subscription cleanup"""
        for symbol, sub in self.subscriptions.items():
            try:
                await self.api.forget(sub.id)
            except Exception as e:
                logger.error(f"Error unsubscribing {symbol}: {str(e)}")
        self.subscriptions.clear()


    def get_close_prices(self, symbol: str) -> List[float]:
        """Retrieves the close prices for the specified symbol.

        Args:
            symbol: The symbol for which to retrieve close prices.

        Returns:
            A list of close prices.
        """
        df = self.data_frames.get(symbol)
        if df is not None:
            return df["close"].tolist()
        else:
            raise ValueError(f"Symbol '{symbol}' not found.")

    def get_ohlc_data(self, symbol: str) -> pd.DataFrame:
        """Retrieves the OHLC data for the specified symbol.

        Args:
            symbol: The symbol for which to retrieve OHLC data.

        Returns:
            A DataFrame containing OHLC data for the symbol.
        """
        df = self.data_frames.get(symbol)
        if df is not None:
            return df
        else:
            raise ValueError(f"Symbol '{symbol}' not found.")

    @property
    def symbol_groups(self) -> DataFrameGroupBy:
        """Returns a GroupBy object of symbols in the data frames.

        Returns:
            A pandas GroupBy object grouped by symbol.
        """
        if not self.data:
            return pd.DataFrame().groupby([])
        
        return pd.concat(self.data.values()).groupby("symbol", as_index=False)

    async def symbol_rolling_groups(self, size: int) -> RollingGroupby:
        """Grabs rolling windows of data for each symbol.

        Args:
            size: The size of the rolling window.

        Returns:
            A RollingGroupby object containing rolling windows of data.
        """
        df = pd.concat(self.data_frames.values())
        return df.groupby("symbol").rolling(size)

    def add_rows(self, data: Dict) -> None:
        """Adds new rows to the stock data.

        Args:
            data: A dictionary of quotes containing 'symbol', 'datetime', 'open', 'close', 'high', 'low', 'volume'.
        """
        column_names = ["open", "close", "high", "low", "volume"]

        for quote in data:
            time_stamp = pd.to_datetime(quote["datetime"], unit="ms", origin="unix")
            row_id = (quote["symbol"], time_stamp)
            row_values = [quote["open"], quote["close"], quote["high"], quote["low"], quote["volume"]]
            new_row = pd.Series(data=row_values, index=column_names)
            self.data_frames[quote["symbol"]].loc[row_id] = new_row

    def do_indicator_exist(self, column_names: List[str]) -> bool:
        """Checks if the specified indicator columns exist in the data frames.

        Args:
            column_names: A list of column names to check.

        Returns:
            True if all columns exist, otherwise False.
        """
        for symbol in self.symbols:
            df = self.data_frames.get(symbol)
            if df is None or not all(col in df.columns for col in column_names):
                return False
        return True

    def _check_signals(
        self,
        indicators: Dict,
        indicators_comp_key: List[str],
        indicators_key: List[str],
    ) -> Union[pd.DataFrame, None]:
        """Checks for buy/sell signals based on indicator conditions.

        Args:
            indicators: A dictionary containing indicator configurations.
            indicators_comp_key: A list of comparative indicator keys.
            indicators_key: A list of regular indicator keys.

        Returns:
            A DataFrame containing buy/sell conditions, or None if no signals.
        """
        last_rows = pd.concat(self.data_frames.values()).tail(1)

        conditions = {}
        if self.do_indicator_exist(indicators_key):
            for indicator in indicators_key:
                buy_condition = indicators[indicator]["buy"]
                sell_condition = indicators[indicator]["sell"]
                buy_operator = indicators[indicator]["buy_operator"]
                sell_operator = indicators[indicator]["sell_operator"]

                buy_condition_met = buy_operator(last_rows[indicator], buy_condition)
                sell_condition_met = sell_operator(last_rows[indicator], sell_condition)

                conditions["buys"] = buy_condition_met.where(lambda x: x).dropna()
                conditions["sells"] = sell_condition_met.where(lambda x: x).dropna()

        return pd.DataFrame(conditions) if conditions else None

    def reset_data(self) -> None:
        """Resets all internal data structures, clearing all stored information."""
        self.data_frames = {
            symbol: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "bid", "ask", "quote"])
            for symbol in self.symbols
        }


