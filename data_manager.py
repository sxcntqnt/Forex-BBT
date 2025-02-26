import pandas as pd
from datetime import datetime
import asyncio
from config import Config
from deriv_api import DerivAPI
from typing import List, Dict, Union, Optional
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import RollingGroupby


from rx import Observable


class DataManager:
    def __init__(
        self, config: Config, api, logger, max_data_points: int = 100000
    ) -> None:
        """Initializes the Stock Data Manager with real-time data from Deriv API.

        Args:
            config: Configuration object with settings like symbols.
            api: Connection object for the Deriv API.
            logger: Logger object for logging.
            max_data_points: Max number of data points to store for each symbol.
        """
        self.config = config
        self.api = api
        self.symbols = config.SYMBOLS
        self.max_data_points = max_data_points
        self.data_frames = {
            symbol: pd.DataFrame(
                columns=["epoch", "bid", "ask", "quote", "id", "pip_size"]
            )
            for symbol in self.symbols
        }
        self.logger = logger
        self.subscriptions = {}  # Store RxPy subscriptions
        self.last_data = {}
        self._frame = pd.DataFrame()

    @property
    def frame(self):
        return self._frame

    async def subscribe_to_ticks(self, symbol: str) -> None:
        """Subscribe to tick stream for a given symbol."""
        if not isinstance(symbol, str) or not symbol:
            self.logger.error(f"Invalid symbol: {symbol}")
            return

        if symbol in self.subscriptions:  # Check if already subscribed
            self.logger.info(f"Already subscribed to {symbol}. Ingesting last data.")
            # Ingest the last data for the symbol
            await self.update(symbol, self.last_data.get(symbol, {}))
            return

        try:
            # Subscribe to the tick stream
            tick_stream = await self.api.subscribe({"ticks": symbol, "subscribe": 1})
            callback = self.create_subs_cb(symbol)
            subscription = tick_stream.subscribe(callback)  # Store the RxPy subscription
            self.subscriptions[symbol] = subscription  # Store the subscription object
            self.logger.info(f"Subscribed to {symbol} tick stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")

    def create_subs_cb(self, symbol: str):
        """Generates a callback function for real-time data updates.

        Args:
            symbol: The symbol to track.

        Returns:
            Callback function to handle real-time tick data.
        """
        def cb(data: Any):
            self.logger.debug(f"Received data for symbol {symbol}: {data}")
            self.last_data[symbol] = data  # Store the last received data
            
            try:
                asyncio.get_event_loop().create_task(self.update(symbol, data))
            except Exception as e:
                self.logger.error(f"Error creating task for symbol {symbol}: {e}")


        return cb
    async def update(self, symbol: str, tick: Dict[str, Union[int, float]]) -> None:
        """Updates the DataFrame for the specified symbol with new tick data.

        Args:
            symbol: The symbol being updated.
            tick: The tick data (a dictionary containing price data).
        """
        try:
            if "tick" not in tick:
                self.logger.error(
                    f"Invalid tick data for symbol {symbol}: 'tick' key missing."
                )
                return

            tick_data = tick["tick"]
            epoch = tick_data.get("epoch")
            bid = tick_data.get("bid")
            ask = tick_data.get("ask")
            quote = tick_data.get("quote")
            tick_id = tick_data.get("id")
            pip_size = tick_data.get("pip_size")

            if None in (epoch, ask, bid, quote):
                self.logger.error(f"Incomplete tick data for symbol {symbol}: {tick}")
                return

            new_data = pd.DataFrame(
                {
                    "epoch": [epoch],
                    "bid": [bid],
                    "ask": [ask],
                    "quote": [quote],
                    "id": [tick_id],
                    "pip_size": [pip_size],
                }
            )

            if new_data.empty:
                self.logger.warning(f"No valid data to concatenate for {symbol}")
                return

            # Concatenate new data with existing data
            self.data_frames[symbol] = pd.concat(
                [self.data_frames.get(symbol).dropna(), new_data]
            ).tail(self.max_data_points)  # Keep only the last max_data_points

        except KeyError as e:
            self.logger.error(f"KeyError: Missing expected key in tick data - {e}")
        except Exception as e:
            self.logger.error(f"Error updating symbol {symbol}: {e}")


    async def start_subscriptions(self) -> None:
        """Starts WebSocket subscriptions for all symbols."""
        if not self.symbols:
            self.logger.warning("No symbols to subscribe to.")
            return
    
        tasks = []
        for symbol in self.symbols:
            tasks.append(self.subscribe_to_ticks(symbol))
    
        try:
            await asyncio.gather(*tasks)
            self.logger.info("All subscriptions started successfully.")
        except Exception as e:
            self.logger.error(f"Error during subscription: {e}")

    async def stop_subscriptions(self) -> None:
        """Properly dispose of all subscription channels."""
        for symbol, sub in self.subscriptions.items():
            try:
                sub.dispose()  # Call dispose on the subscription object
                self.logger.info(f"Unsubscribed from {symbol}")
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {symbol}: {e}")
        self.subscriptions.clear()
        await self.api.clear()  # Clear API state if supported

    def get_close_prices(self, symbol: str) -> List[float]:
        """Retrieves the close prices for the specified symbol.

        Args:
            symbol: The symbol for which to retrieve close prices.

        Returns:
            A list of close prices.
        """
        df = self.data_frames.get(symbol)
        if df is None or "close" not in df.columns:
            self.logger.debug(f"No close prices available for {symbol}")
            return []
        return df["close"].tolist()

    async def grab_historical_data(
        self,
        start_timestamp: int,
        end_timestamp: int,
        symbol: str,
        bar_type: str = "minute",
    ) -> pd.DataFrame:
        """Fetches historical data for a single symbol.

        Args:
            start_timestamp: Start time in seconds or datetime object.
            end_timestamp: End time in seconds or datetime object.
            symbol: Trading symbol to fetch data for.
            bar_type: Type of bars ('minute', '5minute', 'hour', etc.).

        Returns:
            DataFrame with historical OHLC data.
        """
        if isinstance(start_timestamp, datetime):
            start_timestamp = int(start_timestamp.timestamp())
        if isinstance(end_timestamp, datetime):
            end_timestamp = int(end_timestamp.timestamp())

        granularity_map = {"minute": 60, "5minute": 300, "hour": 3600}
        if bar_type not in granularity_map:
            raise ValueError(
                f"Unsupported bar_type: {bar_type}. Supported: {list(granularity_map.keys())}"
            )

        args = {
            "ticks_history": symbol,
            "start": start_timestamp,
            "end": end_timestamp,
            "granularity": granularity_map[bar_type],
            "count": 5000,
            "adjust_start_time": 1,
        }

        try:
            response = await self.api.ticks_history(args)
            if "error" in response:
                raise Exception(f"API error: {response['error']['message']}")
            if "candles" not in response or not response["candles"]:
                self.logger.warning(f"No historical data available for {symbol}.")
                return (
                    pd.DataFrame()
                )  # Return an empty DataFrame instead of raising an error

            historical_df = pd.DataFrame(
                [
                    {
                        "timestamp": candle["epoch"],
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                    }
                    for candle in response["candles"]
                ]
            )

            if symbol in self.data_frames:
                self.data_frames[symbol] = pd.concat(
                    [self.data_frames[symbol], historical_df]
                ).tail(self.max_data_points)
            return historical_df

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    def add_rows(self, data: List[Dict]) -> None:
        """Adds new rows to the data_frames.

        Args:
            data: List of dictionaries with 'symbol', 'timestamp', 'bid', 'ask', 'quote'.
        """
        for row in data:
            symbol = row["symbol"]
            if symbol not in self.data_frames:
                self.logger.warning(f"Symbol {symbol} not in managed symbols.")
                continue

            new_data = pd.DataFrame(
                [
                    {
                        "epoch": row["epoch"],
                        "bid": row.get("bid"),
                        "ask": row.get("ask"),
                        "quote": row.get("quote"),
                        "id": row.get("id"),
                        "pip_size": row.get("pip_size"),
                    }
                ]
            )

            if new_data.empty:
                self.logger.warning(f"No valid data to add for {symbol}")
                continue

            self.data_frames[symbol] = pd.concat(
                [self.data_frames[symbol], new_data]
            ).tail(self.max_data_points)

    def get_ohlc_data(self, symbol: str) -> pd.DataFrame:
        """Retrieves the OHLC data for the specified symbol.

        Args:
            symbol: The symbol for which to retrieve OHLC data.

        Returns:
            A DataFrame containing OHLC data for the symbol.
        """
        if symbol in self.data_frames:
            return self.data_frames[symbol]
        raise ValueError(f"Symbol '{symbol}' not found.")

    @property
    def _symbol_groups(self) -> DataFrameGroupBy:
        """Returns a GroupBy object of symbols in the data frames.

        Returns:
            A pandas GroupBy object grouped by symbol.
        """
        if not self.data_frames:
            return pd.DataFrame().groupby([])

        return pd.concat(self.data_frames.values()).groupby("symbol", as_index=False)

    async def symbol_rolling_groups(self, size: int) -> RollingGroupby:
        """Grabs rolling windows of data for each symbol.

        Args:
            size: The size of the rolling window.

        Returns:
            A RollingGroupby object containing rolling windows of data.
        """
        df = pd.concat(self.data_frames.values())
        return df.groupby("symbol").rolling(size)

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
        window_size: int = 1,
    ) -> Optional[pd.DataFrame]:
        """Checks for buy/sell signals based on indicator conditions over a window.

        Args:
            indicators: Dictionary of indicator configurations.
            indicators_comp_key: List of comparative indicator keys.
            indicators_key: List of regular indicator keys.
            window_size: Number of rows to consider for signals.

        Returns:
            DataFrame with buy/sell conditions or None if no signals.
        """
        all_data = pd.concat(self.data_frames.values())
        if all_data.empty:
            return None

        last_window = all_data.tail(window_size)
        conditions = {}

        for indicator in indicators_key:
            if indicator not in last_window.columns:
                continue
            buy_condition = indicators[indicator]["buy"]
            sell_condition = indicators[indicator]["sell"]
            buy_operator = indicators[indicator]["buy_operator"]
            sell_operator = indicators[indicator]["sell_operator"]

            buy_met = buy_operator(last_window[indicator], buy_condition)
            sell_met = sell_operator(last_window[indicator], sell_condition)

            conditions["buys"] = buy_met.where(lambda x: x).dropna()
            conditions["sells"] = sell_met.where(lambda x: x).dropna()

        return pd.DataFrame(conditions) if conditions else None

    def reset_data(self) -> None:
        """Resets all internal data structures, clearing all stored information."""
        self.data_frames = {
            symbol: pd.DataFrame(
                columns=["epoch", "bid", "ask", "quote", "id", "pip_size"]
            )
            for symbol in self.symbols
        }
        self.last_data.clear()
