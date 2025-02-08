import pandas as pd
from config import Config
from configparser import ConfigParser
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from typing import List, Dict, Union
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import RollingGroupby


class DataManager:
    def __init__(self, config: Config, max_data_points: int = 100000) -> None:
        """Initializes the Stock Data Manager.

        Args:
            config: Configuration object.
            max_data_points (int): Maximum number of data points to store for each symbol.
        """
        self._data = {}  # Stores data for each symbol
        self._ws_connections = {}  # WebSocket connections for each symbol
        self._frame: pd.DataFrame = pd.DataFrame()  # Create the DataFrame
        self._symbol_groups: DataFrameGroupBy = None
        self._symbol_rolling_groups: RollingGroupby = None
        
        # Initialize symbols with empty DataFrames
        self.symbols = {
            symbol: pd.DataFrame(
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "bid",
                    "ask",
                    "quote",
                ]
            )
            for symbol in config.SYMBOLS  # Use symbols from config
        }

        self.max_data_points = max_data_points  # Use the provided max_data_points

    def create_subs_cb(self, symbol: str) -> callable:
        """Generates a callback function for each symbol to handle real-time updates."""
        def callback(data):
            if "tick" in data:
                tick = data
                self.update(symbol, tick)
            else:
                print(f"No tick data for symbol {symbol}")

        return callback

    async def subscribe(self, symbol: str, ws_url: str) -> None:
        """Subscribe to a WebSocket data stream for a given symbol."""
        async with websockets.connect(ws_url) as ws:
            self._ws_connections[symbol] = ws
            subscribe_message = {
                "type": "subscribe",
                "symbol": symbol
            }

            # Send the subscription request
            await ws.send(str(subscribe_message))

            while True:
                response = await ws.recv()
                if response:
                    # Assuming the response is in JSON format
                    data = eval(response)  # For simplicity, converting to dict
                    callback = self.create_subs_cb(symbol)
                    callback(data)

    async def start_subscriptions(self, ws_url: str) -> None:
        """Start WebSocket subscriptions for all symbols."""
        if not self.symbols:
            print("No symbols to subscribe to.")
            return

        tasks = []
        for symbol in self.symbols:
            tasks.append(self.subscribe(symbol, ws_url))

        if tasks:
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                print(f"Error during subscription: {e}")


    async def stop_subscriptions(self) -> None:
        """Stop all WebSocket subscriptions."""
        for ws in self._ws_connections.values():
            await ws.close()
        self._ws_connections.clear()

    async def update(self, symbol: str, tick: Dict[str, Union[int, float]]) -> None:
        """Update the DataFrame for the specified symbol with new tick data."""
        try:
            epoch = tick["tick"]["epoch"]
            ask = tick["tick"]["ask"]
            bid = tick["tick"]["bid"]
            quote = tick["tick"]["quote"]

            new_data = pd.DataFrame(
                {
                    "timestamp": [epoch],
                    "open": [bid],
                    "high": [ask],
                    "low": [bid],
                    "close": [bid],
                    "bid": [bid],
                    "ask": [ask],
                    "quote": [quote],
                }
            )

            if symbol in self.symbols:
                if self.symbols[symbol].empty:
                    self.symbols[symbol] = new_data
                else:
                    last_row = self.symbols[symbol].iloc[-1]
                    if last_row["timestamp"] == epoch:
                        last_row["close"] = bid
                        last_row["high"] = max(last_row["high"], ask)
                        last_row["low"] = min(last_row["low"], bid)
                        self.symbols[symbol].iloc[-1] = last_row
                    else:
                        self.symbols[symbol] = pd.concat([self.symbols[symbol], new_data]).tail(self._max_data_points)
            else:
                print(f"Symbol {symbol} not found in managed symbols.")
        except KeyError as e:
            print(f"KeyError: Missing expected key in tick data - {e}")
        except Exception as e:
            print(f"Error updating symbol {symbol}: {e}")

    def get_close_prices(self, symbol: str) -> List[float]:
        """Retrieves the close prices for the specified symbol."""
        close_prices = []
        for entry in self._data:
            if entry.get("symbol") == symbol:  # Assuming each entry has a 'symbol' key
                close_prices.append(entry["close"])
        return close_prices

    def get_ohlc_data(self, symbol: str) -> pd.DataFrame:
        """Retrieves the OHLC data for the specified symbol."""
        if symbol in self.symbols:
            return self.symbols[symbol]
        else:
            raise ValueError(f"Symbol '{symbol}' not found in managed symbols.")

    @property
    def symbol_groups(self) -> DataFrameGroupBy:
        """Returns the Groups in the StockFrame.

        Returns:
        ----
        {DataFrameGroupBy} -- A `pandas.core.groupby.GroupBy` object with each symbol.
        """
        # Ensure the frame is populated
        if self._frame is None or self._frame.empty:
            return pd.DataFrame().groupby([])

        # Group by Symbol.
        self._symbol_groups = self._frame.groupby(
            by="symbol", as_index=False, sort=True
        )
        return self._symbol_groups

    async def symbol_rolling_groups(self, size: int) -> RollingGroupby:
        """Grabs the windows for each group.

        Arguments:
        ----
        size {int} -- The size of the window.

        Returns:
        ----
        {RollingGroupby} -- A `pandas.core.window.RollingGroupby` object.
        """
        # Ensure symbol groups exist.
        if self._symbol_groups is None:
            self.symbol_groups  # This will initialize _symbol_groups if it is None

        self._symbol_rolling_groups = self._symbol_groups.rolling(size)
        return self._symbol_rolling_groups

    def _set_multi_index(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Sets a multi-index for the DataFrame.

        Arguments:
        ----
        price_df {pd.DataFrame} -- The price DataFrame.

        Returns:
        ----
        {pd.DataFrame} -- A pandas DataFrame with a multi-index.
        """
        # Example implementation; adjust as needed
        price_df.set_index(["symbol", "datetime"], inplace=True)
        return price_df

    def add_rows(self, data: Dict) -> None:
        """Adds a new row to our StockFrame.

        Arguments:
        ----
        data {Dict} -- A list of quotes.
        """
        column_names = ["open", "close", "high", "low", "volume"]

        for quote in data:
            # Parse the Timestamp.
            time_stamp = pd.to_datetime(quote["datetime"], unit="ms", origin="unix")

            # Define the Index Tuple.
            row_id = (quote["symbol"], time_stamp)

            # Define the values.
            row_values = [
                quote["open"],
                quote["close"],
                quote["high"],
                quote["low"],
                quote["volume"],
            ]

            # Create a new row.
            new_row = pd.Series(data=row_values, index=column_names)

            # Add the row and sort the frame index.
            self._frame.loc[row_id] = new_row.values
            self._frame.sort_index(inplace=True)



    def do_indicator_exist(self, column_names: List[str]) -> bool:
        """Checks to see if the indicator columns specified exist."""
        missing_columns = set(column_names).difference(self.symbols.columns)
        if not missing_columns:
            return True
        else:
            raise KeyError(
                f"The following indicator columns are missing from the StockFrame: {missing_columns}"
            )

    def _check_signals(
        self,
        indicators: Dict,
        indicators_comp_key: List[str],
        indicators_key: List[str],
    ) -> Union[pd.DataFrame, None]:
        """Returns the last row of the StockFrame if conditions are met."""
        last_rows = self.symbols.tail(1)
        conditions = {}

        if self.do_indicator_exist(column_names=indicators_key):
            for indicator in indicators_key:
                column = last_rows[indicator]
                buy_condition_target = indicators[indicator]["buy"]
                sell_condition_target = indicators[indicator]["sell"]
                buy_condition_operator = indicators[indicator]["buy_operator"]
                sell_condition_operator = indicators[indicator]["sell_operator"]

                condition_1 = buy_condition_operator(column, buy_condition_target)
                condition_2 = sell_condition_operator(column, sell_condition_target)

                conditions["buys"] = condition_1.where(lambda x: x).dropna()
                conditions["sells"] = condition_2.where(lambda x: x).dropna()

        check_indicators = []
        for indicator in indicators_comp_key:
            check_indicators.extend(indicator.split("_comp_"))

        if self.do_indicator_exist(column_names=check_indicators):
            for indicator in indicators_comp_key:
                parts = indicator.split("_comp_")
                indicator_1 = last_rows[parts[0]]
                indicator_2 = last_rows[parts[1]]

                if indicators[indicator].get("buy_operator"):
                    buy_condition_operator = indicators[indicator]["buy_operator"]
                    condition_1 = buy_condition_operator(indicator_1, indicator_2)
                    conditions["buys"] = condition_1.where(lambda x: x).dropna()

                if indicators[indicator].get("sell_operator"):
                    sell_condition_operator = indicators[indicator]["sell_operator"]
                    condition_2 = sell_condition_operator(indicator_1, indicator_2)
                    conditions["sells"] = condition_2.where(lambda x: x).dropna()

        return pd.DataFrame(conditions) if conditions else None

    def grab_n_bars_ago(self, symbol: str, n: int) -> pd.Series:
        """Grabs the trading bar n bars ago."""
        bars_filtered = self.symbols.filter(like=symbol, axis=0)
        return bars_filtered.iloc[-n]

    def calculate_indicators(self, symbol: str, indicators: Dict[str, Dict]) -> None:
        """Calculates and adds specified indicators to the stock data frame."""
        df = self.get_data(symbol)
        for indicator, params in indicators.items():
            if indicator == "SMA":
                window = params.get("window", 14)
                df[f"{indicator}_{window}"] = df["close"].rolling(window=window).mean()
            elif indicator == "EMA":
                span = params.get("span", 14)
                df[f"{indicator}_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
            # Add more indicators as needed
        self.symbols[symbol] = df

    def reset_data(self) -> None:
        """Resets the internal data structures, clearing all stored information."""
        self.symbols = {
            symbol: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "bid", "ask", "quote"])
            for symbol in self.symbols
        }
        self._data.clear()

