import pandas as pd
from typing import List, Dict, Union
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import RollingGroupby

class DataManager:
    def __init__(self, config, data: List[Dict], symbols: List[str]) -> None:
        """Initializes the Stock Data Manager.

        Args:
            config: Configuration object (not used in this snippet).
            data (List[Dict]): The data to convert to a DataFrame. Normally,
                this is returned from the historical prices endpoint.
            symbols (List[str]): List of symbols to track.
        """
        self._data = data
        self._frame: pd.DataFrame = self.create_frame()
        self._symbol_groups = None
        self._symbol_rolling_groups = None
        self.symbols = {symbol: pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close']) for symbol in symbols}
        self.max_data_points = 1000

    def update(self, symbol: str, tick: Dict[str, Union[int, float]]) -> None:
        """Updates the DataFrame for the specified symbol with new tick data.

        Args:
            symbol (str): The stock symbol to update.
            tick (Dict[str, Union[int, float]]): New tick data containing
                'epoch' for timestamp and 'quote' for price.
        """
        new_data = pd.DataFrame({
            'timestamp': [tick['epoch']],
            'open': [tick['quote']],
            'high': [tick['quote']],
            'low': [tick['quote']],
            'close': [tick['quote']]
        })

        # Update the DataFrame for the given symbol
        if symbol in self.symbols:
            self.symbols[symbol] = pd.concat([self.symbols[symbol], new_data]).tail(self.max_data_points)
        else:
            print(f"Symbol {symbol} not found in managed symbols.")

    def get_close_prices(self, symbol: str) -> List[float]:
        return self.data[symbol]['close'].tolist()

    def get_ohlc_data(self, symbol: str) -> pd.DataFrame:
        return self.data[symbol]

    @property
    def frame(self) -> pd.DataFrame:
        """The frame object.

        Returns:
        ----
        pd.DataFrame -- A pandas data frame with the price data.
        """
        return self._frame

    @property
    def symbol_groups(self) -> DataFrameGroupBy:
        """Returns the Groups in the StockFrame.

        Returns:
        ----
        {DataFrameGroupBy} -- A `pandas.core.groupby.GroupBy` object with each symbol.
        """
        # Group by Symbol.
        self._symbol_groups = self._frame.groupby(
            by='symbol',
            as_index=False,
            sort=True
        )
        return self._symbol_groups

    def symbol_rolling_groups(self, size: int) -> RollingGroupby:
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
            self.symbol_groups

        self._symbol_rolling_groups = self._symbol_groups.rolling(size)
        return self._symbol_rolling_groups

    def create_frame(self) -> pd.DataFrame:
        """Creates a new data frame with the data passed through.

        Returns:
        ----
        {pd.DataFrame} -- A pandas dataframe.
        """
        price_df = pd.DataFrame(data=self._data)
        price_df = self._parse_datetime_column(price_df=price_df)
        price_df = self._set_multi_index(price_df=price_df)

        return price_df

    def _parse_datetime_column(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Parses the datetime column passed through.

        Arguments:
        ----
        price_df {pd.DataFrame} -- The price data frame with a
            datetime column.

        Returns:
        ----
        {pd.DataFrame} -- A pandas dataframe.
        """
        price_df['datetime'] = pd.to_datetime(
            price_df['datetime'],
            unit='ms', 
            origin='unix'
        )
        return price_df

    def _set_multi_index(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Converts the dataframe to a multi-index data frame.

        Arguments:
        ----
        price_df {pd.DataFrame} -- The price data frame.

        Returns:
        ----
        pd.DataFrame -- A pandas dataframe.
        """
        price_df = price_df.set_index(keys=['symbol', 'datetime'])
        return price_df

    def add_rows(self, data: Dict) -> None:
        """Adds a new row to our StockFrame.

        Arguments:
        ----
        data {Dict} -- A list of quotes.
        """
        column_names = ['open', 'close', 'high', 'low', 'volume']

        for quote in data:
            # Parse the Timestamp.
            time_stamp = pd.to_datetime(
                quote['datetime'],
                unit='ms',
                origin='unix'
            )

            # Define the Index Tuple.
            row_id = (quote['symbol'], time_stamp)

            # Define the values.
            row_values = [
                quote['open'],
                quote['close'],
                quote['high'],
                quote['low'],
                quote['volume']
            ]

            # Create a new row.
            new_row = pd.Series(data=row_values, index=column_names)

            # Add the row and sort the frame index.
            self._frame.loc[row_id] = new_row.values
            self._frame.sort_index(inplace=True)

    def do_indicator_exist(self, column_names: List[str]) -> bool:
        """Checks to see if the indicator columns specified exist.

        Arguments:
        ----
        column_names {List[str]} -- A list of column names that will be checked.

        Raises:
        ----
        KeyError: If a column is not found in the StockFrame, a KeyError will be raised.

        Returns:
        ----
        bool -- `True` if all the columns exist.
        """
        missing_columns = set(column_names).difference(self._frame.columns)
        if not missing_columns:
            return True
        else:
            raise KeyError(f"The following indicator columns are missing from the StockFrame: {missing_columns}")

    def _check_signals(self, indicators: Dict, indicators_comp_key: List[str], indicators_key: List[str]) -> Union[pd.DataFrame, None]:
        """Returns the last row of the StockFrame if conditions are met.

        Arguments:
        ----
        indicators {dict} -- A dictionary containing all the indicators to be checked.
        indicators_comp_key {List[str]} -- Indicators where we compare one indicator to another.
        indicators_key {List[str]} -- Indicators where we compare one indicator to a numerical value.

        Returns:
        ----
        {Union[pd.DataFrame, None]} -- If signals are generated, a pandas DataFrame is returned; else None.
        """
        # Grab the last rows.
        last_rows = self.symbol_groups.tail(1)

        # Define a list of conditions.
        conditions = {}

        # Check if all columns exist.
        if self.do_indicator_exist(column_names=indicators_key):
            for indicator in indicators_key:
                column = last_rows[indicator]

                # Grab Buy & Sell Conditions.
                buy_condition_target = indicators[indicator]['buy']
                sell_condition_target = indicators[indicator]['sell']

                buy_condition_operator = indicators[indicator]['buy_operator']
                sell_condition_operator = indicators[indicator]['sell_operator']

                condition_1 = buy_condition_operator(column, buy_condition_target)
                condition_2 = sell_condition_operator(column, sell_condition_target)

                conditions['buys'] = condition_1.where(lambda x: x).dropna()
                conditions['sells'] = condition_2.where(lambda x: x).dropna()

        # Store the indicators in a list.
        check_indicators = []
        
        # Split names to check if indicators exist.
        for indicator in indicators_comp_key:
            check_indicators.extend(indicator.split('_comp_'))

        if self.do_indicator_exist(column_names=check_indicators):
            for indicator in indicators_comp_key:
                parts = indicator.split('_comp_')
                indicator_1 = last_rows[parts[0]]
                indicator_2 = last_rows[parts[1]]

                # If buy operator exists.
                if indicators[indicator].get('buy_operator'):
                    buy_condition_operator = indicators[indicator]['buy_operator']
                    condition_1 = buy_condition_operator(indicator_1, indicator_2)
                    conditions['buys'] = condition_1.where(lambda x: x).dropna()

                # If sell operator exists.
                if indicators[indicator].get('sell_operator'):
                    sell_condition_operator = indicators[indicator]['sell_operator']
                    condition_2 = sell_condition_operator(indicator_1, indicator_2)
                    conditions['sells'] = condition_2.where(lambda x: x).dropna()

        return pd.DataFrame(conditions) if conditions else None

    def grab_n_bars_ago(self, symbol: str, n: int) -> pd.Series:
        """Grabs the trading bar `n` bars ago.

        Arguments:
        ----
        symbol : str -- The symbol to grab the bar for.
        n : int -- The number of bars to look back.

        Returns:
        ----
        pd.Series -- A candle bar represented as a pandas series object.
        """
        bars_filtered = self._frame.filter(like=symbol, axis=0)
        return bars_filtered.iloc[-n]

    def calculate_indicators(self, symbol: str, indicators: Dict[str, Dict]) -> None:
        """Calculates and adds specified indicators to the stock data frame.

        Arguments:
        ----
        symbol : str -- The symbol for which to calculate indicators.
        indicators : Dict[str, Dict] -- A dictionary of indicators and their parameters.
        """
        df = self.get_ohlc_data(symbol)
        for indicator, params in indicators.items():
            if indicator == 'SMA':
                window = params.get('window', 14)
                df[f'{indicator}_{window}'] = df['close'].rolling(window=window).mean()
            elif indicator == 'EMA':
                span = params.get('span', 14)
                df[f'{indicator}_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
            # Add more indicators as needed
        self.data[symbol] = df

    def reset_data(self) -> None:
        """Resets the internal data structures, clearing all stored information."""
        self.data = {symbol: pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close']) for symbol in self.symbols}
        self._frame = self.create_frame()
        self._symbol_groups = None
        self._symbol_rolling_groups = None

    def get_all_symbols_data(self) -> Dict[str, pd.DataFrame]:
        """Retrieves OHLC data for all tracked symbols.

        Returns:
        ----
        Dict[str, pd.DataFrame] -- A dictionary of DataFrames indexed by symbol.
        """
        return self.data
