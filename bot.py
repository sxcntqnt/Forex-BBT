import asyncio
import logging
import json , re
import time as time_true
import pathlib
import pandas as pd

from datetime import datetime
from datetime import timezone
from datetime import timedelta

from typing import List
from typing import Dict
from typing import Union

from deriv_api import DerivAPI
from rx import Observable
from config import Config
from configparser import ConfigParser
from strategy import StrategyManager
from risk_manager import RiskManager
from monitor import Monitor
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from performance_monitor import PerformanceMonitor
from backtester import Backtester


# We are going to be doing some timestamp conversions.


class ForexBot:
    def __init__(
        self,
        config,
        connection,
        paper_trading: bool = True,
        credentials_path: str = None,
        trading_account: str = None,
    ) -> None:
        """Initializes the ForexBot."""
        # Set the attributes
        self.trading_account = trading_account
        self.Backtesters = {}
        self.historical_prices = {}
        self.stock_frame: DataManager = None
        self.paper_trading = paper_trading
        self._bar_size = None
        self._bar_type = None
        self.config = Config()
        self.api = DerivAPI(connection=connection)
        self.args = {}
       
        # Initialize DataManager with the appropriate symbols
        self.data_manager: DataManager = DataManager(config)

        self.strategy_manager = StrategyManager(self.data_manager, config)
        self.risk_manager = RiskManager(config)
        self.backtester = Backtester(config, self.api, self.data_manager, self.strategy_manager)
        self.monitor = Monitor(config, self.backtester)
        self.portfolio_manager = PortfolioManager(config)
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.subscriptions = {}
        self.start_date = datetime.now() - timedelta(days=config.HISTORICAL_DAYS)
        self.end_date = datetime.now()

    async def get_server_time(api):
        response = await api.time({"time": 1})  # Call the async method correctly
        return response.get("time")  # Extract the epoch time

        server_time = await get_server_time(api)
        print(f"Server Epoch Time: {server_time}")

    async def run(self):
        self.running = True
        await self.api.authorize({"authorize": self.config.API_TOKEN})
        self.logger.info("Authorized successfully.")

        # Subscribe to active symbols
        for symbol in self.config.SYMBOLS:
            await self.subscribe_to_symbol(symbol)

        while self.running:
            try:
                await self.check_trades()
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    async def subscribe_to_symbol(self, symbol):
        try:
            tick_stream: Observable = await self.api.subscribe(
                {"ticks": symbol, "subscribe": 1}
            )
            self.subscriptions[symbol] = tick_stream
            tick_stream.subscribe(self.create_tick_callback(symbol))
            self.logger.info(f"Subscribed to {symbol} tick stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")

    def create_tick_callback(self, symbol):
        count = 0

        def callback(data):
            nonlocal count
            count += 1
            self.data_manager.update(
                symbol, data
            )  # Update the DataManager with the tick data
            self.logger.info(f"Received tick for {symbol}: {data}. Count: {count}")

        return callback

    async def check_trades(self):
        for symbol in self.config.SYMBOLS:
            if self.strategy_manager.should_enter_trade(symbol, self.data_manager):
                await self.enter_trade(data, symbol)

        await self.monitor.check_open_positions(self.api)
        await self.performance_monitor.update(self.portfolio_manager)

    async def enter_trade(self, symbol):
        if not self.risk_manager.can_enter_trade(symbol):
            self.logger.warning(
                f"Cannot enter trade for {symbol}: risk management rules not satisfied."
            )
            return

        try:
            position_size = self.risk_manager.calculate_position_size(symbol)
            contract = await self.api.buy(
                {
                    "contract_type": "CALL",
                    "amount": position_size,
                    "symbol": symbol,
                    "duration": 5,
                    "duration_unit": "m",
                }
            )
            self.monitor.add_position(contract)
            self.portfolio_manager.add_trade(symbol, contract)
            self.logger.info(f"Entered trade: {contract}")
        except Exception as e:
            self.logger.error(f"Error entering trade for {symbol}: {e}")

    async def unsubscribe(self):
        for symbol, subscription in self.subscriptions.items():
            try:
                await self.api.forget(subscription["subscription"]["id"])
                self.logger.info(f"Unsubscribed from {symbol} tick stream.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {symbol}: {e}")
        self.subscriptions.clear()

    async def initialize_data_manager(self, config, api):
        """Initialize the DataManager with historical data."""
        start_date = datetime.today()
        end_date = start_date - timedelta(days=config.HISTORICAL_DAYS)
        historical_data = []

        # Assuming config.TIMEFRAME is the bar_type and a default bar_size is defined
        bar_size = (
            config.BAR_SIZE if hasattr(config, "BAR_SIZE") else 1
        )  # Default to 1 if not specified
        bar_type = config.TIMEFRAME  # Assuming TIMEFRAME is the bar_type

        for symbol in config.SYMBOLS:
            # Call the grab_historical_prices method with the correct arguments
            data = await self.grab_historical_prices(
                start_date,
                end_date,
                bar_size,
                bar_type,
                [symbol],  # Pass the symbol as a list
            )
            historical_data.extend(data)

        return DataManager(config, historical_data, config.SYMBOLS)

    def stop(self):
        self.running = False
        asyncio.create_task(
            self.unsubscribe()
        )  # Ensure unsubscription happens asynchronously
        self.logger.info("Stopping the bot...")

    async def run_backtest(self):
        results = await self.backtester.run()
        self.logger.info(f"Backtest results: {results}")
        return results

    @property
    def pre_market_open(self) -> bool:
        """Checks if pre-market is open.

        Uses the datetime module to create US Pre-Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> pre_market_open_flag = trading_robot.pre_market_open
            >>> pre_market_open_flag
            True

        Returns:
        ----
        bool -- True if pre-market is open, False otherwise.

        """

        pre_market_start_time = (
            datetime.utcnow().replace(hour=8, minute=00, second=00).timestamp()
        )

        market_start_time = (
            datetime.utcnow().replace(hour=13, minute=30, second=00).timestamp()
        )

        right_now = datetime.utcnow().timestamp()

        if market_start_time >= right_now >= pre_market_start_time:
            return True
        else:
            return False

    @property
    def post_market_open(self):
        """Checks if post-market is open.

        Uses the datetime module to create US Post-Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> post_market_open_flag = trading_robot.post_market_open
            >>> post_market_open_flag
            True

        Returns:
        ----
        bool -- True if post-market is open, False otherwise.

        """

        post_market_end_time = (
            datetime.utcnow().replace(hour=00, minute=00, second=00).timestamp()
        )

        market_end_time = (
            datetime.utcnow().replace(hour=20, minute=00, second=00).timestamp()
        )

        right_now = datetime.utcnow().timestamp()

        if post_market_end_time >= right_now >= market_end_time:
            return True
        else:
            return False

    @property
    def regular_market_open(self):
        """Checks if regular market is open.

        Uses the datetime module to create US Regular Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> market_open_flag = trading_robot.market_open
            >>> market_open_flag
            True

        Returns:
        ----
        bool -- True if post-market is open, False otherwise.

        """

        market_start_time = (
            datetime.utcnow().replace(hour=13, minute=30, second=00).timestamp()
        )

        market_end_time = (
            datetime.utcnow().replace(hour=20, minute=00, second=00).timestamp()
        )

        right_now = datetime.utcnow().timestamp()

        if market_end_time >= right_now >= market_start_time:
            return True
        else:
            return False

    def create_portfolio(self) -> PortfolioManager:
        """Create a new portfolio.

        Creates a Portfolio Object to help store and organize positions
        as they are added and removed during trading.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> portfolio = trading_robot.create_portfolio()
            >>> portfolio
            <pyrobot.portfolio.Portfolio object at 0x0392BF88>

        Returns:
        ----
        Portfolio -- A pyrobot.Portfolio object with no positions.
        """

        # Initalize the portfolio.
        self.portfolio = Portfolio(account_number=self.trading_account)

        # Assign the Client
        self.portfolio.td_client = self.api

        return self.portfolio

    def create_trade(
        self,
        trade_id: str,
        enter_or_exit: str,
        long_or_short: str,
        order_type: str = "mkt",
        price: float = 0.0,
        stop_limit_price=0.0,
    ) -> Backtester:
        """Initalizes a new instance of a Backtester Object.

        This helps simplify the process of building an order by using pre-built templates that can be
        easily modified to incorporate more complex strategies.

        Arguments:
        ----
        trade_id {str} -- The ID associated with the trade, this can then be used to access the trade during runtime.

        enter_or_exit {str} -- Defines whether this trade will be used to enter or exit a position.
            If used to enter, specify `enter`. If used to exit, speicfy `exit`.

        long_or_short {str} -- Defines whether this trade will be used to go long or short a position.
            If used to go long, specify `long`. If used to go short, speicfy `short`.

        Keyword Arguments:
        ----
        order_type {str} -- Defines the type of order to initalize. Possible values
            are `'mkt', 'lmt', 'stop', 'stop-lmt', 'trailign-stop'` (default: {'mkt'})

        price {float} -- The Price to be associate with the order. If the order type is `stop` or `stop-lmt` then
            it is the stop price, if it is a `lmt` order then it is the limit price, and `mkt` is the market
            price.(default: {0.0})

        stop_limit_price {float} -- Only used if the order is a `stop-lmt` and represents the limit price of
            the `stop-lmt` order. (default: {0.0})

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> new_trade = trading_robot_portfolio.create_trade(
                trade_id='long_1',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt'
            )
            >>> new_trade

            >>> new_market_trade = trading_robot_portfolio.create_trade(
                trade_id='long_2',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt',
                price=12.00
            )
            >>> new_market_trade

            >>> new_stop_trade = trading_robot_portfolio.create_trade(
                trade_id='long_3',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='stop',
                price=2.00
            )
            >>> new_stop_trade

            >>> new_stop_limit_trade = trading_robot_portfolio.create_trade(
                trade_id='long_4',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='stop-lmt',
                price=2.00,
                stop_limit_price=1.90
            )
            >>> new_stop_limit_trade

        Returns:
        ----
        Backtester -- A pyrobot.Backtester object with the specified template.
        """

        # Initalize a new Backtester object.
        trade = Backtester()

        # Create a new trade.
        trade.new_trade(
            trade_id=trade_id,
            order_type=order_type,
            side=long_or_short,
            enter_or_exit=enter_or_exit,
            price=price,
            stop_limit_price=stop_limit_price,
        )

        # Set the Client.
        trade.account = self.trading_account
        trade._td_client = self.api

        self.trades[trade_id] = trade

        return trade

    def delete_trade(self, index: int) -> None:
        """Deletes an exisiting trade from the `trades` collection.

        Arguments:
        ----
        index {int} -- The index of the order.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> new_trade = trading_robot_portfolio.create_trade(
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt'
            )
            >>> trading_robot.delete_trade(index=1)
        """

        if index in self.trades:
            del self.trades[index]

    def grab_current_quotes(self) -> dict:
        """Grabs the current quotes for all positions in the portfolio.

        Makes a call to the TD Ameritrade Get Quotes endpoint with all
        the positions in the portfolio. If only one position exist it will
        return a single dicitionary, otherwise a nested dictionary.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_portfolio.add_position(
            symbol='MSFT',
            asset_type='equity'
            )
            >>> current_quote = trading_robot.grab_current_quotes()
            >>> current_quote
            {
                "MSFT": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "594918104",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": true
                }
            }

            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_portfolio.add_position(
            symbol='MSFT',
            asset_type='equity'
            )
            >>> trading_robot_portfolio.add_position(
            symbol='AAPL',
            asset_type='equity'
            )
            >>> current_quote = trading_robot.grab_current_quotes()
            >>> current_quote

            {
                "MSFT": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "594918104",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": False
                },
                "AAPL": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "037833100",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": False
                }
            }

        Returns:
        ----
        dict -- A dictionary containing all the quotes for each position.

        """

        # First grab all the symbols.
        symbols = self.portfolio.positions.keys()

        # Grab the quotes.
        quotes = self.api.get_quotes(instruments=list(symbols))

        return quotes

    @staticmethod
    def milliseconds_since_epoch(dt):
        """
        Converts a datetime object to milliseconds since epoch.
        
        Args:
            dt (datetime): Datetime object to convert
            
        Returns:
            int: Milliseconds since epoch
        """
        return int(dt.timestamp() * 1000)


    async def grab_historical_prices(
        self,
        start_date: datetime,
        end_date: datetime,
        bar_size: int = 1,
        bar_type: str = "minute",
        symbols: List[str] = None,
    ) -> Dict[str, Union[List[Dict], Dict]]:
        """
        Grabs historical prices for all positions in a portfolio.
        Arguments and functionality explained in the docstring above.
        """
        # Input validation
        if not isinstance(symbols, list) and symbols is not None:
            raise ValueError("Symbols must be a list")

        if not symbols:
            symbols = self.portfolio_manager.positions  # Assuming this is predefined

        if not symbols:
            raise ValueError("No symbols provided for historical data retrieval.")

        if self.api is None:
            raise ValueError("API instance is not initialized!")

        print(f"DEBUG: Processing {len(symbols)} symbols")

        # Convert dates to timestamps
        start_timestamp = self.milliseconds_since_epoch(start_date) // 1000
        end_timestamp = self.milliseconds_since_epoch(end_date) // 1000

        print(f"DEBUG: Timestamp range - start: {start_timestamp}, end: {end_timestamp}")

        if not start_timestamp or not end_timestamp:
            raise ValueError("Invalid timestamps calculated")

        new_prices = []  # For storing all price data
        historical_data = {}  # For symbol-specific historical data

        for symbol in symbols:
            # Clean symbol format (e.g., remove 'frx' or any unwanted parts)
            clean_symbol = symbol.replace('frx', '')

            # Verify symbol format (basic check)
            if not re.match(r'^[a-zA-Z]{2,30}$', clean_symbol):
                print(f"Skipping invalid symbol: {symbol}")
                continue

            self.args = {
                "symbol": clean_symbol,
                "start": int(start_timestamp),
                "end": int(end_timestamp),
                "granularity": 300,
                "style": "candles",
                "count": 5000,
                "adjust_start_time": 1
            }

            print(f"DEBUG: Arguments for ticks_history: {self.args}")

            try:
                # Fetch historical prices using the API
                historical_prices_response = await self.api.ticks_history(self.args)

                if historical_prices_response is None:
                    print(f"No data returned for {symbol}")
                    continue

                if "candles" not in historical_prices_response:
                    print(f"Invalid response structure for {symbol}")
                    continue

                # Store candles for individual symbol
                historical_data[symbol] = {
                    "candles": historical_prices_response["candles"]
                }

                # Process candles and aggregate into new_prices list
                for candle in historical_prices_response["candles"]:
                    new_price_mini_dict = {
                        "symbol": symbol,
                        "open": candle["open"],
                        "close": candle["close"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "volume": candle["volume"],
                        "datetime": candle["epoch"]
                    }
                    new_prices.append(new_price_mini_dict)

            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue

        # Add aggregated prices to the historical data
        historical_data["aggregated"] = new_prices

        # Return the historical data for all symbols and the aggregated data
        return historical_data


    def get_latest_bar(self) -> List[dict]:
        """Returns the latest bar for each symbol in the portfolio.

        Returns:
        ---
        {List[dict]} -- A simplified quote list.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> latest_bars = trading_robot.get_latest_bar()
            >>> latest_bars
        """

        # Grab the info from the last quest.
        bar_size = self._bar_size
        bar_type = self._bar_type

        # Define the start and end date.
        end_date = datetime.today()
        start_date = end_date - timedelta(days=1)
        start = str(milliseconds_since_epoch(dt_object=start_date))
        end = str(milliseconds_since_epoch(dt_object=end_date))

        latest_prices = []

        # Loop through each symbol.
        for symbol in self.portfolio.positions:

            try:

                # Grab the request.
                historical_prices_response = self.api.get_price_history(
                    symbol=symbol,
                    period_type="day",
                    start_date=start,
                    end_date=end,
                    frequency_type=bar_type,
                    frequency=bar_size,
                    extended_hours=True,
                )

            except:

                time_true.sleep(2)

                # Grab the request.
                historical_prices_response = self.api.get_price_history(
                    symbol=symbol,
                    period_type="day",
                    start_date=start,
                    end_date=end,
                    frequency_type=bar_type,
                    frequency=bar_size,
                    extended_hours=True,
                )

            # parse the candles.
            for candle in historical_prices_response["candles"][-1:]:

                new_price_mini_dict = {}
                new_price_mini_dict["symbol"] = symbol
                new_price_mini_dict["open"] = candle["open"]
                new_price_mini_dict["close"] = candle["close"]
                new_price_mini_dict["high"] = candle["high"]
                new_price_mini_dict["low"] = candle["low"]
                new_price_mini_dict["volume"] = candle["volume"]
                new_price_mini_dict["datetime"] = candle["datetime"]
                latest_prices.append(new_price_mini_dict)

        return latest_prices

    def wait_till_next_bar(self, last_bar_timestamp: pd.DatetimeIndex) -> None:
        """Waits the number of seconds till the next bar is released.

        Arguments:
        ----
        last_bar_timestamp {pd.DatetimeIndex} -- The last bar's timestamp.
        """

        last_bar_time = last_bar_timestamp.to_pydatetime()[0].replace(
            tzinfo=timezone.utc
        )
        next_bar_time = last_bar_time + timedelta(seconds=60)
        curr_bar_time = datetime.now(tz=timezone.utc)

        last_bar_timestamp = int(last_bar_time.timestamp())
        next_bar_timestamp = int(next_bar_time.timestamp())
        curr_bar_timestamp = int(curr_bar_time.timestamp())

        time_to_wait_now = next_bar_timestamp - curr_bar_timestamp

        if time_to_wait_now < 0:
            time_to_wait_now = 0

        print("=" * 80)
        print("Pausing for the next bar")
        print("-" * 80)
        print(
            "Curr Time: {time_curr}".format(
                time_curr=curr_bar_time.strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        print(
            "Next Time: {time_next}".format(
                time_next=next_bar_time.strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        print("Sleep Time: {seconds}".format(seconds=time_to_wait_now))
        print("-" * 80)
        print("")

        time_true.sleep(time_to_wait_now)

    def create_stock_frame(self, data: List[dict]) -> DataManager:
        """Generates a new DataManager Object.

        Arguments:
        ----
        data {List[dict]} -- The data to add to the DataManager object.

        Returns:
        ----
        DataManager -- A multi-index pandas data frame built for trading.
        """

        # Create the Frame.
        self.stock_frame = DataManager(data=data)

        return self.stock_frame

    def execute_signals(
        self, signals: List[pd.Series], trades_to_execute: dict
    ) -> List[dict]:
        """Executes the specified trades for each signal.

        Arguments:
        ----
        signals {list} -- A pandas.Series object representing the buy signals and sell signals.
            Will check if series is empty before making any trades.

        Trades:
        ----
        trades_to_execute {dict} -- the trades you want to execute if signals are found.

        Returns:
        ----
        {List[dict]} -- Returns all order responses.

        Usage:
        ----
            >>> trades_dict = {
                    'MSFT': {
                        'trade_func': trading_robot.trades['long_msft'],
                        'trade_id': trading_robot.trades['long_msft'].trade_id
                    }
                }
            >>> signals = indicator_client.check_signals()
            >>> trading_robot.execute_signals(
                    signals=signals,
                    trades_to_execute=trades_dict
                )
        """

        # Define the Buy and sells.
        buys: pd.Series = signals["buys"]
        sells: pd.Series = signals["sells"]

        order_responses = []

        # If we have buys or sells continue.
        if not buys.empty:

            # Grab the buy Symbols.
            symbols_list = buys.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:

                # Check to see if there is a Trade object.
                if symbol in trades_to_execute:

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol, ownership=True
                        )

                    # Set the Execution Flag.
                    trades_to_execute[symbol]["has_executed"] = True
                    trade_obj: Trade = trades_to_execute[symbol]["buy"]["trade_func"]

                    if not self.paper_trading:

                        # Execute the order.
                        order_response = self.execute_orders(trade_obj=trade_obj)

                        order_response = {
                            "order_id": order_response["order_id"],
                            "request_body": order_response["request_body"],
                            "timestamp": datetime.now().isoformat(),
                        }

                        order_responses.append(order_response)

                    else:

                        order_response = {
                            "order_id": trade_obj._generate_order_id(),
                            "request_body": trade_obj.order,
                            "timestamp": datetime.now().isoformat(),
                        }

                        order_responses.append(order_response)

        elif not sells.empty:

            # Grab the buy Symbols.
            symbols_list = sells.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:

                # Check to see if there is a Trade object.
                if symbol in trades_to_execute:

                    # Set the Execution Flag.
                    trades_to_execute[symbol]["has_executed"] = True

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol, ownership=False
                        )

                    trade_obj: Trade = trades_to_execute[symbol]["sell"]["trade_func"]

                    if not self.paper_trading:

                        # Execute the order.
                        order_response = self.execute_orders(trade_obj=trade_obj)

                        order_response = {
                            "order_id": order_response["order_id"],
                            "request_body": order_response["request_body"],
                            "timestamp": datetime.now().isoformat(),
                        }

                        order_responses.append(order_response)

                    else:

                        order_response = {
                            "order_id": trade_obj._generate_order_id(),
                            "request_body": trade_obj.order,
                            "timestamp": datetime.now().isoformat(),
                        }

                        order_responses.append(order_response)

        # Save the response.
        self.save_orders(order_response_dict=order_responses)

        return order_responses

    def execute_orders(self, trade_obj: Backtester) -> dict:
        """Executes a Backtester Object.

        Overview:
        ----
        The `execute_orders` method will execute Backtesters as they're signaled. When executed,
        the `Backtester` object will have the order response saved to it, and the order response will
        be saved to a JSON file for further analysis.

        Arguments:
        ----
        trade_obj {Backtester} -- A Backtester object with the `order` property filled out.

        Returns:
        ----
        {dict} -- An order response dicitonary.
        """

        # Execute the order.
        order_dict = self.api.place_order(
            account=self.trading_account, order=trade_obj.order
        )

        # Store the order.
        trade_obj._order_response = order_dict

        # Process the order response.
        trade_obj._process_order_response()

        return order_dict

    def save_orders(self, order_response_dict: dict) -> bool:
        """Saves the order to a JSON file for further review.

        Arguments:
        ----
        order_response {dict} -- A single order response.

        Returns:
        ----
        {bool} -- `True` if the orders were successfully saved.
        """

        def default(obj):

            if isinstance(obj, bytes):
                return str(obj)

        # Define the folder.
        folder: pathlib.PurePath = pathlib.Path(__file__).parents[1].joinpath("data")

        # See if it exist, if not create it.
        if not folder.exists():
            folder.mkdir()

        # Define the file path.
        file_path = folder.joinpath("orders.json")

        # First check if the file alread exists.
        if file_path.exists():
            with open("data/orders.json", "r") as order_json:
                orders_list = json.load(order_json)
        else:
            orders_list = []

        # Combine both lists.
        orders_list = orders_list + order_response_dict

        # Write the new data back.
        with open(file="data/orders.json", mode="w+") as order_json:
            json.dump(obj=orders_list, fp=order_json, indent=4, default=default)

        return True

    def get_accounts(
        self, account_number: str = None, all_accounts: bool = False
    ) -> dict:
        """Returns all the account balances for a specified account.

        Keyword Arguments:
        ----
        account_number {str} -- The account number you want to query. (default: {None})

        all_accounts {bool} -- Specifies whether you want to grab all accounts `True` or not
            `False`. (default: {False})

        Returns:
        ----
        Dict -- A dictionary containing all the information in your account.

        Usage:
        ----

            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_accounts = trading_robot.session.get_accounts(
                account_number="<YOUR ACCOUNT NUMBER>"
            )
            >>> trading_robot_accounts
            [
                {
                    'account_number': 'ACCOUNT_ID',
                    'account_type': 'CASH',
                    'available_funds': 0.0,
                    'buying_power': 0.0,
                    'cash_available_for_trading': 0.0,
                    'cash_available_for_withdrawl': 0.0,
                    'cash_balance': 0.0,
                    'day_trading_buying_power': 0.0,
                    'long_market_value': 0.0,
                    'maintenance_call': 0.0,
                    'maintenance_requirement': 0.0,
                    'short_balance': 0.0,
                    'short_margin_value': 0.0,
                    'short_market_value': 0.0
                }
            ]
        """

        # Depending on how the client was initalized, either use the state account
        # or the one passed through the function.
        if all_accounts:
            account = "all"
        elif self.trading_account:
            account = self.trading_account
        else:
            account = account_number

        # Grab the accounts.
        accounts = self.api.get_accounts(account=account)

        # Parse the account info.
        accounts_parsed = self._parse_account_balances(accounts_response=accounts)

        return accounts_parsed

    def _parse_account_balances(
        self, accounts_response: Union[Dict, List]
    ) -> List[Dict]:
        """Parses an Account response into a more simplified dictionary.

        Arguments:
        ----
        accounts_response {Union[Dict, List]} -- A response from the `get_accounts` call.

        Returns:
        ----
        List[Dict] -- A list of simplified account dictionaries.
        """

        account_lists = []

        if isinstance(accounts_response, dict):

            account_dict = {}

            for account_type_key in accounts_response:

                account_info = accounts_response[account_type_key]

                account_id = account_info["accountId"]
                account_type = account_info["type"]
                account_current_balances = account_info["currentBalances"]
                # account_inital_balances = account_info['initialBalances']

                account_dict["account_number"] = account_id
                account_dict["account_type"] = account_type
                account_dict["cash_balance"] = account_current_balances["cashBalance"]
                account_dict["long_market_value"] = account_current_balances[
                    "longMarketValue"
                ]

                account_dict["cash_available_for_trading"] = (
                    account_current_balances.get("cashAvailableForTrading", 0.0)
                )
                account_dict["cash_available_for_withdrawl"] = (
                    account_current_balances.get("cashAvailableForWithDrawal", 0.0)
                )
                account_dict["available_funds"] = account_current_balances.get(
                    "availableFunds", 0.0
                )
                account_dict["buying_power"] = account_current_balances.get(
                    "buyingPower", 0.0
                )
                account_dict["day_trading_buying_power"] = account_current_balances.get(
                    "dayTradingBuyingPower", 0.0
                )
                account_dict["maintenance_call"] = account_current_balances.get(
                    "maintenanceCall", 0.0
                )
                account_dict["maintenance_requirement"] = account_current_balances.get(
                    "maintenanceRequirement", 0.0
                )

                account_dict["short_balance"] = account_current_balances.get(
                    "shortBalance", 0.0
                )
                account_dict["short_market_value"] = account_current_balances.get(
                    "shortMarketValue", 0.0
                )
                account_dict["short_margin_value"] = account_current_balances.get(
                    "shortMarginValue", 0.0
                )

                account_lists.append(account_dict)

        elif isinstance(accounts_response, list):

            for account in accounts_response:

                account_dict = {}

                for account_type_key in account:

                    account_info = account[account_type_key]

                    account_id = account_info["accountId"]
                    account_type = account_info["type"]
                    account_current_balances = account_info["currentBalances"]
                    # account_inital_balances = account_info['initialBalances']

                    account_dict["account_number"] = account_id
                    account_dict["account_type"] = account_type
                    account_dict["cash_balance"] = account_current_balances[
                        "cashBalance"
                    ]
                    account_dict["long_market_value"] = account_current_balances[
                        "longMarketValue"
                    ]

                    account_dict["cash_available_for_trading"] = (
                        account_current_balances.get("cashAvailableForTrading", 0.0)
                    )
                    account_dict["cash_available_for_withdrawl"] = (
                        account_current_balances.get("cashAvailableForWithDrawal", 0.0)
                    )
                    account_dict["available_funds"] = account_current_balances.get(
                        "availableFunds", 0.0
                    )
                    account_dict["buying_power"] = account_current_balances.get(
                        "buyingPower", 0.0
                    )
                    account_dict["day_trading_buying_power"] = (
                        account_current_balances.get("dayTradingBuyingPower", 0.0)
                    )
                    account_dict["maintenance_call"] = account_current_balances.get(
                        "maintenanceCall", 0.0
                    )
                    account_dict["maintenance_requirement"] = (
                        account_current_balances.get("maintenanceRequirement", 0.0)
                    )
                    account_dict["short_balance"] = account_current_balances.get(
                        "shortBalance", 0.0
                    )
                    account_dict["short_market_value"] = account_current_balances.get(
                        "shortMarketValue", 0.0
                    )
                    account_dict["short_margin_value"] = account_current_balances.get(
                        "shortMarginValue", 0.0
                    )

                    account_lists.append(account_dict)

        return account_lists

    def get_positions(
        self, account_number: str = None, all_accounts: bool = False
    ) -> List[Dict]:
        """Gets all the positions for a specified account number.

        Arguments:
        ----
        account_number (str, optional): The account number of the account you want
            to pull positions for. Defaults to None.

        all_accounts (bool, optional): If you want to return all the positions for every
            account then set to `True`. Defaults to False.

        Returns:
        ----
        List[Dict]: A list of Position objects.

        Usage:
        ----

            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_positions = trading_robot.session.get_positions(
                account_number="<YOUR ACCOUNT NUMBER>"
            )
            >>> trading_robot_positions
            [
                {
                    'account_number': '111111111',
                    'asset_type': 'EQUITY',
                    'average_price': 0.00,
                    'current_day_profit_loss': -0.96,
                    'current_day_profit_loss_percentage': -5.64,
                    'cusip': '565849106',
                    'description': '',
                    'long_quantity': 3.0,
                    'market_value': 16.05,
                    'settled_long_quantity': 3.0,
                    'settled_short_quantity': 0.0,
                    'short_quantity': 0.0,
                    'sub_asset_type': '',
                    'symbol': 'MRO',
                    'type': ''
                },
                {
                    'account_number': '111111111',
                    'asset_type': 'EQUITY',
                    'average_price': 5.60667,
                    'current_day_profit_loss': -0.96,
                    'current_day_profit_loss_percentage': -5.64,
                    'cusip': '565849106',
                    'description': '',
                    'long_quantity': 3.0,
                    'market_value': 16.05,
                    'settled_long_quantity': 3.0,
                    'settled_short_quantity': 0.0,
                    'short_quantity': 0.0,
                    'sub_asset_type': '',
                    'symbol': 'MRO',
                    'type': ''
                }
            ]
        """

        if all_accounts:
            account = "all"
        elif self.trading_account and account_number is None:
            account = self.trading_account
        else:
            account = account_number

        # Grab the positions.
        positions = self.api.get_accounts(account=account, fields=["positions"])

        # Parse the positions.
        positions_parsed = self._parse_account_positions(positions_response=positions)

        return positions_parsed

    def _parse_account_positions(
        self, positions_response: Union[List, Dict]
    ) -> List[Dict]:
        """Parses the response from the `get_positions` into a more simplified list.

        Arguments:
        ----
        positions_response {Union[List, Dict]} -- Either a list or a dictionary that represents a position.

        Returns:
        ----
        List[Dict] -- A more simplified list of positions.
        """

        positions_lists = []

        if isinstance(positions_response, dict):

            for account_type_key in positions_response:

                account_info = positions_response[account_type_key]

                account_id = account_info["accountId"]
                positions = account_info["positions"]

                for position in positions:
                    position_dict = {}
                    position_dict["account_number"] = account_id
                    position_dict["average_price"] = position["averagePrice"]
                    position_dict["market_value"] = position["marketValue"]
                    position_dict["current_day_profit_loss_percentage"] = position[
                        "currentDayProfitLossPercentage"
                    ]
                    position_dict["current_day_profit_loss"] = position[
                        "currentDayProfitLoss"
                    ]
                    position_dict["long_quantity"] = position["longQuantity"]
                    position_dict["short_quantity"] = position["shortQuantity"]
                    position_dict["settled_long_quantity"] = position[
                        "settledLongQuantity"
                    ]
                    position_dict["settled_short_quantity"] = position[
                        "settledShortQuantity"
                    ]

                    position_dict["symbol"] = position["instrument"]["symbol"]
                    position_dict["cusip"] = position["instrument"]["cusip"]
                    position_dict["asset_type"] = position["instrument"]["assetType"]
                    position_dict["sub_asset_type"] = position["instrument"].get(
                        "subAssetType", ""
                    )
                    position_dict["description"] = position["instrument"].get(
                        "description", ""
                    )
                    position_dict["type"] = position["instrument"].get("type", "")

                    positions_lists.append(position_dict)

        elif isinstance(positions_response, list):

            for account in positions_response:

                for account_type_key in account:

                    account_info = account[account_type_key]

                    account_id = account_info["accountId"]
                    positions = account_info["positions"]

                    for position in positions:
                        position_dict = {}
                        position_dict["account_number"] = account_id
                        position_dict["average_price"] = position["averagePrice"]
                        position_dict["market_value"] = position["marketValue"]
                        position_dict["current_day_profit_loss_percentage"] = position[
                            "currentDayProfitLossPercentage"
                        ]
                        position_dict["current_day_profit_loss"] = position[
                            "currentDayProfitLoss"
                        ]
                        position_dict["long_quantity"] = position["longQuantity"]
                        position_dict["short_quantity"] = position["shortQuantity"]
                        position_dict["settled_long_quantity"] = position[
                            "settledLongQuantity"
                        ]
                        position_dict["settled_short_quantity"] = position[
                            "settledShortQuantity"
                        ]

                        position_dict["symbol"] = position["instrument"]["symbol"]
                        position_dict["cusip"] = position["instrument"]["cusip"]
                        position_dict["asset_type"] = position["instrument"][
                            "assetType"
                        ]
                        position_dict["sub_asset_type"] = position["instrument"].get(
                            "subAssetType", ""
                        )
                        position_dict["description"] = position["instrument"].get(
                            "description", ""
                        )
                        position_dict["type"] = position["instrument"].get("type", "")

                        positions_lists.append(position_dict)

        return positions_lists
