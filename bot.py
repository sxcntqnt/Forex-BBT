# Standard library imports
import json
import asyncio
import re
import time as time_true
import pathlib
from datetime import datetime, timezone, timedelta, UTC
from typing import List, Dict, Union, Optional, Tuple

# Third-party library imports
import pandas as pd
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
import aiofiles
from websockets.client import WebSocketClientProtocol
from deriv_api import DerivAPI
from rx import Observable

# Local application/library-specific imports
from config import Config
from strategy import StrategyManager
from portfolio_manager import PortfolioManager
from backtester import Backtester
from risk_manager import RiskManager
from monitor import Monitor
from data_manager import DataManager
from utils import TimestampUtils, PerformanceMonitor


class ForexBot:
    def __init__(
        self,
        config: Config,
        data_manager: DataManager,
        strategy_manager: StrategyManager,
        logger,
        api,
        paper_trading: bool = True,
    ):
        """Central trading bot with dependency injection."""
        # Verify API connection
        asyncio.create_task(self._verify_api_connection())

        self.config = config
        self.paper_trading = paper_trading
        self.logger = logger
        self.api = api
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.backtester = Backtester(
            config, api, self.data_manager, self.strategy_manager
        )
        self.monitor = Monitor(self.config, self.backtester)
        self.performance_monitor = PerformanceMonitor(self.portfolio_manager)
        self.subscriptions: Dict[str, dict] = {}
        self.active_trades: Dict[str, dict] = {}
        self.trades: Dict[str, Backtester] = {}
        self.running = False
        self.trading_account = "DEMO123"
        self.timestamp_utils = TimestampUtils()
        self.start_date = self.timestamp_utils.from_seconds(
            self.timestamp_utils.to_seconds(datetime.now(tz=timezone.utc))
            - config.HISTORICAL_DAYS * 86400
        )
        self.end_date = datetime.now(tz=timezone.utc)
        self._bar_size = 1
        self._bar_type = "minute"

    def _market_open(self, start_hour: int, end_hour: int) -> bool:
        """Check if the market is open based on the given start and end hours."""
        now = datetime.now(tz=timezone.utc)
        start_time = self.timestamp_utils.to_seconds(
            now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        )
        end_time = self.timestamp_utils.to_seconds(
            now.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        )
        current_time = self.timestamp_utils.to_seconds(now)
        return start_time <= current_time <= end_time

    @property
    def pre_market_open(self) -> bool:
        """Check if the pre-market is open."""
        return self._market_open(8, 9)

    @property
    def regular_market_open(self) -> bool:
        """Check if the regular market is open."""
        return self._market_open(13, 20)

    @property
    def post_market_open(self) -> bool:
        """Check if the post-market is open."""
        return self._market_open(20, 24)

    async def _verify_api_connection(self) -> None:
        """Verify Deriv API connection asynchronously."""
        try:
            response = await self.api.ping({"ping": 1})
            self.logger.debug("Ping response: %s", response)
            if response.get("ping") == "pong":
                self.logger.info("Deriv API connection established")
            else:
                self.logger.error("Unexpected ping response: %s", response)
                raise ValueError("API ping failed")
        except Exception as e:
            self.logger.error("Deriv API initialization failed: %s", str(e))
            raise

    async def create_portfolio(self) -> PortfolioManager:
        self.logger.info(f"Creating portfolio for account {self.trading_account}.")
        portfolio = PortfolioManager(self.config, account_number=self.trading_account)
        portfolio.derivCli = self.api
        self.logger.info(f"Portfolio created: {portfolio}")
        return portfolio

    async def create_trade(
        self,
        trade_id: str,
        enter_or_exit: str,
        long_or_short: str,
        order_type: str = "mkt",
        price: float = 0.0,
        stop_limit_price: float = 0.0,
    ) -> Backtester:
        self.logger.info(
            f"Creating trade {trade_id} - {enter_or_exit} {long_or_short} order."
        )
        trade = Backtester(
            self.config, self.api, self.data_manager, self.strategy_manager
        )
        trade.new_trade(
            trade_id=trade_id,
            order_type=order_type,
            side=long_or_short,
            enter_or_exit=enter_or_exit,
            price=price,
            stop_limit_price=stop_limit_price,
        )
        trade.account = self.trading_account
        trade._td_client = self.api
        self.trades[trade_id] = trade
        self.logger.info(f"Trade created successfully: {trade_id}")
        return trade

    async def grab_historical_data(
        self,
        start_timestamp: int,
        end_timestamp: int,
        symbol: str,
        bar_type: str = "minute",
    ) -> Tuple[Dict, List]:
        """
        Fetches historical data for a single symbol.

        Args:
            symbol (str): The symbol to fetch historical data for.
            start_timestamp (int): Start timestamp for the data request (Unix seconds).
            end_timestamp (int): End timestamp for the data request (Unix seconds).
            bar_type (str): Type of bar (default "minute").

        Returns:
            tuple: (symbol_data, symbol_prices) where symbol_data contains raw candle data
                   and symbol_prices contains parsed candle data.
        """
        # Convert datetime to seconds if provided
        if isinstance(start_timestamp, datetime):
            start_timestamp = self.timestamp_utils.to_seconds(start_timestamp)
        if isinstance(end_timestamp, datetime):
            end_timestamp = self.timestamp_utils.to_seconds(end_timestamp)

        args = {
            "start": start_timestamp,
            "end": end_timestamp,
            "ticks_history": symbol,
            "style": "candles",
            "granularity": 60 if bar_type == "minute" else 300,
            "count": 5000,
            "adjust_start_time": 1,
        }
        self.logger.debug(f"Requesting ticks_history with args: %s", args)

        try:
            response = await self.api.ticks_history(args)
            self.logger.debug(f"Response for {symbol}: %s", response)
            if "candles" not in response or not response["candles"]:
                self.logger.warning(
                    f"No candles data for {symbol} in response: %s", response
                )
                return {}, []
            symbol_data = {"candles": response["candles"]}
            symbol_prices = [
                {
                    "symbol": symbol,
                    "open": float(candle["open"]),
                    "close": float(candle["close"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "volume": candle.get("volume", 0),
                    "datetime": self.timestamp_utils.from_seconds(candle["epoch"]),
                }
                for candle in response["candles"]
            ]
            return symbol_data, symbol_prices
        except Exception as e:
            self.logger.error(
                f"Error fetching data for {symbol}: %s", str(e), exc_info=True
            )
            return {}, []

    async def delete_trade(self, trade_id: str) -> None:
        self.logger.info(f"Attempting to delete trade {trade_id}.")
        if trade_id in self.trades:
            del self.trades[trade_id]
            self.logger.info(f"Trade {trade_id} deleted successfully.")
        else:
            self.logger.error(f"Trade {trade_id} not found.")

    async def grab_current_quotes(self) -> dict:
        """
        Fetches current quotes for all positions in the portfolio.

        Returns:
            dict: A dictionary of current quotes for the portfolio positions.
        """
        self.logger.info("Fetching current quotes for all positions.")
        symbols = self.portfolio_manager.positions.keys()
        try:
            quotes = await self.api.active_symbols({"active_symbols": "brief"})
            self.logger.info(f"Fetched quotes for {len(symbols)} symbols.")
            return {
                s["symbol"]: s
                for s in quotes["active_symbols"]
                if s["symbol"] in symbols
            }
        except Exception as e:
            self.logger.error(f"Error fetching quotes: {str(e)}")
            return {}


    async def get_latest_bar(self) -> List[dict]:
        latest_prices = []
        end_date = datetime.now(tz=UTC)
        start_date = end_date - timedelta(days=1)
        for symbol in self.config.SYMBOLS:
            data = await self.grab_historical_data(
                start_date, end_date, 1, "minute", [symbol]
            )
            if "aggregated" in data and data["aggregated"]:
                latest_prices.append(data["aggregated"][-1])
        return latest_prices

    async def wait_till_next_bar(self, last_bar_timestamp: datetime) -> None:
        last_bar_time = (
            last_bar_timestamp
            if last_bar_timestamp.tzinfo
            else last_bar_timestamp.replace(tzinfo=timezone.utc)
        )
        next_bar_time = last_bar_time + timedelta(seconds=60)
        curr_bar_time = datetime.now(tz=timezone.utc)
        time_to_wait = max(
            int(next_bar_time.timestamp() - curr_bar_time.timestamp()), 0
        )
        self.logger.info(f"Waiting {time_to_wait} seconds for next bar.")
        await asyncio.sleep(time_to_wait)


    async def create_stock_frame(self, data: List[dict]) -> DataManager:
        self.stock_frame = DataManager(self.config, self.api, self.logger, data=data)
        return self.stock_frame

    async def execute_signals(
        self, signals: List[pd.Series], trades_to_execute: dict
    ) -> List[dict]:
        """Executes the specified trades for each signal asynchronously.

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
                if symbol in trades_to_execute:
                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol, ownership=True
                        )

                    trades_to_execute[symbol]["has_executed"] = True
                    trade_obj: Trade = trades_to_execute[symbol]["buy"]["trade_func"]

                    if not self.paper_trading:
                        # Execute the order asynchronously.
                        order_response = await self.execute_orders(trade_obj=trade_obj)
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
            # Grab the sell Symbols.
            symbols_list = sells.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:
                if symbol in trades_to_execute:
                    trades_to_execute[symbol]["has_executed"] = True

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol, ownership=False
                        )

                    trade_obj: Trade = trades_to_execute[symbol]["sell"]["trade_func"]

                    if not self.paper_trading:
                        # Execute the order asynchronously.
                        order_response = await self.execute_orders(trade_obj=trade_obj)
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

        # Save the response asynchronously.
        await self.save_orders(order_response_dict=order_responses)

        return order_responses

    async def execute_orders(self, trade_obj: Backtester) -> dict:
        """Executes a Backtester Object asynchronously.

        Arguments:
        ----
        trade_obj {Backtester} -- A Backtester object with the `order` property filled out.

        Returns:
        ----
        {dict} -- An order response dictionary.
        """
        # Execute the order asynchronously.
        order_dict = await self.api.place_order(
            account=self.trading_account, order=trade_obj.order
        )

        # Store the order.
        trade_obj._order_response = order_dict

        # Process the order response.
        trade_obj._process_order_response()

        return order_dict

    async def save_orders(self, order_response_dict: dict) -> bool:
        """Saves the order to a JSON file for further review asynchronously.

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

        # See if it exists, if not create it.
        if not folder.exists():
            folder.mkdir()

        # Define the file path.
        file_path = folder.joinpath("orders.json")

        # First check if the file already exists.
        if file_path.exists():
            async with aiofiles.open(file_path, mode="r") as order_json:
                orders_list = json.load(await order_json.read())
        else:
            orders_list = []

        # Combine both lists.
        orders_list = orders_list + order_response_dict

        # Write the new data back asynchronously.
        async with aiofiles.open(file_path, mode="w+") as order_json:
            await order_json.write(json.dumps(orders_list, indent=4, default=default))

        return True

    async def get_account_balances(self) -> List[Dict]:
        """Fetches account balances asynchronously and parses them.

        Returns:
        ----
        List[Dict]: A list of simplified account balance dictionaries.
        """
        accounts_response = (
            await self.api.get_accounts()
        )  # Assuming this is an async call
        return self._parse_account_balances(accounts_response)

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
            for account_type_key in accounts_response:
                account_info = accounts_response[account_type_key]
                account_dict = self._extract_account_data(account_info)
                account_lists.append(account_dict)

        elif isinstance(accounts_response, list):
            for account in accounts_response:
                for account_type_key in account:
                    account_info = account[account_type_key]
                    account_dict = self._extract_account_data(account_info)
                    account_lists.append(account_dict)

        return account_lists

    def _extract_account_data(self, account_info: Dict) -> Dict:
        """Extracts relevant data from an account information dictionary.

        Arguments:
        ----
        account_info {Dict} -- The account information dictionary.

        Returns:
        ----
        {Dict} -- A dictionary containing the extracted account data.
        """
        account_dict = {
            "account_number": account_info["accountId"],
            "account_type": account_info["type"],
            "cash_balance": account_info["currentBalances"]["cashBalance"],
            "long_market_value": account_info["currentBalances"]["longMarketValue"],
            "cash_available_for_trading": account_info["currentBalances"].get(
                "cashAvailableForTrading", 0.0
            ),
            "cash_available_for_withdrawal": account_info["currentBalances"].get(
                "cashAvailableForWithDrawal", 0.0
            ),
            "available_funds": account_info["currentBalances"].get(
                "availableFunds", 0.0
            ),
            "buying_power": account_info["currentBalances"].get("buyingPower", 0.0),
            "day_trading_buying_power": account_info["currentBalances"].get(
                "dayTradingBuyingPower", 0.0
            ),
            "maintenance_call": account_info["currentBalances"].get(
                "maintenanceCall", 0.0
            ),
            "maintenance_requirement": account_info["currentBalances"].get(
                "maintenanceRequirement", 0.0
            ),
            "short_balance": account_info["currentBalances"].get("shortBalance", 0.0),
            "short_market_value": account_info["currentBalances"].get(
                "shortMarketValue", 0.0
            ),
            "short_margin_value": account_info["currentBalances"].get(
                "shortMarginValue", 0.0
            ),
        }
        return account_dict

    async def get_positions(
        self, account_number: str = None, all_accounts: bool = False
    ) -> List[Dict]:
        """Gets all the positions for a specified account number asynchronously.

        Arguments:
        ----
        account_number (str, optional): The account number of the account you want
            to pull positions for. Defaults to None.

        all_accounts (bool, optional): If you want to return all the positions for every
            account then set to `True`. Defaults to False.

        Returns:
        ----
        List[Dict]: A list of Position objects.
        """
        if all_accounts:
            account = "all"
        elif self.trading_account and account_number is None:
            account = self.trading_account
        else:
            account = account_number

        # Grab the positions asynchronously.
        positions = await self.api.get_accounts(account=account, fields=["positions"])

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
                    position_dict = self._extract_position_data(account_id, position)
                    positions_lists.append(position_dict)

        elif isinstance(positions_response, list):
            for account in positions_response:
                for account_type_key in account:
                    account_info = account[account_type_key]
                    account_id = account_info["accountId"]
                    positions = account_info["positions"]

                    for position in positions:
                        position_dict = self._extract_position_data(
                            account_id, position
                        )
                        positions_lists.append(position_dict)

        return positions_lists

    def _extract_position_data(self, account_id: str, position: Dict) -> Dict:
        """Extracts relevant data from a position dictionary.

        Arguments:
        ----
        account_id {str} -- The account ID associated with the position.
        position {Dict} -- The position dictionary.

        Returns:
        ----
        {Dict} -- A dictionary containing the extracted position data.
        """
        position_dict = {
            "account_number": account_id,
            "average_price": position["averagePrice"],
            "market_value": position["marketValue"],
            "current_day_profit_loss_percentage": position[
                "currentDayProfitLossPercentage"
            ],
            "current_day_profit_loss": position["currentDayProfitLoss"],
            "long_quantity": position["longQuantity"],
            "short_quantity": position["shortQuantity"],
            "settled_long_quantity": position["settledLongQuantity"],
            "settled_short_quantity": position["settledShortQuantity"],
            "symbol": position["instrument"]["symbol"],
            "cusip": position["instrument"]["cusip"],
            "asset_type": position["instrument"]["assetType"],
            "sub_asset_type": position["instrument"].get("subAssetType", ""),
            "description": position["instrument"].get("description", ""),
            "type": position["instrument"].get("type", ""),
        }
        return position_dict

    async def run(self):
        self.running = True
        self.logger.info("ForexBot started running.")
        await self._verify_api_connection()
        for symbol in self.config.SYMBOLS:
            await self.subscribe_to_symbol(symbol, self.api)
        while self.running:
            await self.check_trades()
            await self.performance_monitor.update(self.portfolio_manager)
            await asyncio.sleep(1)

    async def subscribe_to_symbol(self, symbol, api):
        try:
            await api.ticks(symbol)
            api.subscribe(self.create_tick_callback(symbol))
            self.subscriptions[symbol] = {"symbol": symbol}
            self.logger.info(f"Subscribed to {symbol} tick stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")

    def create_tick_callback(self, symbol):
        def callback(data):
            self.data_manager.update(symbol, data)
            self.logger.debug(f"Tick for {symbol}: {data}")

        return callback

    async def check_trades(self):
        """Check if any trade conditions are met and execute if necessary."""
        for symbol in self.config.SYMBOLS:
            if await self.strategy_manager.should_enter_trade(symbol):
                await self.enter_trade(symbol)
        await self.monitor.check_open_positions(self.api)
        await self.performance_monitor.update(self.portfolio_manager)

    async def enter_trade(self, symbol):
        if not self.risk_manager.can_enter_trade(symbol):
            self.logger.warning(f"Risk limit reached for {symbol}")
            return
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

    async def unsubscribe(self):
        for symbol in self.subscriptions:
            await self.api.forget(symbol)
        self.subscriptions.clear()

    async def initialize_data_manager(self):
        """Initializes the DataManager with historical data for specified symbols."""
        start_date = datetime.strptime(self.config.BACKTEST_START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.BACKTEST_END_DATE, "%Y-%m-%d")
        
        # Convert to timestamps
        start_ts = self.timestamp_utils.to_seconds(start_date)
        end_ts = self.timestamp_utils.to_seconds(end_date)
        
        # Loop over symbols or specify one
        for symbol in self.config.SYMBOLS:  # Assuming SYMBOLS is a list of symbols
            symbol_data, symbol_prices = await self.grab_historical_data(start_ts, end_ts, symbol)
            
            # Initialize DataManager with historical data
            self.data_manager = DataManager(
                self.config, 
                self.api, 
                self.logger, 
                data={symbol: {"raw": symbol_data, "prices": symbol_prices}}
            )
            self.logger.info(f"DataManager initialized with historical data for {symbol}.")

    async def stop(self):
        self.running = False
        await self.unsubscribe()
        self.logger.info("ForexBot stopped.")

    async def run_backtest(self):
        results = await self.backtester.run()
        self.logger.info(f"Backtest results: {results}")
        return results
