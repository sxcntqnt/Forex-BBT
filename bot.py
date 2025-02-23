# Standard library imports
import json
import asyncio
import re
import time as time_true
import pathlib
from datetime import datetime, timezone, timedelta, UTC
from typing import List, Dict, Union, Optional

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

    async def fetch_historical_data(
        self,
        symbol: str,
        start_timestamp: int,
        end_timestamp: int,
        bar_type: str = "minute",
    ) -> Union[Dict, List]:
        """
        Fetches historical data for a single symbol.

        Args:
            symbol (str): The symbol to fetch historical data for.
            start_timestamp (int): Start timestamp for the data request.
            end_timestamp (int): End timestamp for the data request.
            bar_type (str): Type of bar (default "minute").

        Returns:
            tuple: (symbol_data, symbol_prices) where symbol_data contains raw candle data
                   and symbol_prices contains parsed candle data.
        """
        clean_symbol = symbol.replace("frx", "")
        if not re.match(r"^[a-zA-Z]{2,30}$", clean_symbol):
            self.logger.warning(f"Skipping invalid symbol: {symbol}")
            return {}, []

        args = {
            "ticks_history": clean_symbol,
            "start": start_timestamp,
            "end": end_timestamp,
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
                    "datetime": self.timestamp_utils.from_seconds(
                        candle["epoch"]
                    ),  # Changed to seconds
                }
                for candle in response["candles"]
            ]
            return symbol_data, symbol_prices
        except Exception as e:
            self.logger.error(
                f"Error fetching data for {symbol}: %s", str(e), exc_info=True
            )
            return {}, []

    async def grab_historical_prices(
        self,
        start_date: datetime,
        end_date: datetime,
        bar_size: int = 1,
        bar_type: str = "minute",
        symbols: List[str] = None,
    ) -> Dict[str, Union[List[Dict], Dict]]:
        """
        Grabs historical price data for the given symbols and date range.

        Args:
            start_date (datetime): Start date for fetching the historical data.
            end_date (datetime): End date for fetching the historical data.
            bar_size (int): Size of the bars (default 1).
            bar_type (str): Type of bar (default "minute").
            symbols (List[str]): List of symbols to fetch data for (default is config.SYMBOLS).

        Returns:
            Dict[str, Union[List[Dict], Dict]]: A dictionary with the historical data.
        """
        self.logger.info(
            f"Fetching historical prices for {len(symbols or self.config.SYMBOLS)} symbols."
        )
        if not symbols:
            symbols = self.config.SYMBOLS

        start_timestamp = self.timestamp_utils.to_seconds(
            start_date
        )  # Changed to seconds
        end_timestamp = self.timestamp_utils.to_seconds(end_date)  # Changed to seconds
        historical_data = {}
        new_prices = []

        tasks = [
            self.fetch_historical_data(symbol, start_timestamp, end_timestamp, bar_type)
            for symbol in symbols
        ]
        results = await asyncio.gather(
            *tasks, return_exceptions=True
        )  # Capture exceptions

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Task for {symbol} failed: %s", str(result))
                continue
            symbol_data, symbol_prices = result
            if symbol_data:
                historical_data[symbol] = symbol_data
            new_prices.extend(symbol_prices)

        historical_data["aggregated"] = new_prices
        self.logger.info(f"Fetched historical data for {len(historical_data)} symbols.")
        return historical_data

    async def get_latest_bar(self) -> List[dict]:
        latest_prices = []
        end_date = datetime.now(tz=UTC)
        start_date = end_date - timedelta(days=1)
        for symbol in self.config.SYMBOLS:
            data = await self.grab_historical_prices(
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
        self.logger.info("Executing signals (placeholder implementation)")
        return []

    async def execute_orders(self, trade_obj: Backtester) -> dict:
        self.logger.info("Executing order (placeholder)")
        return {"order_id": "mock123"}

    async def save_orders(self, order_response_dict: list) -> bool:
        async with aiofiles.open("data/orders.json", mode="w") as f:
            await f.write(json.dumps(order_response_dict, indent=4))
        return True

    async def get_accounts(
        self, account_number: str = None, all_accounts: bool = False
    ) -> dict:
        return {"DEMO123": {"balance": 10000}}

    async def get_positions(
        self, account_number: str = None, all_accounts: bool = False
    ) -> List[Dict]:
        return []

    def _parse_account_balances(
        self, accounts_response: Union[Dict, List]
    ) -> List[Dict]:
        return [{"account_number": "DEMO123", "cash_balance": 10000}]

    def _parse_account_positions(
        self, positions_response: Union[List, Dict]
    ) -> List[Dict]:
        return []

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
        start_date = datetime.strptime(self.config.BACKTEST_START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.BACKTEST_END_DATE, "%Y-%m-%d")
        data = await self.grab_historical_prices(start_date, end_date)
        self.data_manager = DataManager(
            self.config, self.api, self.logger, data=data["aggregated"]
        )
        self.logger.info("DataManager initialized with historical data.")

    async def stop(self):
        self.running = False
        await self.unsubscribe()
        self.logger.info("ForexBot stopped.")

    async def run_backtest(self):
        results = await self.backtester.run()
        self.logger.info(f"Backtest results: {results}")
        return results
