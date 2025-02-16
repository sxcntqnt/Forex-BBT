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
from typing import Dict, Union
from typing import Optional
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
import aiofiles
from websockets import WebSocketClientProtocol
from deriv_api import DerivAPI
from rx import Observable
from config import Config
from configparser import ConfigParser
from strategy import StrategyManager
from portfolio_manager import PortfolioManager
from backtester import  Backtester
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
        config: Config,
        data_manager: DataManager,  # Add this parameter
        strategy_manager: StrategyManager,  # Add this parameter
        logger,
	api,
        paper_trading: bool = True

    ):
        """Central trading bot with dependency injection"""
        self.config = config
        self.paper_trading = paper_trading
        self.logger =logger

        # Core components
        self.api = api
        self.data_manager = DataManager(config, api, logger)
        self.strategy = StrategyManager(data_manager, api)
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.backtester = Backtester(config, api, data_manager, strategy_manager)
        # State management
        self.active_trades: Dict[str, dict] = {}
        self.running = False

        # Default state variables
        self.subscriptions = {}
        self.start_date = datetime.now() - timedelta(days=config.HISTORICAL_DAYS)
        self.end_date = datetime.now()

    def _market_open(self, start_hour: int, end_hour: int) -> bool:
        """Helper function to check if the market is within specified hours in UTC."""
        start_time = datetime.utcnow().replace(hour=start_hour, minute=0, second=0).timestamp()
        end_time = datetime.utcnow().replace(hour=end_hour, minute=0, second=0).timestamp()
        current_time = datetime.utcnow().timestamp()
        return start_time <= current_time <= end_time

    @property
    def pre_market_open(self) -> bool:
        """Checks if pre-market is open (8:00 AM - 9:30 AM UTC)."""
        return self._market_open(8, 9)

    @property
    def regular_market_open(self) -> bool:
        """Checks if regular market is open (1:30 PM - 8:00 PM UTC)."""
        return self._market_open(13, 20)

    @property
    def post_market_open(self) -> bool:
        """Checks if post-market is open (8:00 PM - 12:00 AM UTC)."""
        return self._market_open(20, 24)

    async def create_portfolio(self) -> PortfolioManager:
        """Asynchronously create a new portfolio and assign it a trading client."""
        self.logger.info(f"Creating portfolio for account {self.trading_account}.")
        
        # Initialize and configure portfolio
        portfolio = PortfolioManager(account_number=self.trading_account)
        portfolio.td_client = self.api
        
        self.logger.info(f"Portfolio created: {portfolio}")
        return portfolio

    async def create_trade(
        self,
        trade_id: str,
        enter_or_exit: str,
        long_or_short: str,
        order_type: str = "mkt",
        price: float = 0.0,
        stop_limit_price: float = 0.0
    ) -> Backtester:
        """Asynchronously initializes a new trade using a Backtester object."""
        self.logger.info(f"Creating trade {trade_id} - {enter_or_exit} {long_or_short} order.")

        # Initialize Backtester for this trade
        trade = Backtester()
        trade.new_trade(
            trade_id=trade_id,
            order_type=order_type,
            side=long_or_short,
            enter_or_exit=enter_or_exit,
            price=price,
            stop_limit_price=stop_limit_price,
        )
        
        # Set client for trade
        trade.account = self.trading_account
        trade._td_client = self.api
        
        # Store the trade in the trade dictionary
        self.trades[trade_id] = trade
        
        self.logger.info(f"Trade created successfully: {trade_id}")
        return trade

    async def delete_trade(self, trade_id: str) -> None:
        """Deletes an existing trade by trade_id from the `trades` collection."""
        self.logger.info(f"Attempting to delete trade {trade_id}.")
        
        if trade_id in self.trades:
            del self.trades[trade_id]
            self.logger.info(f"Trade {trade_id} deleted successfully.")
        else:
            self.logger.error(f"Trade {trade_id} not found.")


    async def grab_current_quotes(self) -> dict:
        """Asynchronously fetches current quotes for all positions in the portfolio."""
        self.logger.info("Fetching current quotes for all positions.")
        
        # First grab all the symbols.
        symbols = self.portfolio.positions.keys()

        try:
            # Fetch quotes using the API
            quotes = await self.api.get_quotes(instruments=list(symbols))
            self.logger.info(f"Fetched quotes for {len(symbols)} symbols.")
            return quotes
        except Exception as e:
            self.logger.error(f"Error fetching quotes: {str(e)}")
            return {}

    @staticmethod
    def milliseconds_since_epoch(dt: datetime) -> int:
        """Converts a datetime object to milliseconds since epoch."""
        return int(dt.timestamp() * 1000)

    async def grab_historical_prices(
        self,
        start_date: datetime,
        end_date: datetime,
        bar_size: int = 1,
        bar_type: str = "minute",
        symbols: List[str] = None,
    ) -> Dict[str, Union[List[Dict], Dict]]:
        """Asynchronously fetches historical prices for positions in the portfolio."""
        self.logger.info(f"Fetching historical prices for {len(symbols)} symbols.")
        
        if not isinstance(symbols, list) and symbols is not None:
            raise ValueError("Symbols must be a list")

        if not symbols:
            symbols = self.portfolio_manager.positions  # Assuming this is predefined

        if not symbols:
            raise ValueError("No symbols provided for historical data retrieval.")

        if self.api is None:
            raise ValueError("API instance is not initialized!")

        # Convert dates to timestamps
        start_timestamp = self.milliseconds_since_epoch(start_date) // 1000
        end_timestamp = self.milliseconds_since_epoch(end_date) // 1000

        self.logger.debug(f"Timestamp range - start: {start_timestamp}, end: {end_timestamp}")

        # Initialize containers
        new_prices = []  # For storing all price data
        historical_data = {}  # For symbol-specific historical data

        for symbol in symbols:
            clean_symbol = symbol.replace('frx', '')

            # Verify symbol format (basic check)
            if not re.match(r'^[a-zA-Z]{2,30}$', clean_symbol):
                self.logger.warning(f"Skipping invalid symbol: {symbol}")
                continue

            self.args = {
                "symbol": clean_symbol,
                "start": start_timestamp,
                "end": end_timestamp,
                "granularity": 300,  # Assuming the granularity is fixed at 5 minutes
                "style": "candles",   # Assuming you want candles as the default style
                "count": 5000,
                "adjust_start_time": 1
            }

            try:
                historical_prices_response = await self.api.ticks_history(self.args)

                if historical_prices_response is None:
                    self.logger.warning(f"No data returned for {symbol}")
                    continue

                if "candles" not in historical_prices_response:
                    self.logger.error(f"Invalid response structure for {symbol}")
                    continue

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
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue

        # Add aggregated prices to the historical data
        historical_data["aggregated"] = new_prices

        self.logger.info(f"Successfully fetched historical data for {len(historical_data)} symbols.")
        return historical_data

    async def get_latest_bar(self) -> List[dict]:
        """Returns the latest bar for each symbol in the portfolio."""
        
        bar_size = self._bar_size
        bar_type = self._bar_type
        
        end_date = datetime.today()
        start_date = end_date - timedelta(days=1)
        start = str(milliseconds_since_epoch(dt_object=start_date))
        end = str(milliseconds_since_epoch(dt_object=end_date))
        
        latest_prices = []
        
        # Loop through each symbol asynchronously
        for symbol in self.portfolio.positions:
            try:
                # Fetch the price history asynchronously (assuming API supports async)
                historical_prices_response = await self.api.get_price_history(
                    symbol=symbol,
                    period_type="day",
                    start_date=start,
                    end_date=end,
                    frequency_type=bar_type,
                    frequency=bar_size,
                    extended_hours=True,
                )
                
            except Exception as e:
                print(f"Error fetching price history for {symbol}: {e}")
                await asyncio.sleep(2)  # Non-blocking sleep for retries

                # Retry fetching the price history if there was an error
                try:
                    historical_prices_response = await self.api.get_price_history(
                        symbol=symbol,
                        period_type="day",
                        start_date=start,
                        end_date=end,
                        frequency_type=bar_type,
                        frequency=bar_size,
                        extended_hours=True,
                    )
                except Exception as e:
                    print(f"Retry failed for {symbol}: {e}")
                    continue

            # Parse the last candle
            for candle in historical_prices_response["candles"][-1:]:
                new_price_mini_dict = {
                    "symbol": symbol,
                    "open": candle["open"],
                    "close": candle["close"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "volume": candle["volume"],
                    "datetime": candle["datetime"]
                }
                latest_prices.append(new_price_mini_dict)

        return latest_prices

    async def wait_till_next_bar(self, last_bar_timestamp: pd.DatetimeIndex) -> None:
        """Waits the number of seconds till the next bar is released."""
        
        last_bar_time = last_bar_timestamp.to_pydatetime()[0].replace(tzinfo=timezone.utc)
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
            "Curr Time: {time_curr}".format(time_curr=curr_bar_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        print(
            "Next Time: {time_next}".format(time_next=next_bar_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        print("Sleep Time: {seconds}".format(seconds=time_to_wait_now))
        print("-" * 80)
        print("")
        
        # Use asyncio.sleep for async non-blocking delay
        await asyncio.sleep(time_to_wait_now)

    async def create_stock_frame(self, data: List[dict]) -> 'DataManager':
        """Generates a new DataManager Object."""
        
        # Create the Frame asynchronously (assuming DataManager can be async)
        self.stock_frame = DataManager(data=data)
        
        return self.stock_frame

    async def execute_signals(self, signals: List[pd.Series], trades_to_execute: dict) -> List[dict]:
        """Executes the specified trades for each signal."""
        
        buys: pd.Series = signals["buys"]
        sells: pd.Series = signals["sells"]

        order_responses = []

        if not buys.empty:
            symbols_list = buys.index.get_level_values(0).to_list()

            for symbol in symbols_list:
                if symbol in trades_to_execute:
                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(symbol=symbol, ownership=True)

                    trades_to_execute[symbol]["has_executed"] = True
                    trade_obj: Trade = trades_to_execute[symbol]["buy"]["trade_func"]

                    if not self.paper_trading:
                        order_response = await self.execute_orders(trade_obj=trade_obj)
                        order_responses.append(order_response)
                    else:
                        order_response = {
                            "order_id": trade_obj._generate_order_id(),
                            "request_body": trade_obj.order,
                            "timestamp": datetime.now().isoformat(),
                        }
                        order_responses.append(order_response)

        if not sells.empty:
            symbols_list = sells.index.get_level_values(0).to_list()

            for symbol in symbols_list:
                if symbol in trades_to_execute:
                    trades_to_execute[symbol]["has_executed"] = True

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(symbol=symbol, ownership=False)

                    trade_obj: Trade = trades_to_execute[symbol]["sell"]["trade_func"]

                    if not self.paper_trading:
                        order_response = await self.execute_orders(trade_obj=trade_obj)
                        order_responses.append(order_response)
                    else:
                        order_response = {
                            "order_id": trade_obj._generate_order_id(),
                            "request_body": trade_obj.order,
                            "timestamp": datetime.now().isoformat(),
                        }
                        order_responses.append(order_response)

        # Save responses after execution
        await self.save_orders(order_responses)
        return order_responses

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    async def execute_orders(self, trade_obj: Backtester) -> dict:
        """Executes a Backtester Object with retry logic."""
        try:
            order_dict = await self.api.place_order(account=self.trading_account, order=trade_obj.order)
            trade_obj._order_response = order_dict
            trade_obj._process_order_response()
            return order_dict
        except Exception as e:
            print(f"Error while placing order: {e}")
            raise

    async def save_orders(self, order_response_dict: dict) -> bool:
        """Save orders asynchronously to avoid blocking."""
        async with aiofiles.open('data/orders.json', mode='w') as order_json:
            await order_json.write(json.dumps(order_response_dict, indent=4))
        return True

    async def get_accounts(self, account_number: str = None, all_accounts: bool = False) -> dict:
        """Fetch accounts concurrently."""
        
        accounts = await self.api.get_accounts(account=account_number if not all_accounts else "all")
        accounts_parsed = self._parse_account_balances(accounts_response=accounts)
        return accounts_parsed

    async def get_positions(self, account_number: str = None, all_accounts: bool = False) -> List[Dict]:
        """Fetch positions concurrently."""
        
        positions = await self.api.get_positions(account=account_number if not all_accounts else "all")
        positions_parsed = self._parse_account_positions(positions_response=positions)
        return positions_parsed

    def _parse_account_balances(self, accounts_response: Union[Dict, List]) -> List[Dict]:
        """Parses an Account response into a more simplified dictionary."""
        
        account_lists = []
        if isinstance(accounts_response, dict):
            account_dict = {}
            for account_type_key in accounts_response:
                account_info = accounts_response[account_type_key]
                account_id = account_info["accountId"]
                account_type = account_info["type"]
                account_current_balances = account_info["currentBalances"]

                account_dict["account_number"] = account_id
                account_dict["account_type"] = account_type
                account_dict["cash_balance"] = account_current_balances["cashBalance"]
                account_dict["long_market_value"] = account_current_balances["longMarketValue"]
                account_dict["cash_available_for_trading"] = account_current_balances.get("cashAvailableForTrading", 0.0)
                account_dict["cash_available_for_withdrawl"] = account_current_balances.get("cashAvailableForWithDrawal", 0.0)
                account_dict["available_funds"] = account_current_balances.get("availableFunds", 0.0)
                account_dict["buying_power"] = account_current_balances.get("buyingPower", 0.0)
                account_dict["day_trading_buying_power"] = account_current_balances.get("dayTradingBuyingPower", 0.0)
                account_dict["maintenance_call"] = account_current_balances.get("maintenanceCall", 0.0)
                account_dict["maintenance_requirement"] = account_current_balances.get("maintenanceRequirement", 0.0)
                account_dict["short_balance"] = account_current_balances.get("shortBalance", 0.0)
                account_dict["short_market_value"] = account_current_balances.get("shortMarketValue", 0.0)
                account_dict["short_margin_value"] = account_current_balances.get("shortMarginValue", 0.0)

                account_lists.append(account_dict)

        elif isinstance(accounts_response, list):
            for account in accounts_response:
                for account_type_key in account:
                    account_info = account[account_type_key]
                    account_id = account_info["accountId"]
                    account_type = account_info["type"]
                    account_current_balances = account_info["currentBalances"]

                    account_dict = {
                        "account_number": account_id,
                        "account_type": account_type,
                        "cash_balance": account_current_balances["cashBalance"],
                        "long_market_value": account_current_balances["longMarketValue"],
                        "cash_available_for_trading": account_current_balances.get("cashAvailableForTrading", 0.0),
                        "cash_available_for_withdrawl": account_current_balances.get("cashAvailableForWithDrawal", 0.0),
                        "available_funds": account_current_balances.get("availableFunds", 0.0),
                        "buying_power": account_current_balances.get("buyingPower", 0.0),
                        "day_trading_buying_power": account_current_balances.get("dayTradingBuyingPower", 0.0),
                        "maintenance_call": account_current_balances.get("maintenanceCall", 0.0),
                        "maintenance_requirement": account_current_balances.get("maintenanceRequirement", 0.0),
                        "short_balance": account_current_balances.get("shortBalance", 0.0),
                        "short_market_value": account_current_balances.get("shortMarketValue", 0.0),
                        "short_margin_value": account_current_balances.get("shortMarginValue", 0.0),
                    }

                    account_lists.append(account_dict)

        return account_lists

    def _parse_account_positions(self, positions_response: Union[List, Dict]) -> List[Dict]:
        """Parses the response from the `get_positions` into a more simplified list."""
        
        positions_lists = []

        if isinstance(positions_response, dict):
            for account_type_key in positions_response:
                account_info = positions_response[account_type_key]
                account_id = account_info["accountId"]
                positions = account_info["positions"]

                for position in positions:
                    position_dict = {
                        "account_number": account_id,
                        "average_price": position["averagePrice"],
                        "market_value": position["marketValue"],
                        "current_day_profit_loss_percentage": position["currentDayProfitLossPercentage"],
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
                    positions_lists.append(position_dict)

        elif isinstance(positions_response, list):
            for account in positions_response:
                for account_type_key in account:
                    account_info = account[account_type_key]
                    account_id = account_info["accountId"]
                    positions = account_info["positions"]

                    for position in positions:
                        position_dict = {
                            "account_number": account_id,
                            "average_price": position["averagePrice"],
                            "market_value": position["marketValue"],
                            "current_day_profit_loss_percentage": position["currentDayProfitLossPercentage"],
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
                        positions_lists.append(position_dict)

        return positions_lists

    async def start(self):
        """Starts the Forex bot by running the main trading loop asynchronously."""
        self.running = True
        self.logger.info("ForexBot started.")

        # Start the required tasks like monitoring, strategy execution, etc.
        await asyncio.gather(
            self.monitor.start(),
            self.strategy_manager.run(),
        )

    async def run(self):
        """Main bot execution loop."""
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
        """Subscribe to tick stream for a given symbol."""
        try:
            tick_stream = await self.api.subscribe({"ticks": symbol, "subscribe": 1})
            self.subscriptions[symbol] = tick_stream
            tick_stream.subscribe(self.create_tick_callback(symbol))
            self.logger.info(f"Subscribed to {symbol} tick stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")

    def create_tick_callback(self, symbol):
        """Create a callback function for processing tick data."""
        count = 0

        def callback(data):
            nonlocal count
            count += 1
            self.data_manager.update(symbol, data)  # Update the DataManager with tick data
            self.logger.info(f"Received tick for {symbol}: {data}. Count: {count}")

        return callback

    async def check_trades(self):
        """Check if any trade conditions are met and execute if necessary."""
        for symbol in self.config.SYMBOLS:
            if self.strategy_manager.should_enter_trade(symbol, self.data_manager):
                await self.enter_trade(symbol)

        # Monitor open positions and update portfolio
        await self.monitor.check_open_positions(self.api)
        await self.performance_monitor.update(self.portfolio_manager)

    async def enter_trade(self, symbol):
        """Enter a trade if the risk manager allows it."""
        if not self.risk_manager.can_enter_trade(symbol):
            self.logger.warning(f"Cannot enter trade for {symbol}: risk management rules not satisfied.")
            return

        try:
            position_size = self.risk_manager.calculate_position_size(symbol)
            contract = await self.api.buy({
                "contract_type": "CALL",
                "amount": position_size,
                "symbol": symbol,
                "duration": 5,
                "duration_unit": "m",
            })
            self.monitor.add_position(contract)
            self.portfolio_manager.add_trade(symbol, contract)
            self.logger.info(f"Entered trade: {contract}")
        except Exception as e:
            self.logger.error(f"Error entering trade for {symbol}: {e}")

    async def unsubscribe(self):
        """Unsubscribe from all tick streams."""
        for symbol, subscription in self.subscriptions.items():
            try:
                await self.api.forget(subscription["subscription"]["id"])
                self.logger.info(f"Unsubscribed from {symbol} tick stream.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {symbol}: {e}")
        self.subscriptions.clear()

    async def initialize_data_manager(self, config, api):
        """Initialize DataManager with historical data."""
        start_date = datetime.today()
        end_date = start_date - timedelta(days=config.HISTORICAL_DAYS)
        historical_data = []

        bar_size = config.BAR_SIZE if hasattr(config, "BAR_SIZE") else 1  # Default to 1 if not specified
        bar_type = config.TIMEFRAME

        for symbol in config.SYMBOLS:
            data = await self.grab_historical_prices(
                start_date, end_date, bar_size, bar_type, [symbol]
            )
            historical_data.extend(data)

        return DataManager(config, historical_data, config.SYMBOLS)

    def stop(self, api):
        """Stop the bot."""
        self.running = False
        asyncio.create_task(api.unsubscribe())  # Ensure unsubscription happens asynchronously
        self.logger.info("Stopping the bot...")

    async def run_backtest(self):
        """Run backtest on the strategy."""
        results = await self.backtester.run()
        self.logger.info(f"Backtest results: {results}")
        return results

