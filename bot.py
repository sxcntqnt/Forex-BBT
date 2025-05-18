import json
import asyncio
import pathlib
from datetime import datetime, timezone, timedelta, UTC
from typing import List, Dict, Union, Optional

import pandas as pd
from tenacity import retry, wait_fixed, stop_after_attempt
import aiofiles
from deriv_api import DerivAPI

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
        api: DerivAPI,
        paper_trading: bool = True,
    ):
        """Central trading bot with dependency injection."""
        self.config = config
        self.paper_trading = paper_trading
        self.logger = logger
        self.api = api
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)
        self.backtester = Backtester(config, api, data_manager, strategy_manager)
        self.monitor = Monitor(config, self.backtester)
        self.performance_monitor = PerformanceMonitor(self.portfolio_manager)
        self.subscriptions: Dict[str, dict] = {}
        self.active_trades: Dict[str, dict] = {}
        self.trades: Dict[str, Backtester] = {}
        self.running = False
        self.trading_account = "DEMO123"
        self.timestamp_utils = TimestampUtils()
        self.start_date = self.timestamp_utils.from_seconds(
            self.timestamp_utils.to_seconds(datetime.now(tz=UTC))
            - config.HISTORICAL_DAYS * 86400
        )
        self.end_date = datetime.now(tz=UTC)

    async def initialize(self):
        """Initialize bot, assuming DataManager and API are pre-configured."""
        self.logger.info("ForexBot initialized successfully.")

    async def run(self):
        """Run the bot's main loop."""
        self.running = True
        self.logger.info("ForexBot started running.")
        await self.initialize()
        while self.running:
            if self.is_market_open():
                await self.check_trades()
                await self.performance_monitor.update(self.portfolio_manager)
            else:
                self.logger.debug("Market closed, skipping trade checks.")
            await asyncio.sleep(1)

    def is_market_open(self, session: str = "regular") -> bool:
        """Check if the market is open for the specified session."""
        now = datetime.now(tz=UTC)
        if now.weekday() >= 5 and (now.weekday() == 5 and now.hour < 17):  # Close Friday 17:00 UTC
            self.logger.debug("Market closed on weekends.")
            return False
        return True  # Forex market is 24/5

    async def create_portfolio(self) -> PortfolioManager:
        """Create and initialize portfolio."""
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
        """Create a new trade."""
        self.logger.info(f"Creating trade {trade_id} - {enter_or_exit} {long_or_short} order.")
        trade = Backtester(self.config, self.api, self.data_manager, self.strategy_manager)
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
        self.logger.info(f"Trade created: {trade_id}")
        return trade

    async def delete_trade(self, trade_id: str) -> None:
        """Delete a trade by ID."""
        if trade_id in self.trades:
            del self.trades[trade_id]
            self.logger.info(f"Trade {trade_id} deleted.")
        else:
            self.logger.warning(f"Trade {trade_id} not found.")

    async def grab_current_quotes(self) -> Dict[str, Dict]:
        """Fetch current quotes for portfolio symbols."""
        self.logger.info("Fetching current quotes.")
        symbols = list(self.portfolio_manager.positions.keys())
        if not symbols:
            self.logger.debug("No symbols in portfolio to fetch quotes.")
            return {}
        try:
            quotes = await self.api.active_symbols({"active_symbols": "brief"})
            result = {s["symbol"]: s for s in quotes["active_symbols"] if s["symbol"] in symbols}
            self.logger.info(f"Fetched quotes for {len(result)} symbols.")
            return result
        except Exception as e:
            self.logger.error(f"Error fetching quotes: {e}")
            return {}

    async def execute_signals(self, signals: Dict[str, pd.Series], trades_to_execute: Dict) -> List[Dict]:
        """Execute trades based on signals."""
        order_responses = []
        for signal_type, signal_data in [("buys", signals.get("buys", pd.Series())), ("sells", signals.get("sells", pd.Series()))]:
            if signal_data.empty:
                continue
            symbols = signal_data.index.get_level_values(0).unique()
            for symbol in symbols:
                if symbol not in trades_to_execute or trades_to_execute[symbol].get("has_executed"):
                    continue
                trade_info = trades_to_execute[symbol][signal_type]
                trade_obj = trade_info["trade_func"]
                try:
                    order_response = await self._execute_trade(trade_obj)
                    order_responses.append(order_response)
                    trades_to_execute[symbol]["has_executed"] = True
                    self.portfolio_manager.set_ownership_status(symbol, signal_type == "buys")
                    self.logger.info(f"Executed {signal_type} trade for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to execute {signal_type} trade for {symbol}: {e}")
        if order_responses:
            await self.save_orders(order_responses)
        return order_responses

    async def _execute_trade(self, trade_obj: Backtester) -> Dict:
        """Execute a single trade."""
        try:
            order_id = trade_obj._generate_order_id() if self.paper_trading else None
            if not self.paper_trading:
                order_dict = await self.api.place_order(account=self.trading_account, order=trade_obj.order)
                trade_obj._order_response = order_dict
                trade_obj._process_order_response()
                order_id = order_dict["order_id"]
            return {
                "order_id": order_id,
                "request_body": trade_obj.order,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            raise

    async def save_orders(self, order_response_dict: List[Dict]) -> bool:
        """Save order responses to JSON file."""
        folder = pathlib.Path(__file__).parents[1] / "data"
        folder.mkdir(exist_ok=True)
        file_path = folder / "orders.json"
        try:
            orders_list = []
            if file_path.exists():
                async with aiofiles.open(file_path, mode="r") as f:
                    content = await f.read()
                    if content:
                        orders_list = json.loads(content)
            orders_list.extend(order_response_dict)
            async with aiofiles.open(file_path, mode="w") as f:
                await f.write(json.dumps(orders_list, indent=4))
            self.logger.info(f"Saved {len(order_response_dict)} orders to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving orders: {e}")
            return False

    async def get_account_balances(self) -> List[Dict]:
        """Fetch account balances."""
        try:
            accounts_response = await self.api.get_accounts()
            return self._parse_response(accounts_response, self._extract_account_data)
        except Exception as e:
            self.logger.error(f"Error fetching account balances: {e}")
            return []

    async def get_positions(self, account_number: str = None, all_accounts: bool = False) -> List[Dict]:
        """Fetch account positions."""
        account = "all" if all_accounts else (account_number or self.trading_account)
        try:
            positions = await self.api.get_accounts(account=account, fields=["positions"])
            return self._parse_response(positions, self._extract_position_data)
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []

    def _parse_response(self, response: Union[List, Dict], extract_func) -> List[Dict]:
        """Generic parser for account or position responses."""
        result = []
        if isinstance(response, dict):
            for key, info in response.items():
                item = extract_func(info, info.get("accountId"))
                result.append(item)
        elif isinstance(response, list):
            for account in response:
                for key, info in account.items():
                    if "positions" in info:
                        for position in info["positions"]:
                            item = extract_func(position, info["accountId"])
                            result.append(item)
                    else:
                        item = extract_func(info, info["accountId"])
                        result.append(item)
        return result

    def _extract_account_data(self, account_info: Dict, account_id: str) -> Dict:
        """Extract account balance data."""
        return {
            "account_number": account_id,
            "account_type": account_info["type"],
            "cash_balance": account_info["currentBalances"]["cashBalance"],
            "long_market_value": account_info["currentBalances"]["longMarketValue"],
            "cash_available_for_trading": account_info["currentBalances"].get("cashAvailableForTrading", 0.0),
            "cash_available_for_withdrawal": account_info["currentBalances"].get("cashAvailableForWithDrawal", 0.0),
            "available_funds": account_info["currentBalances"].get("availableFunds", 0.0),
            "buying_power": account_info["currentBalances"].get("buyingPower", 0.0),
            "day_trading_buying_power": account_info["currentBalances"].get("dayTradingBuyingPower", 0.0),
            "maintenance_call": account_info["currentBalances"].get("maintenanceCall", 0.0),
            "maintenance_requirement": account_info["currentBalances"].get("maintenanceRequirement", 0.0),
            "short_balance": account_info["currentBalances"].get("shortBalance", 0.0),
            "short_market_value": account_info["currentBalances"].get("shortMarketValue", 0.0),
            "short_margin_value": account_info["currentBalances"].get("shortMarginValue", 0.0),
        }

    def _extract_position_data(self, position: Dict, account_id: str) -> Dict:
        """Extract position data."""
        return {
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

    async def check_trades(self):
        """Check trade conditions and execute trades."""
        for symbol in self.config.SYMBOLS:
            try:
                if await self.strategy_manager.should_enter_trade(symbol):
                    await self.enter_trade(symbol)
                if symbol in self.active_trades and await self.strategy_manager.should_exit_trade(symbol):
                    await self.exit_trade(symbol)
            except Exception as e:
                self.logger.error(f"Error checking trades for {symbol}: {e}")
        await self.monitor.check_open_positions(self.api)

    async def enter_trade(self, symbol: str, contract_type: str = "CALL"):
        """Enter a new trade."""
        if not self.is_market_open():
            self.logger.warning(f"Cannot enter trade for {symbol}: Market closed.")
            return
        if not self.risk_manager.can_enter_trade(symbol):
            self.logger.warning(f"Risk limit reached for {symbol}")
            return
        try:
            position_size = self.risk_manager.calculate_position_size(symbol)
            contract = await self.api.buy(
                {
                    "contract_type": contract_type,
                    "amount": position_size,
                    "symbol": symbol,
                    "duration": 5,
                    "duration_unit": "m",
                }
            )
            self.monitor.add_position(contract)
            self.portfolio_manager.add_trade(symbol, contract)
            self.active_trades[symbol] = {"contract_id": contract["contract_id"]}
            self.logger.info(f"Entered trade for {symbol}: {contract}")
        except Exception as e:
            self.logger.error(f"Failed to enter trade for {symbol}: {e}")

    async def exit_trade(self, symbol: str):
        """Exit an active trade."""
        if symbol not in self.active_trades:
            self.logger.debug(f"No active trade for {symbol}.")
            return
        trade = self.active_trades[symbol]
        try:
            response = await self.api.sell({"contract_id": trade["contract_id"]})
            if response.get("error"):
                self.logger.error(f"Error exiting trade for {symbol}: {response['error']}")
            else:
                self.logger.info(f"Exited trade for {symbol}: {response}")
                del self.active_trades[symbol]
        except Exception as e:
            self.logger.error(f"Exception exiting trade for {symbol}: {e}")

    async def stop(self):
        """Stop the bot and clean up."""
        self.running = False
        await self.data_manager.stop_subscriptions()
        self.logger.info("ForexBot stopped.")

    async def run_backtest(self):
        """Run a backtest and return results."""
        try:
            results = await self.backtester.run()
            self.logger.info(f"Backtest results: {results}")
            return results
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {}
