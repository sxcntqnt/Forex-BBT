import asyncio
import json
import pathlib
from datetime import datetime, timezone, UTC
from typing import Dict, List, Optional
import pandas as pd
import logging
from deriv_api import DerivAPI
from config import Config
from strategy import StrategyManager
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from monitor import Monitor
from data_manager import DataManager
from utils import TimestampUtils
from backtester import Backtester

class ForexBot:
    def __init__(
        self,
        config: Config,
        data_manager: DataManager,
        strategy_manager: Optional[StrategyManager] = None,
        logger=None,
        api: Optional[DerivAPI] = None,
        paper_trading: bool = True,
    ):
        self.config = config
        self.paper_trading = paper_trading
        self.logger = logger or logging.getLogger("ForexBot")
        self.logger.debug("Starting ForexBot initialization")
        self.api = api or DerivAPI(app_id=config.APP_ID)
        self.logger.debug("DerivAPI initialized")
        self.data_manager = data_manager
        self.logger.debug("DataManager assigned")
        self.portfolio_manager = PortfolioManager(config, data_manager, self.api, self.logger, account_number="DEMO123")
        self.logger.debug("PortfolioManager initialized")
        self.strategy_manager = strategy_manager or StrategyManager(data_manager, self.api, self.logger, self.portfolio_manager)
        self.logger.debug(f"StrategyManager assigned, has rsi_strategy: {hasattr(self.strategy_manager, 'rsi_strategy')}")
        self.risk_manager = RiskManager(config)
        self.logger.debug("RiskManager initialized")
        self.monitor = Monitor(config, self.portfolio_manager, self.api, self.logger)
        self.logger.debug("Monitor initialized")
        try:
            self.backtester = Backtester(config, self.api, data_manager, self.strategy_manager, self.logger)
            self.logger.debug("Backtester initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Backtester: {e}", exc_info=True)
            raise
        self.active_trades: Dict[str, Dict] = {}
        self.running = False
        self.trading_account = "DEMO123"
        self.timestamp_utils = TimestampUtils()
        self.start_date = self.timestamp_utils.from_seconds(
            self.timestamp_utils.to_seconds(datetime.now(tz=UTC)) - config.historical_days * 86400
        )
        self.end_date = datetime.now(tz=UTC)
        self.logger.debug("ForexBot initialization completed")

    async def initialize(self):
        """Initialize the bot and validate symbols."""
        invalid_symbols = [s for s in self.config.symbols if not s.startswith('frx')]
        if invalid_symbols:
            self.logger.error(f"Invalid symbols in config: {invalid_symbols}")
            raise ValueError(f"Invalid symbols: {invalid_symbols}")
        await self.data_manager.initialize()
        self.logger.info("ForexBot initialized successfully.")

    async def run(self):
        """Run the bot's main loop."""
        self.running = True
        self.logger.info("ForexBot started running.")
        await self.initialize()
        backtest_results = await self.backtester.run()
        self.logger.info(f"Backtest results: {backtest_results}")
        while self.running:
            if self.is_market_open():
                await self.check_trades()
                await self.monitor.performance_monitor.update()
            else:
                self.logger.debug("Market closed, skipping trade checks.")
            await asyncio.sleep(1)

    async def run_backtest(self):
        """Run backtests using both stratestic and backtesting libraries."""
        return await self.backtester.run()

    def is_market_open(self) -> bool:
        """Check if the forex market is open (24/5, Sunday 17:00 UTC to Friday 17:00 UTC)."""
        now = datetime.now(tz=UTC)
        if now.weekday() == 5 or (now.weekday() == 6 and now.hour < 17):
            self.logger.debug("Market closed on weekends.")
            return False
        return True

    async def enter_trade(self, symbol: str, contract_type: str = "CALL"):
        """Enter a new trade."""
        if not self.is_market_open():
            self.logger.warning(f"Cannot enter trade for {symbol}: Market closed.")
            return
        if not self.risk_manager.can_enter_trade(symbol):
            self.logger.warning(f"Risk limit reached for {symbol}")
            return
        try:
            snapshot = self.data_manager.get_snapshot(symbol)
            if snapshot is None or snapshot.empty:
                self.logger.warning(f"No data for {symbol}")
                return
            current_price = snapshot['close'].iloc[-1]
            position_size = self.risk_manager.calculate_position_size(symbol)
            contract = {
                "contract_type": contract_type,
                "amount": position_size,
                "symbol": symbol,
                "duration": 5,
                "duration_unit": "m",
                "buy_price": current_price,
                "contract_id": f"sim_{symbol}_{int(datetime.now(tz=UTC).timestamp())}" if self.paper_trading else None
            }
            if not self.paper_trading:
                response = await self.api.buy(contract)
                contract = response.get("buy", {})
                contract["contract_id"] = response.get("contract_id")
                contract["buy_price"] = contract.get("buy_price", current_price)
            self.portfolio_manager.add_trade(symbol, contract)
            self.active_trades[symbol] = {"contract_id": contract["contract_id"], "buy_price": contract["buy_price"], "amount": position_size}
            self.logger.info(f"Entered trade for {symbol}: contract_id={contract['contract_id']}")
        except Exception as e:
            self.logger.error(f"Failed to enter trade for {symbol}: {e}", exc_info=True)

    async def exit_trade(self, symbol: str):
        """Exit an active trade."""
        if symbol not in self.active_trades:
            self.logger.debug(f"No active trade for {symbol}.")
            return
        trade = self.active_trades[symbol]
        contract_id = trade["contract_id"]
        try:
            if self.paper_trading:
                snapshot = self.data_manager.get_snapshot(symbol)
                profit = 0.0
                if snapshot is not None and not snapshot.empty:
                    current_price = snapshot['close'].iloc[-1]
                    entry_price = trade.get('buy_price', current_price)
                    profit = (current_price - entry_price) * trade.get('amount', 0)
                self.monitor.performance_monitor.record_trade(profit)
                self.portfolio_manager.close_trade(symbol, contract_id)
                del self.active_trades[symbol]
                self.logger.info(f"Simulated exit for {symbol}: contract_id={contract_id}, profit=${profit:.2f}")
            else:
                response = await self.api.sell({"contract_id": contract_id})
                if response.get("error"):
                    self.logger.error(f"Error exiting trade for {symbol}: {response['error']}")
                else:
                    profit = response.get("profit", 0.0)
                    self.monitor.performance_monitor.record_trade(profit)
                    self.portfolio_manager.close_trade(symbol, contract_id)
                    del self.active_trades[symbol]
                    self.logger.info(f"Exited trade for {symbol}: contract_id={contract_id}, profit=${profit:.2f}")
        except Exception as e:
            self.logger.error(f"Exception exiting trade for {symbol}: {e}", exc_info=True)

    async def check_trades(self):
        """Check trade conditions and manage positions."""
        for symbol in self.config.symbols:
            try:
                can_enter = self.risk_manager.can_enter_trade(symbol)
                should_enter = await self.strategy_manager.should_enter_trade(symbol)
                self.logger.debug(f"{symbol}: can_enter={can_enter}, should_enter={should_enter}")
                if can_enter and should_enter:
                    await self.enter_trade(symbol)
                if symbol in self.active_trades:
                    should_exit = await self.strategy_manager.should_exit_trade(symbol)
                    self.logger.debug(f"{symbol}: should_exit={should_exit}")
                    if should_exit:
                        await self.exit_trade(symbol)
            except Exception as e:
                self.logger.error(f"Error checking trades for {symbol}: {e}", exc_info=True)
        await self.monitor.check_open_positions()

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
            self.logger.error(f"Error saving orders: {e}", exc_info=True)
            return False

    async def stop(self):
        """Stop the bot and clean up."""
        self.running = False
        await self.data_manager.stop_subscriptions()
        self.logger.info("ForexBot stopped.")
