import asyncio
import logging
from deriv_api import DerivAPI
from rx import Observable
from strategy import StrategyManager
from risk_manager import RiskManager
from monitor import Monitor
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from performance_monitor import PerformanceMonitor
from backtester import Backtester


import logging
import asyncio
from deriv_api import DerivAPI
from rx import Observable

class ForexBot:
    def __init__(self, config, connection):
        self.config = config
        self.api = DerivAPI(connection=connection)
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager(config)
        self.monitor = Monitor(config)
        self.data_manager : DataManager =None 
        self.portfolio_manager = PortfolioManager(config)
        self.performance_monitor = PerformanceMonitor()
        self.backtester = Backtester(config, self.api, self.data_manager, self.strategy_manager)
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.subscriptions = {}

    async def run(self):
        self.running = True
        await self.api.authorize({'authorize': self.config.API_TOKEN})
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
            tick_stream: Observable = await self.api.subscribe({'ticks': symbol, 'subscribe': 1})
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
            self.data_manager.update(symbol, data)
            self.logger.info(f"Received tick for {symbol}: {data}. Count: {count}")
        return callback

    async def check_trades(self):
        for symbol in self.config.SYMBOLS:
            if self.strategy_manager.should_enter_trade(symbol, self.data_manager):
                await self.enter_trade(symbol)

        await self.monitor.check_open_positions(self.api)
        await self.performance_monitor.update(self.portfolio_manager)

    async def enter_trade(self, symbol):
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
                "duration_unit": "m"
            })
            self.monitor.add_position(contract)
            self.portfolio_manager.add_trade(symbol, contract)
            self.logger.info(f"Entered trade: {contract}")
        except Exception as e:
            self.logger.error(f"Error entering trade for {symbol}: {e}")

    async def unsubscribe(self):
        for symbol, subscription in self.subscriptions.items():
            try:
                await self.api.forget(subscription['subscription']['id'])
                self.logger.info(f"Unsubscribed from {symbol} tick stream.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {symbol}: {e}")
        self.subscriptions.clear()

    def stop(self):
        self.running = False
        asyncio.create_task(self.unsubscribe())  # Ensure unsubscription happens asynchronously
        self.logger.info("Stopping the bot...")

    async def run_backtest(self):
        results = await self.backtester.run()
        self.logger.info(f"Backtest results: {results}")
        return results
