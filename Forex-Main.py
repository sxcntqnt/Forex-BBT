import asyncio
import logging
import sys
import time
import tracemalloc
from functools import partial
from contextlib import closing
from datetime import datetime, timezone, timedelta

from deriv_api import DerivAPI
from bot import ForexBot
from config import Config
from data_manager import DataManager
from strategy import StrategyManager
from web_interface import start_web_interface


# Configure logging
with closing(logging.FileHandler("bot_debug.log")) as file_handler:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )
logger = logging.getLogger("Main")

class MockDerivAPI:
    """Mock DerivAPI for demonstration purposes."""
    async def ping(self, data):
        return {"ping": "pong"}
    
    async def authorize(self, data):
        return {"authorize": {"loginid": "DEMO123"}}
    
    async def clear(self):
        logger.info("API connection closed")

async def main():
    """Main function with event loop starvation prevention."""
    tracemalloc.start()
    config = None
    bot = None
    tasks = []

    try:
        # 1. Initialize configuration
        config = Config()
        logger.info("Initialized config with symbols: %s", config.SYMBOLS)

        # 2. Initialize the API object (mocked for this example)
        logger.debug("Initializing MockDerivAPI...")
        api = MockDerivAPI()
        response = await api.ping({'ping': 1})
        logger.debug("Ping response: %s", response)

        # 3. Initialize core components
        data_manager = DataManager(config=config, api=api, logger=logger)
        strategy_manager = StrategyManager(data_manager=data_manager, api=api, logger=logger)

        # 4. Create and configure bot
        bot = ForexBot(
            config=config,
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            logger=logger,
            api=api,
        )

        start = bot.timestamp_utils.from_seconds(bot.timestamp_utils.to_seconds(datetime.now(tz=timezone.utc)) - 86400)
        end = datetime.now(tz=timezone.utc)
        data = await bot.grab_historical_prices(start, end, symbols=["frxEURUSD"])
        print(data["frxEURUSD"]["candles"][0]["datetime"])
        # 5. Authorize with timeout
        try:
            auth_response = await asyncio.wait_for(
                api.authorize({"authorize": config.DERIV_API_TOKEN}), timeout=15
            )
            if not auth_response.get("authorize", {}).get("loginid"):
                raise ConnectionError("Authorization failed")
            logger.info("Authorization successful")
        except asyncio.TimeoutError:
            logger.error("Authorization timed out")
            return
        except Exception as e:
            logger.error("Authorization failed: %s", str(e))
            return

        # 6. Create tasks with watchdog
        tasks = [
            asyncio.create_task(bot.data_manager.start_subscriptions(), name="data_subs"),
            asyncio.create_task(bot.initialize_data_manager(), name="start_subs"),
            asyncio.create_task(bot.run(), name="main_loop"),
            asyncio.create_task(start_web_interface(bot), name="web_interface"),
            asyncio.create_task(run_blocking_tasks(bot, config), name="blocking_tasks"),
            asyncio.create_task(event_loop_watchdog(), name="watchdog"),
        ]

        # 7. Run with timeout protection
        await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=config.MAX_RUNTIME,
        )

    except asyncio.TimeoutError:
        logger.warning("Main execution timeout reached")
    except Exception as e:
        logger.critical("Critical error: %s", str(e), exc_info=True)
    finally:
        # 8. Cleanup
        logger.info("Initiating shutdown sequence...")
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if bot:
            await bot.stop()
        if api:
            await api.clear()
        tracemalloc.stop()
        logger.info("Cleanup completed")

async def event_loop_watchdog():
    """Monitors event loop health."""
    while True:
        start_time = time.monotonic()
        await asyncio.sleep(5)
        delay = time.monotonic() - start_time - 5
        if delay > 0.5:
            logger.warning("Event loop starvation detected! Delay: %.2fs", delay)
        current_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        logger.debug("Active tasks: %d", len(current_tasks))

async def run_blocking_tasks(bot: ForexBot, config: Config):
    """Run CPU-intensive tasks in executor."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            await loop.run_in_executor(None, partial(process_blocking_operations, bot, config))
            await asyncio.sleep(1)
        except Exception as e:
            logger.error("Blocking task error: %s", str(e))
            await asyncio.sleep(5)

def process_blocking_operations(bot: ForexBot, config: Config):
    """Process blocking operations."""
    bot.create_tick_callback(config.SYMBOLS)

if __name__ == "__main__":
    try:
        asyncio.run(main(), debug=True)
    except Exception as e:
        logger.critical("Application bootstrap failed: %s", str(e))
        sys.exit(1)
