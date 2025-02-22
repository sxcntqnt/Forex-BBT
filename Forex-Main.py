import asyncio
import logging
import sys
import time
import tracemalloc
from contextlib import closing
from datetime import datetime, timezone
from functools import partial
from typing import Dict, List, Optional, Union

from config import Config
from data_manager import DataManager
from deriv_api import DerivAPI
from bot import ForexBot
from utils import TimestampUtils
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


async def main() -> Optional[Dict[str, Union[List[Dict], Dict]]]:
    """Main function with event loop starvation prevention and data return."""
    tracemalloc.start()
    config = None
    bot = None
    api = None
    tasks: List[asyncio.Task] = []
    historical_data = None

    try:
        # 1. Initialize configuration
        config = Config()
        logger.info("Initialized config with symbols: %s", config.SYMBOLS)

        # 2. Initialize the Deriv API object
        logger.debug("Initializing DerivAPI...")
        api = DerivAPI(
            endpoint=config.EndPoint,
            app_id=config.APP_ID,
            api_token=config.DERIV_API_TOKEN if config.DERIV_API_TOKEN else None,
        )
        response = await api.ping({"ping": 1})
        logger.debug("Ping response: %s", response)
        if response.get("ping") != "pong":
            raise ConnectionError("API ping failed")

        # 3. Initialize core components
        data_manager = DataManager(config=config, api=api, logger=logger)
        strategy_manager = StrategyManager(
            data_manager=data_manager, api=api, logger=logger
        )

        # 4. Create and configure bot
        bot = ForexBot(
            config=config,
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            logger=logger,
            api=api,
        )

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
            return None
        except Exception as e:
            logger.error("Authorization failed: %s", str(e))
            return None

        # 6. Fetch historical data (non-blocking test)
        timestamp_utils = TimestampUtils()
        start = timestamp_utils.from_seconds(
            timestamp_utils.to_seconds(datetime.now(tz=timezone.utc)) - 86400
        )
        end = datetime.now(tz=timezone.utc)
        logger.info("Fetching historical prices for frxEURUSD...")
        historical_data = await asyncio.wait_for(
            bot.grab_historical_prices(start, end, symbols=["frxEURUSD"]), timeout=30
        )
        if "frxEURUSD" in historical_data and historical_data["frxEURUSD"]["candles"]:
            logger.info(
                "First candle datetime: %s",
                historical_data["frxEURUSD"]["candles"][0]["datetime"],
            )
        else:
            logger.warning("No historical data fetched for frxEURUSD")

        # 7. Create tasks (limited runtime for testing)
        tasks = [
            asyncio.create_task(run_with_timeout(bot.run(), 60), name="main_loop"),
            asyncio.create_task(
                run_with_timeout(bot.data_manager.start_subscriptions(), 60),
                name="data_subs",
            ),
            asyncio.create_task(
                run_with_timeout(bot.initialize_data_manager(), 60), name="start_subs"
            ),
            asyncio.create_task(
                run_with_timeout(start_web_interface(bot), 60), name="web_interface"
            ),
            asyncio.create_task(
                run_with_timeout(run_blocking_tasks(bot, config), 60),
                name="blocking_tasks",
            ),
            asyncio.create_task(
                run_with_timeout(event_loop_watchdog(), 60), name="watchdog"
            ),
        ]

        # 8. Run tasks with overall timeout
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=config.MAX_RUNTIME
        )

    except asyncio.TimeoutError:
        logger.warning("Main execution timeout reached")
    except Exception as e:
        logger.critical("Critical error: %s", str(e), exc_info=True)
    finally:
        # 9. Cleanup
        logger.info("Initiating shutdown sequence...")
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug("Task %s cancelled", task.get_name())
        if bot:
            await bot.stop()
        if api:
            await api.clear()
        tracemalloc.stop()
        logger.info("Cleanup completed")
        return historical_data


async def run_with_timeout(coro, timeout: float):
    """Run a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Task %s timed out after %s seconds",
            asyncio.current_task().get_name(),
            timeout,
        )
        raise


async def event_loop_watchdog():
    """Monitors event loop health with a finite duration."""
    start_time_total = time.monotonic()
    while time.monotonic() - start_time_total < 60:  # Run for 60 seconds max
        start_time = time.monotonic()
        await asyncio.sleep(config.WATCHDOG_INTERVAL)
        delay = time.monotonic() - start_time - config.WATCHDOG_INTERVAL
        if delay > config.STARVATION_THRESHOLD:
            logger.warning("Event loop starvation detected! Delay: %.2fs", delay)
        current_tasks = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]
        logger.debug("Active tasks: %d", len(current_tasks))


async def run_blocking_tasks(bot: ForexBot, config: Config):
    """Run CPU-intensive tasks in executor with a finite duration."""
    loop = asyncio.get_event_loop()
    start_time_total = time.monotonic()
    while time.monotonic() - start_time_total < 60:  # Run for 60 seconds max
        try:
            await loop.run_in_executor(
                None, partial(process_blocking_operations, bot, config)
            )
            await asyncio.sleep(1)
        except Exception as e:
            logger.error("Blocking task error: %s", str(e))
            await asyncio.sleep(5)


def process_blocking_operations(bot: ForexBot, config: Config):
    """Process blocking operations."""
    symbols = (
        config.SYMBOLS.split(",") if isinstance(config.SYMBOLS, str) else config.SYMBOLS
    )
    if symbols:
        bot.create_tick_callback(symbols[0].strip())


if __name__ == "__main__":
    try:
        result = asyncio.run(main(), debug=True)
        if result:
            logger.info(
                "Historical data returned: %s",
                result.get("frxEURUSD", {}).get("candles", [])[:1],
            )
        else:
            logger.warning("No data returned from main()")
    except Exception as e:
        logger.critical("Application bootstrap failed: %s", str(e))
        sys.exit(1)
