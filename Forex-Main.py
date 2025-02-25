import asyncio
import logging
import sys
import os
import tracemalloc
import time

from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Union, Tuple
from functools import partial

from config import Config
from data_manager import DataManager
from deriv_api import DerivAPI
from bot import ForexBot
from utils import TimestampUtils
from strategy import StrategyManager
from web_interface import start_web_interface

# Configure logging with process ID
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_debug.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Main")
logger.debug("Logging initialized in PID %d", os.getpid())


async def main() -> Tuple[Dict[str, Union[List[Dict], Dict]], "ForexBot", DerivAPI]:
    """Main function to start the bot with multiple tasks and return historical data."""
    logger.info("Starting main() in process %d", os.getpid())
    tracemalloc.start()
    config = Config()
    bot = None
    api = None
    tasks = []
    historical_data = None

    try:
        logger.info("Initialized config with symbols: %s", config.SYMBOLS)

        logger.debug("Initializing DerivAPI...")
        api = DerivAPI(app_id=config.APP_ID)
        response = await api.ping({"ping": 1})
        logger.debug("Ping response: %s", response)
        if response.get("ping") != "pong":
            raise ConnectionError("API ping failed")

        data_manager = DataManager(config=config, api=api, logger=logger)
        strategy_manager = StrategyManager(data_manager=data_manager, api=api, logger=logger)

        bot = ForexBot(
            config=config,
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            logger=logger,
            api=api,
        )

        # Authorization
        try:
            auth_response = await asyncio.wait_for(
                api.authorize({"authorize": config.DERIV_API_TOKEN}), timeout=15
            )
            if not auth_response.get("authorize", {}).get("loginid"):
                raise ConnectionError("Authorization failed")
            logger.info("Authorization successful")
        except asyncio.TimeoutError:
            logger.error("Authorization timed out")
            raise
        except Exception as e:
            logger.error("Authorization failed: %s", str(e))
            raise

        # Fetching historical data
        timestamp_utils = TimestampUtils()
        start_ts = timestamp_utils.to_seconds(datetime.now(tz=timezone.utc)) - 86400  # 1 day ago
        end_ts = timestamp_utils.to_seconds(datetime.now(tz=timezone.utc))

        logger.info("Fetching historical prices for frxEURUSD...")
        try:
            # Fetch historical data for all symbols in config.SYMBOLS
            for symbol in config.SYMBOLS:
                raw_data, prices = await asyncio.wait_for(
                    data_manager.grab_historical_data(start_ts, end_ts, symbol), timeout=60
                )
                historical_data[symbol] = raw_data  # Store raw data per symbol
                logger.debug("Raw historical data for %s: %s", symbol, raw_data)

                if not raw_data or "candles" not in raw_data or not raw_data["candles"]:
                    logger.warning("No candles data for %s: %s", symbol, raw_data)
                else:
                    # Convert epoch to datetime for logging (optional)
                    first_candle = raw_data["candles"][0]
                    first_candle["datetime"] = datetime.fromtimestamp(first_candle["epoch"]).isoformat()
                    logger.info("First candle datetime for %s: %s", symbol, first_candle["datetime"])

        except asyncio.TimeoutError:
            logger.error("Historical data fetch timed out after 60 seconds")
            historical_data = None
        except Exception as e:
            logger.error("Error fetching historical data: %s, type: %s", str(e), type(e).__name__, exc_info=True)
            historical_data = None

        # Starting ForexBot with multiple background tasks
        logger.info("Starting ForexBot with multiple background tasks...")
        tasks = [
            asyncio.create_task(bot.run(), name="bot_run"),
            asyncio.create_task(bot.data_manager.start_subscriptions(), name="data_subs"),
            asyncio.create_task(bot.initialize_data_manager(), name="data_init"),
            asyncio.create_task(start_web_interface(bot), name="web_interface"),
            asyncio.create_task(run_blocking_tasks(bot, config), name="blocking_tasks"),
            asyncio.create_task(event_loop_watchdog(config), name="watchdog"),
        ]
        await asyncio.sleep(2)  # Allow tasks to start

        logger.info("All tasks scheduled")
        return historical_data, bot, api  # Return the expected values

    except Exception as e:
        logger.critical("Critical error in main: %s", str(e), exc_info=True)
        for task in tasks:
            task.cancel()  # Cancel all tasks
            try:
                await task  # Await cancellation
            except asyncio.CancelledError:
                logger.debug("Task %s cancelled during error cleanup", task.get_name())
        if bot:
            await bot.stop()  # Ensure the bot is stopped
        if api:
            await api.clear()  # Clear the API
        raise  # Re-raise the exception for further handling

    finally:
        tracemalloc.stop()

async def event_loop_watchdog(config: Config):
    """Monitors event loop health indefinitely."""
    logger.info("Watchdog started in process %d", os.getpid())
    while True:
        start_time = time.monotonic()
        await asyncio.sleep(config.WATCHDOG_INTERVAL)
        delay = time.monotonic() - start_time - config.WATCHDOG_INTERVAL
        if delay > config.STARVATION_THRESHOLD:
            logger.warning("Event loop starvation detected! Delay: %.2fs", delay)
        current_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        logger.debug("Active tasks: %d", len(current_tasks))

async def run_blocking_tasks(bot: ForexBot, config: Config):
    """Run CPU-intensive tasks in executor indefinitely."""
    logger.info("Blocking tasks started in process %d", os.getpid())
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
    symbols = config.SYMBOLS.split(",") if isinstance(config.SYMBOLS, str) else config.SYMBOLS
    if symbols:
        bot.create_tick_callback(symbols[0].strip())


async def stop_bot(bot: ForexBot, api: DerivAPI):
    logger.info("Stopping bot")
    await bot.stop()
    await api.clear()
    logger.info("Bot stopped")


if __name__ == "__main__":
    logger.info("Main process started with PID %d", os.getpid())
    try:
        async def run_with_cleanup():
            try:
                result = await main()  # Call the main function
                logger.debug(f"Main returned: {result}")  # Log the result for debugging
                return result
            except Exception as e:
                logger.error(f"Error in main function: {str(e)}")
                return None  # Return None or a default value in case of an error

        # Unpack the result safely
        result = asyncio.run(run_with_cleanup(), debug=True)

        # Check if result is valid before unpacking
        if result is None or len(result) != 3:
            logger.critical("Unexpected result from main function. Expected 3 values, got: %s", result)
            sys.exit(1)

        historical_data, bot, api = result  # Unpack the result safely

        logger.info("Main completed")
        if historical_data and any(historical_data.values()):
            for symbol, data in historical_data.items():
                if data and "candles" in data and data["candles"]:
                    logger.info("First candle for %s: %s", symbol, data["candles"][0])
        else:
            logger.warning("No historical data returned from main()")

        logger.info("Bot running in background. Press Ctrl+C to stop.")
        loop = asyncio.get_event_loop()
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")
            asyncio.run(stop_bot(bot, api))  # Fixed syntax: pass as separate args
            loop.close()

    except Exception as e:
        logger.critical("Application bootstrap failed: %s", str(e))
        sys.exit(1)
