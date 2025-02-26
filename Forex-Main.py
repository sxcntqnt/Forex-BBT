"""Main module for Forex trading bot."""
import pdb
import asyncio
import logging
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from functools import partial
from typing import Dict, List, Tuple, Union

from deriv_api import DerivAPI  # Third-party import first
from config import Config  # Then first-party imports
from data_manager import DataManager
from strategy import StrategyManager
from bot import ForexBot
from utils import TimestampUtils, PerformanceMonitor
from web_interface import start_web_interface

logger = logging.getLogger(__name__)

import nest_asyncio

nest_asyncio.apply()  # This needs to be called only once

async def main() -> Tuple[Dict[str, Union[List[Dict], Dict]], "ForexBot", DerivAPI]:
    """Main function to start the bot with multiple tasks and return historical data."""
    logger.info("Starting main() in process %d", os.getpid())
    tracemalloc.start()
    local_config = Config()
    local_bot = None
    local_api = None
    historical_data_result = {}

    try:
        logger.info(f"Type of local_config.SYMBOLS: {type(local_config.SYMBOLS)}")
        logger.info(f"Value of local_config.SYMBOLS: {local_config.SYMBOLS}")

        logger.info("Initialized config with symbols: %s", local_config.SYMBOLS)

        logger.debug("Initializing DerivAPI...")
        local_api = DerivAPI(app_id=local_config.APP_ID)
        response = await local_api.ping({"ping": 1})
        logger.debug("Ping response: %s", response)
        if response.get("ping") != "pong":
            raise ConnectionError("API ping failed")

        local_data_manager = DataManager(
            config=local_config, api=local_api, logger=logger
        )
        local_strategy_manager = StrategyManager(
            data_manager=local_data_manager, api=local_api, logger=logger
        )

        local_bot = ForexBot(
            config=local_config,
            data_manager=local_data_manager,
            strategy_manager=local_strategy_manager,
            logger=logger,
            api=local_api,
        )

        try:
            auth_response = await asyncio.wait_for(
                local_api.authorize({"authorize": local_config.DERIV_API_TOKEN}),
                timeout=15,
            )
            if not auth_response.get("authorize", {}).get("loginid"):
                raise ConnectionError("Authorization failed")
            logger.info("Authorization successful")
        except asyncio.TimeoutError:
            logger.error("Authorization timed out")
            raise
        except Exception as auth_error:
            logger.error("Authorization failed: %s", str(auth_error))
            raise

        timestamp_utils = TimestampUtils()
        start_date = datetime.fromisoformat(local_config.BACKTEST_START_DATE).replace(
            tzinfo=timezone.utc
        )
        end_date = datetime.fromisoformat(local_config.BACKTEST_END_DATE).replace(
            tzinfo=timezone.utc
        )

        start_ts = timestamp_utils.to_seconds(start_date)
        end_ts = timestamp_utils.to_seconds(end_date)

        logger.info("Fetching historical prices from %s to %s...", start_date, end_date)

        logger.info(f"local_config.SYMBOLS before loop: {local_config.SYMBOLS}") #added
        for local_symbol in local_config.SYMBOLS:
            logger.info("Fetching historical data for %s...", local_symbol)
            if not local_symbol: # Add this check
                logger.error("local_symbol is empty! Skipping.")
                continue  # Skip to the next iteration
            try:
                raw_data = await asyncio.wait_for(
                    local_data_manager.grab_historical_data(
                        start_ts, end_ts, local_symbol
                    ),
                    timeout=60,
                )
                if not raw_data.empty:
                    historical_data_result[local_symbol] = raw_data
                    logger.debug(
                        "Raw historical data for %s: %s", local_symbol, raw_data
                    )
                    first_candle = raw_data.iloc[0]

                    logger.debug(f"Type of first_candle: {type(first_candle)}")
                    logger.debug(f"Value of first_candle: {first_candle}")

                    pdb.set_trace()  # <---- ADD THIS LINE: Set a breakpoint HERE
                    first_candle_datetime = datetime.fromtimestamp(first_candle["timestamp"]).isoformat()
                    raw_data.at[raw_data.index[0], "datetime"] = first_candle_datetime # Add the date time into the raw_data
                    logger.info(
                        "First candle datetime for %s: %s",
                        local_symbol,
                        first_candle_datetime,
                    )
                else:
                    historical_data_result[local_symbol] = {"candles": []}
                    logger.info("No historical data for %s", local_symbol)

            except asyncio.TimeoutError:
                logger.error(
                    "Historical data fetch for %s timed out after 60 seconds",
                    local_symbol,
                    f"Type of local_symbol: {type(local_symbol)}" # Added type info
                )
                historical_data_result[local_symbol] = {"candles": []}
            except Exception as data_error:
                logger.error(
                    "Error fetching historical data for %s: %s",
                    local_symbol,
                    str(data_error),
                )
                historical_data_result[local_symbol] = {"candles": []}
        logger.info(f"local_config.SYMBOLS after loop: {local_config.SYMBOLS}") #added

        logger.info("Starting ForexBot with multiple background tasks...")
        tasks = [
            asyncio.create_task(local_bot.run(), name="bot_run"),
            asyncio.create_task(
                local_bot.data_manager.start_subscriptions(), name="data_subs"
            ),
            asyncio.create_task(
                local_bot.initialize_data_manager(local_data_manager), name="data_init"
            ),
            asyncio.create_task(start_web_interface(local_bot), name="web_interface"),
            asyncio.create_task(
                run_blocking_tasks(local_bot, local_config), name="blocking_tasks"
            ),
        ]

        await asyncio.gather(*tasks)
        pdb.set_trace()
    except Exception as main_error:
        logger.error("An error occurred in main: %s", str(main_error))
    finally:
        if local_api:
            await local_api.disconnect()
        logger.info("Main function completed")
        return historical_data_result, local_bot, local_api


async def event_loop_watchdog(config: Config) -> None:
    """Monitors event loop health indefinitely."""
    logger.info("Watchdog started in process %d", os.getpid())
    while True:
        start_time = time.monotonic()
        await asyncio.sleep(config.WATCHDOG_INTERVAL)
        delay = time.monotonic() - start_time - config.WATCHDOG_INTERVAL
        if delay > config.STARVATION_THRESHOLD:
            logger.warning("Event loop starvation detected! Delay: %.2fs", delay)
        current_tasks = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]
        logger.debug("Active tasks: %d", len(current_tasks))


async def run_blocking_tasks(local_bot: ForexBot, local_config: Config) -> None:
    """Run CPU-intensive tasks in executor indefinitely."""
    logger.info("Blocking tasks started in process %d", os.getpid())
    event_loop = asyncio.get_event_loop()
    while True:
        try:
            await event_loop.run_in_executor(
                None, partial(process_blocking_operations, local_bot, local_config)
            )
            await asyncio.sleep(1)
        except ValueError as blocking_error:
            logger.error("Blocking task error: %s", str(blocking_error))
            await asyncio.sleep(5)


def process_blocking_operations(local_bot: ForexBot, local_config: Config) -> None:
    """Process blocking operations."""
    symbols = (
        local_config.SYMBOLS.split(",")
        if isinstance(local_config.SYMBOLS, str)
        else local_config.SYMBOLS
    )
    if symbols:
        symbol_to_use = symbols[0].strip()
        logger.info(f"Creating tick callback for symbol: {symbol_to_use}")
        local_bot.create_tick_callback(symbol_to_use)


async def stop_bot(local_bot: ForexBot, local_api: DerivAPI) -> None:
    """Stop the bot and clean up resources."""
    logger.info("Stopping bot")
    await local_bot.stop()
    await local_api.clear()
    logger.info("Bot stopped")


if __name__ == "__main__":
    logger.info("Main process started with PID %d", os.getpid())
    try:
        async def run_with_cleanup() -> Tuple:
            try:
                main_result = await main()
                logger.debug("Main returned: %s", main_result)
                return main_result
            except Exception as cleanup_error:
                logger.error("Error in main function: %s", str(cleanup_error))
                return None, None, None

        # Use asyncio.run to execute the run_with_cleanup coroutine
        main_result = asyncio.run(run_with_cleanup(), debug=True)

        if main_result is None or len(main_result) != 3:
            logger.critical(
                "Unexpected result from main function. Expected 3 values, got: %s",
                main_result,
            )
            sys.exit(1)

        historical_data_main, forex_bot, deriv_api = main_result

        logger.info("Main completed")
        if historical_data_main and any(historical_data_main.values()):
            for symbol_key, data in historical_data_main.items():
                if data and "candles" in data and data["candles"]:
                    logger.info(
                        "First candle for %s: %s", symbol_key, data["candles"][0]
                    )
        else:
            logger.warning("No historical data returned from main()")

        logger.info("Bot running in background. Press Ctrl+C to stop.")

        # Define an async function to handle the main loop
        async def main_loop():
            try:
                while True:
                    await asyncio.sleep(1)  # Keep the program running
            except KeyboardInterrupt:
                logger.info("Received interrupt, stopping...")
                await stop_bot(forex_bot, deriv_api)

        # Run the main loop
        asyncio.run(main_loop())

    except Exception as bootstrap_error:
        logger.critical("Application bootstrap failed: %s", str(bootstrap_error))
        sys.exit(1)
