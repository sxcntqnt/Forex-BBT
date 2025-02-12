import asyncio
import logging
import sys
import time
import tracemalloc
from functools import partial

import nest_asyncio
from deriv_api import DerivAPI
from bot import ForexBot
from config import Config
from data_manager import DataManager
from strategy import StrategyManager
from web_interface import start_web_interface

# Apply nest_asyncio for environments with existing event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot_debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Main")


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

        # 2. Initialize the DerivAPI object
        try:
            api = DerivAPI(endpoint=config.EndPoint, app_id=config.APP_ID)
            logger.debug(
                "Initialized DerivAPI with endpoint: %s and app_id: %s",
                config.EndPoint,
                config.APP_ID,
            )
        except Exception as e:
            logger.error("Failed to initialize DerivAPI: %s", str(e))
            return

        # 3. Initialize core components
        data_manager = DataManager(config=config, api=api)
        strategy_manager = StrategyManager(data_manager=data_manager, api=api)

        # 4. Create and configure bot
        bot = ForexBot(
            config=config,
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            api=api,
        )

        # 5. Authorize with timeout
        try:
            auth_response = await asyncio.wait_for(
                api.authorize({"authorize": config.API_TOKEN}), timeout=5
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
            asyncio.create_task(
                bot.data_manager.start_subscriptions(), name="data_subs"
            ),
            asyncio.create_task(bot.run(), name="main_loop"),
            asyncio.create_task(start_web_interface(bot), name="web_interface"),
            asyncio.create_task(run_blocking_tasks(bot), name="blocking_tasks"),
            asyncio.create_task(event_loop_watchdog(), name="watchdog"),
        ]

        # 7. Run with timeout protection
        await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=config.MAX_RUNTIME,  # Add to config (e.g., 3600 seconds)
        )

    except asyncio.TimeoutError:
        logger.warning("Main execution timeout reached")
    except Exception as e:  # pylint: disable=broad-except
        logger.critical("Critical error: %s", str(e), exc_info=True)
    finally:
        # 8. Cleanup with cancellation handling
        logger.info("Initiating shutdown sequence...")

        # Cancel all running tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close resources
        if bot:
            await bot.stop()
        if api:
            await api.clear()  # Assuming DerivAPI has a close method

        tracemalloc.stop()
        logger.info("Cleanup completed")


async def event_loop_watchdog():
    """Monitors event loop health and prevents starvation."""
    while True:
        start_time = time.monotonic()
        await asyncio.sleep(5)  # Check every 5 seconds

        # Measure event loop delay
        delay = time.monotonic() - start_time - 5
        if delay > 0.5:  # Threshold in seconds
            logger.warning("Event loop starvation detected! Delay: %.2fs", delay)

        # Log task states
        current_tasks = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]
        logger.debug("Active tasks: %d", len(current_tasks))


async def run_blocking_tasks(bot: ForexBot):
    """Run CPU-intensive tasks in executor."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Offload blocking operations to thread pool
            await loop.run_in_executor(
                None, partial(process_blocking_operations, bot)  # Default executor
            )
            await asyncio.sleep(1)  # Yield control
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Blocking task error: %s", str(e))
            await asyncio.sleep(5)


def process_blocking_operations(bot: ForexBot):
    """Process blocking operations."""
    # Implement blocking operations here
    pass


if __name__ == "__main__":
    try:
        asyncio.run(main(), debug=True)
    except Exception as e:
        logger.critical("Application bootstrap failed: %s", str(e))
        sys.exit(1)
