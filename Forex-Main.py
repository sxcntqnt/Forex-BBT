import asyncio,nest_asyncio
import sys
import tracemalloc
import websockets
from deriv_api import DerivAPI
from bot import ForexBot
from config import Config
from web_interface import start_web_interface
from data_manager import DataManager  # Assuming DataManager is in a file named data_manager.py
from strategy import StrategyManager
import threading  # For running blocking tasks in a separate thread
import logging
import os


import asyncio
import nest_asyncio
import sys
import tracemalloc
import websockets
from deriv_api import DerivAPI
from bot import ForexBot
from config import Config
from web_interface import start_web_interface
from data_manager import DataManager
from strategy import StrategyManager
import logging
import os

# Apply nest_asyncio for environments with existing event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Main')

async def main():
    """Main function with proper resource management"""
    tracemalloc.start()
    config = None
    connection = None
    bot = None

    try:
        # 1. Initialize configuration
        config = Config()
        logger.info(f"Initialized config with symbols: {config.SYMBOLS}")

        # 2. Establish connection
        connection = await websockets.connect(config.EndPoint)
        logger.debug(f"Connected to {config.EndPoint}")

        # 3. Initialize core components
        data_manager = DataManager(connection=connection, config=config)
        strategy_manager = StrategyManager(data_manager=data_manager)

        # Create ForexBot with all dependencies
        bot = ForexBot(
            config=config,
            connection=connection,
            data_manager=data_manager,
            strategy_manager=strategy_manager
        )

        # 4. Authorize connection
        auth_response = await bot.api.authorize({'authorize': config.API_TOKEN})
        if not auth_response.get('authorize', {}).get('loginid'):
            raise ConnectionError("Authorization failed")

        # 5. Create and run tasks
        tasks = [
            asyncio.create_task(bot.data_manager.start_subscriptions(), name="data_subs"),
            asyncio.create_task(bot.run(), name="main_loop"),
            asyncio.create_task(start_web_interface(bot), name="web_interface"),
            asyncio.create_task(run_blocking_tasks(bot), name="blocking_tasks")
        ]

        await asyncio.gather(*tasks)

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
    finally:
        # 6. Cleanup resources
        logger.info("Starting shutdown sequence...")
        
        if bot:
            await bot.shutdown()
            
        if connection and not connection.closed:
            await connection.close()
            
        tracemalloc.stop()
        logger.info("Cleanup completed")

async def run_blocking_tasks(bot: ForexBot):
    """Handle synchronous operations with proper error handling"""
    logger.info("Starting blocking tasks handler")
    
    while True:
        try:
            # 1. Get latest data
            latest_data = {
                symbol: bot.data_manager.get_ohlc_data(symbol)
                for symbol in bot.config.SYMBOLS
            }

            # 2. Generate signals
            signals = await asyncio.get_event_loop().run_in_executor(
                None,
                bot.strategy_manager.check_signals,
                latest_data
            )

            # 3. Execute trades if signals exist
            if signals:
                logger.info(f"Executing signals: {signals}")
                await bot.execute_signals(signals)

            # 4. Throttle checks
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Blocking task error: {str(e)}", exc_info=True)
            await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Application bootstrap failed: {str(e)}")
        sys.exit(1)
