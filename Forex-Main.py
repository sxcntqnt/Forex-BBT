import sys,time
import asyncio
import websockets
from deriv_api import DerivAPI
from bot import ForexBot
from config import Config
from web_interface import start_web_interface
from data_manager import DataManager  # Assuming DataManager is in a file named data_manager.py

async def main():
    # Initialize the configuration
    config = Config()

    # Check if API_TOKEN is set
    if not config.API_TOKEN:
        print("API_TOKEN is not set in the configuration file.")
        await asyncio.sleep(2)  # Replace blocking sleep
        sys.exit("Exiting....")

    # Establish WebSocket connection
    connection = await websockets.connect(config.EndPoint)

    # Initialize the ForexBot with the connection
    bot = ForexBot(config, connection)

    # Initialize DataManager before starting bot tasks
    data_manager = DataManager(config)

    # Start subscribing to symbols in DataManager
    subscription_task = asyncio.create_task(data_manager.start_subscriptions(config.EndPoint))

    # Start the web interface and bot tasks
    web_task = asyncio.create_task(start_web_interface(bot))
    bot_task = asyncio.create_task(bot.run())

    try:
        # Run all tasks concurrently
        await asyncio.gather(subscription_task, web_task, bot_task)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Ensure subscriptions are stopped when shutting down
        await data_manager.stop_subscriptions()
        await connection.close()  # Close the WebSocket connection

if __name__ == "__main__":
    asyncio.run(main())
