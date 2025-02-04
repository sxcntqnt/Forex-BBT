import sys,time
import asyncio
import websockets
from deriv_api import DerivAPI
from bot import ForexBot
from config import Config
from web_interface import start_web_interface


async def main():
    config = Config()

    if not config.API_TOKEN:
        print("DERIV_TOKEN environment variable is not set")
        await asyncio.sleep(2)  # Replace blocking sleep
        sys.exit("Exiting....")

    connection = await websockets.connect(config.EndPoint)
    bot = ForexBot(config, connection)

    
    # Initialize DataManager before starting bot tasks
    data_manager = await ForexBot.initialize_data_manager(bot, config, connection)

    # Start the web interface and bot tasks
    web_task = asyncio.create_task(start_web_interface(bot))
    bot_task = asyncio.create_task(bot.run())

    try:
        await asyncio.gather(web_task, bot_task)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")

if __name__ == "__main__":
    asyncio.run(main())
