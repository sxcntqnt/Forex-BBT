import asyncio
from bot import ForexBot
from config import Config
from web_interface import start_web_interface

async def main():
    config = Config()
    bot = ForexBot(config)
    
    web_task = asyncio.create_task(start_web_interface(bot))
    bot_task = asyncio.create_task(bot.run())
    
    await asyncio.gather(web_task, bot_task)

if __name__ == "__main__":
    asyncio.run(main())

