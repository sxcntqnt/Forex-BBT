from aiohttp import web
import json
import logging

# Assuming ForexBot is imported from the correct module
from bot import ForexBot

# Create a custom AppKey for the bot
BOT_KEY = web.AppKey("bot")


async def start_web_interface(bot: ForexBot):
    app = web.Application()
    app[BOT_KEY] = bot  # Use the AppKey to store the bot instance

    async def get_bot_status():
        try:
            return {
                "running": bot.running,
                "open_trades": bot.portfolio_manager.get_open_positions(),
                "total_exposure": bot.portfolio_manager.get_total_exposure(),
            }
        except Exception as e:
            logging.error("Error getting bot status: %s", str(e))
            return {"error": "Failed to retrieve bot status"}

    async def status_handler(request):
        status = await get_bot_status()
        return web.json_response(status)

    async def start_handler(request):
        if not bot.running:
            await bot.run()
            return web.json_response({"status": "Bot started"})
        return web.json_response({"status": "Bot is already running"})

    async def stop_handler(request):
        if bot.running:
            await bot.stop()  # Ensure stop is awaited if it's an async method
            return web.json_response({"status": "Bot stopped"})
        return web.json_response({"status": "Bot is not running"})

    async def backtest_handler(request):
        try:
            results = await bot.run_backtest()
            return web.json_response(results)
        except Exception as e:
            logging.error("Error during backtest: %s", str(e))
            return web.json_response({"error": "Backtest failed"}, status=500)

    # Define routes
    app.router.add_get("/status", status_handler)
    app.router.add_post("/start", start_handler)
    app.router.add_post("/stop", stop_handler)
    app.router.add_post("/backtest", backtest_handler)

    # Start the web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()

    logging.info("Web interface started at http://localhost:8080")

    # Keep the server running
    return runner  # Return the runner to allow for cleanup later
