from aiohttp import web
import json

# Create a custom AppKey for the bot
BOT_KEY = web.AppKey('bot')

async def start_web_interface(bot):
    app = web.Application()
    app[BOT_KEY] = bot  # Use the AppKey to store the bot instance

    async def get_bot_status():
        return {
            'running': bot.running,
            'open_trades': bot.portfolio_manager.get_open_positions(),
            'total_exposure': bot.portfolio_manager.get_total_exposure(),
        }

    async def status_handler(request):
        status = await get_bot_status()
        return web.json_response(status)

    async def start_handler(request):
        if not bot.running:
            await bot.run()
            return web.json_response({'status': 'Bot started'})
        return web.json_response({'status': 'Bot is already running'})

    async def stop_handler(request):
        if bot.running:
            await bot.stop()  # Ensure stop is awaited if it's an async method
            return web.json_response({'status': 'Bot stopped'})
        return web.json_response({'status': 'Bot is not running'})

    async def backtest_handler(request):
        results = await bot.run_backtest()
        return web.json_response(results)

    # Define routes
    app.router.add_get('/status', status_handler)
    app.router.add_post('/start', start_handler)
    app.router.add_post('/stop', stop_handler)
    app.router.add_post('/backtest', backtest_handler)

    # Start the web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    print("Web interface started at http://localhost:8080")
