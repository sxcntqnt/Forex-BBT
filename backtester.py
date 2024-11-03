import pandas as pd
from datetime import datetime
import os
import numpy as np
from deriv_api import DerivAPI

class Backtester:
    def __init__(self, config, api, data_manager,  strategy_manager):
        self.config = config
        self.api = api
        self.strategy_manager = strategy_manager
        self.data_manager = data_manager  # Make sure to pass the DataManager instance


    async def run(self):
        results = {}
        for symbol in self.config.SYMBOLS:
            historical_data = await self.fetch_historical_data(symbol)
            results[symbol] = self.backtest_symbol(symbol, historical_data)
        return results

    async def fetch_historical_data(self, symbol):
        # Convert date strings to timestamps
        start_timestamp = int(datetime.strptime(self.config.BACKTEST_START_DATE, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(self.config.BACKTEST_END_DATE, '%Y-%m-%d').timestamp())

        candles = await self.api.ticks_history({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 5000,
            "end": end_timestamp,
            "start": start_timestamp,
            "style": "candles"
        })
        return pd.DataFrame(candles['candles'])

    def backtest_symbol(self, symbol, data):
        balance = 10000
        trades = []

        for i in range(len(data)):
            if self.strategy_manager.should_enter_trade(symbol, self.data_manager):
                entry_price = data.iloc[i]['close']
                stop_loss = entry_price - (self.config.STOP_LOSS_PIPS / 10000)
                take_profit = entry_price + (self.config.TAKE_PROFIT_PIPS / 10000)
                
                for j in range(i+1, len(data)):
                    current_price = data.iloc[j]['close']
                    if current_price <= stop_loss:
                        profit = (stop_loss - entry_price) * 10000
                        trades.append(profit)
                        balance += profit
                        break
                    elif current_price >= take_profit:
                        profit = (take_profit - entry_price) * 10000
                        trades.append(profit)
                        balance += profit
                        break

