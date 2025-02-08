from configparser import ConfigParser

config.add_section('Settings')
config.set('Settings', 'SYMBOLS', 'frxAUDCAD,frxEURUSD,frxGBPUSD') #, 'frxAUDCHF', 'frxAUDJPY', 'frxAUDNZD', 'frxAUDUSD', 'frxEURAUD', 'frxEURCAD', 'frxEURCHF', 'frxEURGBP', 'frxEURJPY', 'frxEURNZD', 'frxEURUSD', 'frxGBPAUD', 'frxGBPCAD', 'frxGBPCHF', 'frxGBPJPY', 'frxGBPUSD', 'frxNZDUSD', 'frxUSDCAD', 'frxUSDCHF', 'frxUSDJPY', 'frxUSDMXN', 'frxUSDNOK', 'frxUSDPLN', 'frxUSDSEK', 'frxXAGUSD', 'frxXAUUSD', 'frxXPDUSD', 'frxXPTUSD']
config.set('Settings', 'TIMEFRAME', '5m')
config.set('Settings', 'RISK_PERCENTAGE', '0.01')
config.set('Settings', 'MAX_TRADES_PER_SYMBOL', '2')
config.set('Settings', 'STOP_LOSS_PIPS', '20')
config.set('Settings', 'TAKE_PROFIT_PIPS', '40')
config.set('Settings', 'TRAILING_STOP_PIPS', '15')
config.set('Settings', 'HISTORICAL_DAYS', '90')
config.set('Settings', 'BACKTEST_START_DATE', '2023-01-01')
config.set('Settings', 'BACKTEST_END_DATE', '2023-06-30')

import os
import time
import sys
from configparser import ConfigParser

# Check if config.ini already exists
config_file = 'config/config.ini'

if not os.path.exists(config_file):
    # Create the config.ini file if it doesn't exist
    config = ConfigParser()
    config.add_section('DEFAULT')
    config.set('DEFAULT', 'DERIV_API_TOKEN', '')
    config.set('DEFAULT', 'APP_ID', '')

    config.add_section('Settings')
    config.set('Settings', 'SYMBOLS', 'frxAUDCAD,frxEURUSD,frxGBPUSD')
    config.set('Settings', 'TIMEFRAME', '5m')
    config.set('Settings', 'RISK_PERCENTAGE', '0.01')
    config.set('Settings', 'MAX_TRADES_PER_SYMBOL', '2')
    config.set('Settings', 'STOP_LOSS_PIPS', '20')
    config.set('Settings', 'TAKE_PROFIT_PIPS', '40')
    config.set('Settings', 'TRAILING_STOP_PIPS', '15')
    config.set('Settings', 'HISTORICAL_DAYS', '90')
    config.set('Settings', 'BACKTEST_START_DATE', '2023-01-01')
    config.set('Settings', 'BACKTEST_END_DATE', '2023-06-30')

    with open(config_file, mode='w') as f:
        config.write(f)
    print(f'{config_file} was created successfully.')
else:
    print(f'{config_file} already exists. No need to overwrite.')

class Config:
    def __init__(self, config_file='config/config.ini'):
        # Initialize ConfigParser
        self.config = ConfigParser()
        self.config.read(config_file)

        # Ensure APP_ID is set in config.ini
        app_id = self.config['DEFAULT'].get('APP_ID', None)
        if not app_id:
            print("APP_ID is not set in config.ini")
            time.sleep(2)
            sys.exit('Exiting...')

        # Set the API token from config
        self.API_TOKEN = self.config['DEFAULT'].get('DERIV_API_TOKEN', '')

        # EndPoint URL should be based on the APP_ID in the config
        self.EndPoint = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"

        # Settings from the 'Settings' section of the config file
        self.SYMBOLS = self.config['Settings'].get('SYMBOLS', '').split(',')
        self.TIMEFRAME = self.config['Settings'].get('TIMEFRAME', '5m')
        self.RISK_PERCENTAGE = self.config['Settings'].getfloat('RISK_PERCENTAGE', fallback=0.01)
        self.MAX_TRADES_PER_SYMBOL = self.config['Settings'].getint('MAX_TRADES_PER_SYMBOL', fallback=2)
        self.STOP_LOSS_PIPS = self.config['Settings'].getint('STOP_LOSS_PIPS', fallback=20)
        self.TAKE_PROFIT_PIPS = self.config['Settings'].getint('TAKE_PROFIT_PIPS', fallback=40)
        self.TRAILING_STOP_PIPS = self.config['Settings'].getint('TRAILING_STOP_PIPS', fallback=15)
        self.HISTORICAL_DAYS = self.config['Settings'].getint('HISTORICAL_DAYS', fallback=90)
        self.BACKTEST_START_DATE = self.config['Settings'].get('BACKTEST_START_DATE', '2023-01-01')
        self.BACKTEST_END_DATE = self.config['Settings'].get('BACKTEST_END_DATE', '2023-06-30')

