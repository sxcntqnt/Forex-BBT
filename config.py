import os, time, sys
from configparser import ConfigParser

class Config:
    def __init__(self, config_file='config/config.ini'):
        # Load configuration from the ini file
        self.config = ConfigParser()
        self.config.read(config_file)

        # Check for APP_ID after loading environment variables
        if not self.config['DEFAULT'].get("APP_ID"):
            print("APP_ID environment variable is not set")
            time.sleep(2)
            sys.exit('Exiting...')


        # Load other configuration variables
        self.API_TOKEN = self.config['DEFAULT'].get('DERIV_API_TOKEN')
        self.APP_ID = self.config['DEFAULT'].get('APP_ID')
        self.EndPoint = self.config['DEFAULT'].get('EndPoint')
        self.SYMBOLS = self.config['Settings'].get('SYMBOLS')
        self.TIMEFRAME = self.config['Settings'].get('TIMEFRAME')
        self.RISK_PERCENTAGE = self.config['Settings'].getfloat('RISK_PERCENTAGE', fallback=0.01)
        self.MAX_TRADES_PER_SYMBOL = self.config['Settings'].getint('MAX_TRADES_PER_SYMBOL', fallback=2)
        self.STOP_LOSS_PIPS = self.config['Settings'].getint('STOP_LOSS_PIPS', fallback=20)
        self.TAKE_PROFIT_PIPS = self.config['Settings'].getint('TAKE_PROFIT_PIPS', fallback=40)
        self.TRAILING_STOP_PIPS = self.config['Settings'].getint('TRAILING_STOP_PIPS', fallback=15)
        self.HISTORICAL_DAYS = self.config['Settings'].getint('HISTORICAL_DAYS', fallback=30)
        self.BACKTEST_START_DATE = self.config['Settings'].get('BACKTEST_START_DATE')
        self.BACKTEST_END_DATE = self.config['Settings'].get('BACKTEST_END_DATE')
