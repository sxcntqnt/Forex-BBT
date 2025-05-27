from config import Config

class RiskManager:
    def __init__(self, config: Config):
        self.config = config
        self.open_trades = {symbol: 0 for symbol in config.symbols}

    def can_enter_trade(self, symbol):
        return self.open_trades[symbol] < self.config.MAX_TRADES_PER_SYMBOL

    def calculate_position_size(self, symbol):
        account_balance = 10000  # This should be fetched from the API
        risk_amount = account_balance * self.config.RISK_PERCENTAGE
        return risk_amount / len(self.config.symbols)

    def update_open_trades(self, symbol, count):
        self.open_trades[symbol] = count
