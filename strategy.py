import talib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class StrategyManager:
    def __init__(self):
        self.strategies = [
            RSIStrategy(),
            MACDStrategy(),
            MLStrategy()
        ]

    def should_enter_trade(self, symbol, data_manager):
        return any(strategy.should_enter_trade(symbol, data_manager) for strategy in self.strategies)

class RSIStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30

    def should_enter_trade(self, symbol, data_manager):
        close_prices = np.array(data_manager.get_close_prices(symbol)) 
        
        if len(close_prices) < self.rsi_period:
            return False

        rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
        return rsi[-1] < self.rsi_oversold

class MACDStrategy:
    def __init__(self):
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9

    def should_enter_trade(self, symbol, data_manager):
        close_prices = np.array(data_manager.get_close_prices(symbol))
        
        if len(close_prices) < self.slow_period:
            return False

        macd, signal, _ = talib.MACD(close_prices, fastperiod=self.fast_period, slowperiod=self.slow_period, signalperiod=self.signal_period)
        return macd[-1] > signal[-1] and macd[-2] <= signal[-2]

class MLStrategy:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False

    def should_enter_trade(self, symbol, data_manager):
        if not self.trained:
            self.train(data_manager)

        features = self.extract_features(symbol, data_manager)
        prediction = self.model.predict([features])[0]
        current_price = data_manager.get_close_prices(symbol)[-1]
        
        return prediction > current_price * 1.001  # Predict 0.1% increase

    def train(self, data_manager):
        X, y = self.prepare_training_data(data_manager)
        self.model.fit(X, y)
        self.trained = True

    def prepare_training_data(self, DataManager):
        X, y = [], []
        for symbol in DataManager.symbols:
            prices = DataManager.get_close_prices(symbol)
        
            # Check if there are enough prices
            if len(prices) < 11:
                print(f"Not enough prices for {symbol}: {len(prices)}")
                continue

            for i in range(len(prices) - 11):
                features = self.extract_features(symbol, DataManager, i)
                target = (prices[i+10] - prices[i+9]) / prices[i+9]  # Next candle's return
                X.append(features)
                y.append(target)

        # Convert to NumPy arrays
        X, y = np.array(X), np.array(y)

        # Check if X and y are empty
        if X.size == 0 or y.size == 0:
            raise ValueError("Not enough data to train the model.")

        return X, y

    def extract_features(self, symbol, data_manager, offset=0):
        prices = data_manager.get_close_prices(symbol)[offset:offset + 10]
        returns = np.diff(prices) / prices[:-1]
        return [
            np.mean(returns),
            np.std(returns),
            (prices[-1] - prices[0]) / prices[0],  # 10-period return
            talib.RSI(np.array(prices), timeperiod=9)[-1],
            *talib.BBANDS(np.array(prices), timeperiod=5, nbdevup=2, nbdevdn=2)[0]
        ]


# Strategy Improvement Plan

"""
1. Parameter Tuning
   - Objective: Optimize strategy parameters for better performance using historical data.
   - Approach:
     - Implement a systematic way to test different parameter values for each strategy.
     - Use techniques such as Grid Search or Bayesian Optimization to explore parameter space.
     - Define a performance metric (e.g., Sharpe ratio, win rate) to evaluate each set of parameters.
     - Save the best-performing parameters for use in the strategy.
   - Example Implementation:
     - For each strategy (e.g., RSI, MACD), create an `optimize_parameters` method that iterates over a range of parameter values and evaluates performance on historical data.
     - Store results in a structured format (e.g., dictionary or DataFrame) for easy analysis.

2. Ensemble Strategies
   - Objective: Improve decision-making by combining signals from multiple strategies.
   - Approach:
     - Modify the `StrategyManager` to aggregate signals from all strategies.
     - Implement logic to determine how to weigh signals (e.g., simple majority, weighted by past performance).
     - Ensure that each strategy can operate independently and provide a clear signal (buy/sell/hold).
   - Example Implementation:
     - Create a `should_enter_trade` method in `StrategyManager` that considers the signals from all strategies.
     - Return a composite signal based on the aggregation method chosen.

3. Risk Management
   - Objective: Minimize potential losses by implementing risk management rules.
   - Approach:
     - Each strategy should define risk parameters such as stop-loss and take-profit levels.
     - Implement logic to check these parameters before entering trades.
     - Use trailing stop-loss or position sizing methods to manage risk effectively.
   - Example Implementation:
     - Extend each strategy to incorporate risk management checks in the `should_enter_trade` method.
     - Keep track of the maximum drawdown for each strategy and halt trading if a threshold is breached.

4. Performance Tracking
   - Objective: Monitor and evaluate strategy performance over time.
   - Approach:
     - Implement a logging system to track key performance metrics (e.g., win rate, average return).
     - Use data visualization tools to present performance metrics for easier analysis.
     - Set up alerts for performance issues (e.g., if drawdown exceeds a certain percentage).
   - Example Implementation:
     - Use Python's logging module to log performance metrics after each trade or at regular intervals.
     - Store results in a database or a CSV file for further analysis.

5. Modularity
   - Objective: Ensure strategies can be easily modified or extended.
   - Approach:
     - Define a base class for strategies to enforce a common interface.
     - Each strategy should be self-contained, encapsulating its logic and parameters.
     - Use dependency injection to pass necessary components (like DataManager) to each strategy.
   - Example Implementation:
     - Create a base class (e.g., BaseStrategy) that defines the interface for all strategies.
     - Implement each specific strategy as a subclass that adheres to this interface.
"""
#code example 1
'''
class RSIStrategy:
    def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def optimize_parameters(self, data_manager, symbol):
        """
        Use Grid Search to find optimal RSI parameters based on historical performance.
        """
        best_score = -float('inf')
        best_params = {}
        for rsi_period in range(5, 30):
            for rsi_overbought in range(60, 80, 5):
                for rsi_oversold in range(20, 40, 5):
                    self.rsi_period = rsi_period
                    self.rsi_overbought = rsi_overbought
                    self.rsi_oversold = rsi_oversold
                    score = self.evaluate_performance(data_manager, symbol)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'rsi_period': rsi_period,
                            'rsi_overbought': rsi_overbought,
                            'rsi_oversold': rsi_oversold
                        }
        return best_params

    def evaluate_performance(self, data_manager, symbol):
        """
        Evaluate the performance of the strategy with the current parameters.
        This method should return a score based on predefined criteria (e.g., profit factor).
        """
        # Placeholder implementation for performance evaluation
        return 0
'''
#code example 2
'''
class StrategyManager:
    def __init__(self):
        self.strategies = [
            RSIStrategy(),
            MACDStrategy(),
            MLStrategy()
        ]

    def should_enter_trade(self, symbol, data_manager):
        """
        Aggregate trade signals from all strategies.
        A simple majority vote can be used to decide whether to enter a trade.
        """
        signals = [strategy.should_enter_trade(symbol, data_manager) for strategy in self.strategies]
        return sum(signals) > len(signals) / 2  # More than half signal to enter trade

'''
#code example 3
'''
class RSIStrategy:
    def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30, stop_loss=0.01):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss = stop_loss

    def should_enter_trade(self, symbol, data_manager):
        """
        Check if the trade should be executed based on RSI and risk management parameters.
        """
        close_prices = np.array(data_manager.get_close_prices(symbol))
        if len(close_prices) < self.rsi_period:
            return False

        rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
        current_price = close_prices[-1]

        # Risk management logic
        if rsi[-1] < self.rsi_oversold:
            # Implement stop-loss logic here (for example purposes, we check for a loss of 1%)
            if current_price * (1 - self.stop_loss) < self.get_last_trade_price():
                return False
            return True
        return False

    def get_last_trade_price(self):
        """
        Placeholder method to get the last trade's entry price.
        This would normally come from your trading logic.
        """
        return 1.0  # Replace with actual logic
'''
#code example 4
'''
import logging

class StrategyManager:
    def __init__(self):
        self.strategies = [
            RSIStrategy(),
            MACDStrategy(),
            MLStrategy()
        ]
        logging.basicConfig(filename='strategy_performance.log', level=logging.INFO)

    def log_performance(self, strategy_name, win_rate, average_return):
        """
        Log performance metrics for each strategy.
        This helps in analyzing and refining strategies over time.
        """
        logging.info(f'{strategy_name} - Win Rate: {win_rate:.2f}, Average Return: {average_return:.2f}')
'''
#code example 5
'''
class BaseStrategy:
    def should_enter_trade(self, symbol, data_manager):
        """
        Base method for checking whether to enter a trade.
        This should be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class RSIStrategy(BaseStrategy):
    # Implementation remains the same, but now inherits from BaseStrategy
'''
