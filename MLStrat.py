from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import talib
import asyncio
import joblib
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_pacf
from config import Config
from typing import Optional, Tuple
import logging
import os

logger = logging.getLogger('MLStrategy')

class MLStrategy:
    def __init__(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("Requires valid Config instance")
            
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.ML_N_ESTIMATORS,
            max_depth=config.ML_MAX_DEPTH,
            random_state=config.SEED
        )
        self.trained = False
        self.min_training_samples = config.ML_MIN_SAMPLES
        self.model_path = os.path.join(config.MODEL_DIR, 'random_forest.joblib')
        
        # Ensure model directory exists
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    async def should_enter_trade(self, symbol: str, data_manager) -> bool:
        """Evaluate trading signal with robustness checks"""
        try:
            if not self._validate_symbol_data(symbol, data_manager):
                return False

            if not self.trained and not await self._attempt_training(data_manager, symbol):
                return False

            features = self._extract_features_with_validation(symbol, data_manager)
            if features is None:
                return False

            prediction = self.model.predict([features])[0]
            current_price = data_manager.get_close_prices(symbol)[-1]
            
            return self._validate_prediction(prediction, current_price)
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {str(e)}", exc_info=True)
            return False

    async def _attempt_training(self, data_manager, symbol: str, retries: int = 3) -> bool:
        """Training with retry logic and validation"""
        for attempt in range(retries):
            try:
                await self.train(data_manager, symbol)
                if self.trained:
                    return True
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"Training attempt {attempt+1} failed: {str(e)}")
        return False

    async def train(self, data_manager, symbol: str) -> None:
        """Enhanced training method with statistical validation"""
        try:
            prices = data_manager.get_close_prices(symbol)
            
            # Statistical validation
            stationarity = self._check_stationarity(prices)
            significant_lags = self._calculate_pacf(prices)
            
            if not stationarity['is_stationary']:
                prices = self._make_stationary(prices)
                
            X, y = self._prepare_training_data(prices, significant_lags)
            
            if len(X) < self.min_training_samples:
                raise ValueError(f"Insufficient training samples: {len(X)}/{self.min_training_samples}")
                
            self.model.fit(X, y)
            self.trained = True
            joblib.dump(self.model, self.model_path)
            
        except Exception as e:
            self.trained = False
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def _prepare_training_data(self, prices: list, significant_lags: list) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data with enhanced features"""
        X, y = [], []
        for i in range(len(prices) - self.config.ML_WINDOW_SIZE - 1):
            window = prices[i:i+self.config.ML_WINDOW_SIZE]
            
            # Stationarity check
            if self._check_stationarity(window)['is_stationary']:
                features = self._extract_features(window, significant_lags)
                target = self._calculate_target(window)
                X.append(features)
                y.append(target)
                
        return np.array(X), np.array(y)

    def _extract_features_with_validation(self, symbol: str, data_manager):
        """Feature extraction with validation checks"""
        try:
            prices = data_manager.get_close_prices(symbol)
            if len(prices) < self.config.ML_WINDOW_SIZE:
                logger.warning(f"Insufficient data for {symbol}: {len(prices)}")
                return None
                
            significant_lags = self._calculate_pacf(prices)
            return self._extract_features(prices[-self.config.ML_WINDOW_SIZE:], significant_lags)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}", exc_info=True)
            return None

    def _extract_features(self, prices: list, significant_lags: list) -> list:
        """Enhanced feature engineering"""
        returns = np.diff(prices) / prices[:-1]
        price_series = np.array(prices)
        
        # Basic features
        features = [
            np.mean(returns),
            np.std(returns),
            (prices[-1] - prices[0]) / prices[0],  # Window return
            talib.RSI(price_series, timeperiod=9)[-1],
        ]
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(price_series, timeperiod=5)
        features.extend([upper[-1], middle[-1], lower[-1]])
        
        # PACF features
        features.extend([prices[-lag] for lag in significant_lags if lag <= len(prices)])
        
        return features

    def _check_stationarity(self, series: list) -> dict:
        """Statistical stationarity checks using ADF and KPSS tests"""
        adf_result = adfuller(series)
        kpss_result = kpss(series)
        
        return {
            'is_stationary': adf_result[1] < 0.05 and kpss_result[1] > 0.05,
            'adf_pvalue': adf_result[1],
            'kpss_pvalue': kpss_result[1]
        }

    def _make_stationary(self, series: list, max_diff: int = 2) -> list:
        """Make series stationary through differencing"""
        for d in range(1, max_diff+1):
            diff_series = np.diff(series, n=d)
            if self._check_stationarity(diff_series)['is_stationary']:
                return diff_series.tolist()
        return series

    def _calculate_pacf(self, series: list, nlags: Optional[int] = None) -> list:
        """Calculate significant partial autocorrelation lags"""
        nlags = nlags or min(len(series)//2 - 1, 20)
        pacf_values = plot_pacf(series, lags=nlags, alpha=0.05, method="ywm")
        significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > 2/np.sqrt(len(series))]
        return significant_lags

    def _calculate_target(self, window: list) -> float:
        """Calculate target variable with smoothing"""
        future_window = window[self.config.ML_WINDOW_SIZE:]
        if len(future_window) == 0:
            return 0
        return (future_window[-1] - window[-1]) / window[-1]

    def _validate_symbol_data(self, symbol: str, data_manager) -> bool:
        """Validate data requirements for symbol"""
        if symbol not in self.config.SYMBOLS:
            logger.warning(f"Symbol {symbol} not in configured symbols")
            return False
            
        prices = data_manager.get_close_prices(symbol)
        if len(prices) < self.config.ML_WINDOW_SIZE * 2:
            logger.warning(f"Insufficient data for {symbol}: {len(prices)}")
            return False
            
        return True

    def _validate_prediction(self, prediction: float, current_price: float) -> bool:
        """Validate prediction against business rules"""
        if np.isnan(prediction) or not np.isfinite(prediction):
            logger.warning("Invalid prediction value")
            return False
            
        threshold = current_price * (1 + self.config.ML_THRESHOLD)
        return prediction > threshold

    def load_model(self) -> None:
        """Load pre-trained model from disk"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.trained = True
            logger.info("Loaded pre-trained model")

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

