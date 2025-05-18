from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import talib
import asyncio
import joblib
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from config import Config
from typing import Optional, Tuple, List
import logging
import os
from sklearn.model_selection import GridSearchCV

class MLStrategy:
    def __init__(self, config: Config, logger: logging.Logger):
        if not isinstance(config, Config):
            raise TypeError("Requires valid Config instance")
        self.config = config
        self.logger = logger
        self._validate_config()
        self.model = RandomForestRegressor(
            n_estimators=config.ML_N_ESTIMATORS,
            max_depth=config.ML_MAX_DEPTH,
            random_state=config.SEED,
        )
        self.trained = False
        self.min_training_samples = config.ML_MIN_SAMPLES
        self.model_path = os.path.join(config.MODEL_DIR, "random_forest.joblib")
        self.stop_loss = self._pips_to_percentage(config.STOP_LOSS_PIPS)
        self.take_profit = self._pips_to_percentage(config.TAKE_PROFIT_PIPS)
        self.trailing_stop = self._pips_to_percentage(config.TRAILING_STOP_PIPS)
        self.risk_percentage = config.RISK_PERCENTAGE
        self.performance_metrics = {'trades': 0, 'wins': 0, 'total_return': 0.0, 'total_pips': 0.0}
        self.last_trade_price = {}
        self.highest_price = {}
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        self.load_model()
        self.num_lags = 2  # Fix number of lag features to ensure 9 total features

    def _validate_config(self) -> None:
        """Validate required config parameters."""
        required = [
            'SYMBOLS', 'MODEL_DIR', 'ML_N_ESTIMATORS', 'ML_MAX_DEPTH', 'ML_MIN_SAMPLES',
            'ML_WINDOW_SIZE', 'ML_THRESHOLD', 'SEED', 'STOP_LOSS_PIPS', 'TAKE_PROFIT_PIPS',
            'TRAILING_STOP_PIPS', 'RISK_PERCENTAGE'
        ]
        for param in required:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing config parameter: {param}")

    def _pips_to_percentage(self, pips: int) -> float:
        """Convert pips to percentage for R_50 (volatility index)."""
        return pips * 0.0001

    async def should_enter_trade(self, symbol: str, data_manager) -> bool:
        """Evaluate trading signal with risk management."""
        try:
            if not await self._validate_symbol_data(symbol, data_manager):
                return False
            if not self.trained and not await self._attempt_training(data_manager, symbol):
                return False
            df = data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                self.logger.warning(f"No snapshot data for {symbol}")
                return False
            features = await self._extract_features_with_validation(symbol, data_manager)
            if features is None:
                return False
            self.logger.debug(f"Features for {symbol}: {features}")
            prediction = self.model.predict([features])[0]
            current_price = df['close'].iloc[-1]
            if not await self._validate_prediction(prediction, current_price):
                return False
            if current_price * (1 - self.stop_loss) < self._get_last_trade_price(symbol):
                self.logger.debug(f"Stop-loss triggered for {symbol}")
                return False
            self.last_trade_price[symbol] = current_price
            self.highest_price[symbol] = current_price
            self.performance_metrics['trades'] += 1
            return True
        except Exception as e:
            self.logger.error(f"Prediction error for {symbol}: {str(e)}", exc_info=True)
            return False

    async def should_exit_trade(self, symbol: str, data_manager) -> bool:
        """Evaluate if a trade should be exited with take-profit and trailing stop."""
        try:
            if not await self._validate_symbol_data(symbol, data_manager):
                return False
            df = data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                self.logger.warning(f"No snapshot data for {symbol}")
                return False
            features = await self._extract_features_with_validation(symbol, data_manager)
            if features is None:
                return False
            self.logger.debug(f"Exit features for {symbol}: {features}")
            prediction = self.model.predict([features])[0]
            current_price = df['close'].iloc[-1]
            entry_price = self._get_last_trade_price(symbol)
            self.highest_price[symbol] = max(self.highest_price.get(symbol, current_price), current_price)
            if current_price >= entry_price * (1 + self.take_profit):
                self._record_trade(symbol, current_price, entry_price)
                return True
            if current_price <= self.highest_price[symbol] * (1 - self.trailing_stop):
                self._record_trade(symbol, current_price, entry_price)
                return True
            threshold = current_price * (1 - self.config.ML_THRESHOLD)
            if prediction < threshold:
                self._record_trade(symbol, current_price, entry_price)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Exit prediction error for {symbol}: {str(e)}", exc_info=True)
            return False

    def _record_trade(self, symbol: str, exit_price: float, entry_price: float) -> None:
        """Record trade outcome and update metrics."""
        profit = (exit_price - entry_price) / entry_price
        pips = (exit_price - entry_price) / 0.0001
        self.performance_metrics['total_return'] += profit
        self.performance_metrics['total_pips'] += pips
        self.performance_metrics['wins'] += 1 if profit > 0 else 0
        self._log_performance(symbol)

    async def _attempt_training(self, data_manager, symbol: str, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                await self.train(data_manager, symbol)
                if self.trained:
                    self.logger.info(f"Training successful for {symbol}")
                    return True
                await asyncio.sleep(2**attempt)
            except Exception as e:
                self.logger.warning(f"Training attempt {attempt+1} failed: {e}")
        self.logger.error(f"All training attempts failed for {symbol}")
        return False

    async def train(self, data_manager, symbol: str) -> None:
        try:
            df = data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                raise ValueError(f"No data for {symbol}")
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"Invalid index for {symbol}: {df.index}")
            prices = df['close'].values
            stationarity = await self._check_stationarity(prices)
            if not stationarity["is_stationary"]:
                prices = await self._make_stationary(prices)
            significant_lags = await self._calculate_pacf(prices)
            X, y = await self._prepare_training_data(prices, significant_lags)
            if len(X) < self.min_training_samples:
                raise ValueError(f"Insufficient training samples: {len(X)}/{self.min_training_samples}")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.trained = True
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Trained model with {X.shape[1]} features")
        except Exception as e:
            self.trained = False
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    async def _prepare_training_data(self, prices: list, significant_lags: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(prices) - self.config.ML_WINDOW_SIZE - 1):
            window = prices[i : i + self.config.ML_WINDOW_SIZE]
            features = await self._extract_features(window, significant_lags)
            if len(features) != 9:
                self.logger.warning(f"Skipping sample with {len(features)} features")
                continue
            target = await self._calculate_target(prices, i)
            X.append(features)
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        if len(X) == 0:
            raise ValueError("No valid training samples generated")
        return X, y

    async def _extract_features_with_validation(self, symbol: str, data_manager) -> Optional[List[float]]:
        try:
            df = data_manager.get_snapshot(symbol)
            if df is None or len(df) < self.config.ML_WINDOW_SIZE:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
                return None
            if 'close' not in df.columns:
                self.logger.error(f"Missing 'close' column in snapshot for {symbol}")
                return None
            prices = df['close'].values[-self.config.ML_WINDOW_SIZE:]
            significant_lags = await self._calculate_pacf(prices)
            features = await self._extract_features(prices, significant_lags)
            if len(features) != 9:
                self.logger.error(f"Feature count mismatch for {symbol}: got {len(features)}, expected 9")
                return None
            if not all(np.isfinite(features)):
                self.logger.error(f"Invalid features for {symbol}: contains NaN or inf")
                return None
            return features
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {symbol}: {str(e)}", exc_info=True)
            return None

    async def _extract_features(self, prices: list, significant_lags: List[int]) -> List[float]:
        returns = np.diff(prices) / prices[:-1]
        price_series = np.array(prices)
        features = [
            np.mean(returns) if len(returns) > 0 else 0.0,
            np.std(returns) if len(returns) > 0 else 0.0,
            (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0,
            talib.RSI(price_series, timeperiod=9)[-1] if len(price_series) >= 9 else 0.0,
        ]
        upper, middle, lower = talib.BBANDS(price_series, timeperiod=5)
        features.extend([
            upper[-1] if not np.isnan(upper[-1]) else 0.0,
            middle[-1] if not np.isnan(middle[-1]) else 0.0,
            lower[-1] if not np.isnan(lower[-1]) else 0.0,
        ])
        # Fix lag features to ensure exactly 2 lags
        lag_features = [prices[-lag] if lag <= len(prices) and lag > 0 else 0.0 for lag in sorted(significant_lags)[:self.num_lags]]
        while len(lag_features) < self.num_lags:
            lag_features.append(0.0)
        features.extend(lag_features)
        self.logger.debug(f"Feature breakdown for {len(prices)} prices: base={len(features)-len(lag_features)}, lags={len(lag_features)}, total={len(features)}")
        return features

    async def _check_stationarity(self, series: list) -> dict:
        try:
            series = np.array(series)
            if len(series) < 2:
                return {"is_stationary": False, "adf_pvalue": 1.0, "kpss_pvalue": 0.0}
            adf_result = adfuller(series, maxlag=10, regression='c')
            kpss_result = kpss(series, regression='c', nlags='auto')
            return {
                "is_stationary": adf_result[1] < 0.05 and kpss_result[1] > 0.05,
                "adf_pvalue": adf_result[1],
                "kpss_pvalue": kpss_result[1],
            }
        except Exception as e:
            self.logger.warning(f"Stationarity check failed: {str(e)}")
            return {"is_stationary": False, "adf_pvalue": 1.0, "kpss_pvalue": 0.0}

    async def _make_stationary(self, series: list, max_diff: int = 2) -> list:
        series = np.array(series)
        for d in range(1, max_diff + 1):
            diff_series = np.diff(series, n=d)
            if (await self._check_stationarity(diff_series))["is_stationary"]:
                return diff_series.tolist()
        return series.tolist()

    async def _calculate_pacf(self, series: list, nlags: Optional[int] = None) -> List[int]:
        try:
            series = np.array(series)
            if len(series) < 2:
                return []
            nlags = nlags or min(len(series) // 2 - 1, 20)
            pacf_values = pacf(series, nlags=nlags, method='ywm')
            significant = [
                i + 1 for i, val in enumerate(pacf_values[1:])
                if abs(val) > 2 / np.sqrt(len(series))
            ]
            self.logger.debug(f"PACF significant lags: {significant}")
            return significant
        except Exception as e:
            self.logger.warning(f"PACF calculation failed: {str(e)}")
            return []

    async def _calculate_target(self, prices: list, index: int) -> float:
        try:
            if index + self.config.ML_WINDOW_SIZE + 1 >= len(prices):
                return 0.0
            current_price = prices[index + self.config.ML_WINDOW_SIZE - 1]
            future_price = prices[index + self.config.ML_WINDOW_SIZE]
            return (future_price - current_price) / current_price if current_price != 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Target calculation failed: {str(e)}")
            return 0.0

    async def _validate_symbol_data(self, symbol: str, data_manager) -> bool:
        if symbol not in self.config.SYMBOLS:
            self.logger.warning(f"Symbol {symbol} not in configured symbols")
            return False
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.config.ML_WINDOW_SIZE * 2:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
            return False
        if 'close' not in df.columns:
            self.logger.error(f"Missing 'close' column in snapshot for {symbol}")
            return False
        return True

    async def _validate_prediction(self, prediction: float, current_price: float) -> bool:
        if np.isnan(prediction) or not np.isfinite(prediction):
            self.logger.warning("Invalid prediction value")
            return False
        threshold = current_price * (1 + self.config.ML_THRESHOLD)
        return prediction > threshold

    def load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.trained = True
                self.logger.info("Loaded pre-trained model")
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
                self.trained = False

    def _get_last_trade_price(self, symbol: str) -> float:
        return self.last_trade_price.get(symbol, 1.0)

    def _log_performance(self, symbol: str) -> None:
        win_rate = self.performance_metrics['wins'] / self.performance_metrics['trades'] if self.performance_metrics['trades'] > 0 else 0
        avg_return = self.performance_metrics['total_return'] / self.performance_metrics['trades'] if self.performance_metrics['trades'] > 0 else 0
        avg_pips = self.performance_metrics['total_pips'] / self.performance_metrics['trades'] if self.performance_metrics['trades'] > 0 else 0
        self.logger.info(f"MLStrategy {symbol} - Win Rate: {win_rate:.2f}, Avg Return: {avg_return:.2f}, Avg Pips: {avg_pips:.2f}")
        with open('performance.csv', 'a') as f:
            f.write(f"MLStrategy,{symbol},{win_rate:.2f},{avg_return:.2f},{avg_pips:.2f}\n")
        if win_rate < 0.3:
            self.logger.warning(f"Low win rate for MLStrategy on {symbol}: {win_rate:.2f}")

    async def optimize_parameters(self, data_manager, symbol: str) -> dict:
        try:
            df = data_manager.get_snapshot(symbol)
            if df is None or df.empty:
                return {}
            if 'close' not in df.columns:
                self.logger.error(f"Missing 'close' column in snapshot for {symbol}")
                return {}
            prices = df['close'].values
            significant_lags = await self._calculate_pacf(prices)
            X, y = await self._prepare_training_data(prices, significant_lags)
            if len(X) < self.min_training_samples:
                return {}
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            return grid_search.best_params_
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {str(e)}")
            return {}
