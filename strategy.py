from typing import Any, Dict, Union, List
import talib
import numpy as np
import asyncio
import pandas as pd
from data_manager import DataManager
from config import Config
import logging
from MLStrat import MLStrategy

class BaseStrategy:
    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        raise NotImplementedError("Subclasses must implement should_enter_trade")
    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        raise NotImplementedError("Subclasses must implement should_exit_trade")

class StrategyManager:
    def __init__(self, data_manager: DataManager, api, logger: logging.Logger):
        if not isinstance(data_manager, DataManager):
            raise TypeError("Requires initialized DataManager instance")
        self.api = api
        self.data_manager = data_manager
        self.config = data_manager.config
        self._validate_config()
        self.symbols = self.config.SYMBOLS
        self.logger = logger
        try:
            self._price_groups = data_manager._symbol_groups
            self.logger.debug(f"Initialized _price_groups: {list(self._price_groups.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to access _symbol_groups: {e}")
            raise
        self._current_indicators = {}
        self._indicator_signals = {}
        self._indicators_key = []
        self._indicators_comp_key = []
        self.performance_metrics = {
            name: {'trades': 0, 'wins': 0, 'total_return': 0.0, 'total_pips': 0.0}
            for name in ['RSIStrategy', 'MACDStrategy', 'MLStrategy']
        }

        # Construct DataFrame with symbol and timestamp MultiIndex
        dfs = []
        for symbol, df in data_manager.data.items():
            if df.empty:
                continue
            # Ensure timestamp index and add symbol column
            df = df.copy()
            df['symbol'] = symbol
            dfs.append(df)
        
        if not dfs:
            self.logger.warning("No DataFrames available to concatenate.")
            self._frame = pd.DataFrame({
                'symbol': [],
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            }).set_index(['symbol', 'timestamp'])
        else:
            self._frame = pd.concat(dfs)
            self._frame = self._frame.reset_index().set_index(['symbol', 'timestamp'])
            self.logger.debug(f"Concatenated DataFrame: shape={self._frame.shape}, index={self._frame.index.names}")

        self.strategies = [
            RSIStrategy(self.config, logger),
            MACDStrategy(self.config, logger),
            MLStrategy(self.config, logger)
        ]
        self.logger.debug(f"Initialized strategies: {[s.__class__.__name__ for s in self.strategies]}")
        self.strategy_weights = {'RSIStrategy': 0.3, 'MACDStrategy': 0.3, 'MLStrategy': 0.4}

    def _validate_config(self) -> None:
        required = [
            'SYMBOLS', 'STOP_LOSS_PIPS', 'TAKE_PROFIT_PIPS', 'TRAILING_STOP_PIPS',
            'RISK_PERCENTAGE', 'ML_N_ESTIMATORS', 'ML_MAX_DEPTH', 'ML_MIN_SAMPLES'
        ]
        for param in required:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing config parameter: {param}")

    def _pips_to_percentage(self, pips: int) -> float:
        return pips * 0.0001

    async def should_enter_trade(self, symbol: str) -> bool:
        results = await asyncio.gather(
            *(strategy.should_enter_trade(symbol, self.data_manager) for strategy in self.strategies),
            return_exceptions=True
        )
        score = 0.0
        for strategy, result in zip(self.strategies, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error in {strategy.__class__.__name__}: {result}")
                continue
            if result:
                self.logger.debug(f"{strategy.__class__.__name__} recommends entering trade for {symbol}")
                score += self.strategy_weights[strategy.__class__.__name__]
        return score > 0.5

    async def should_exit_trade(self, symbol: str) -> bool:
        results = await asyncio.gather(
            *(strategy.should_exit_trade(symbol, self.data_manager) for strategy in self.strategies),
            return_exceptions=True
        )
        score = 0.0
        for strategy, result in zip(self.strategies, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error in {strategy.__class__.__name__}: {result}")
                continue
            if result:
                self.logger.debug(f"{strategy.__class__.__name__} recommends exiting trade for {symbol}")
                score += self.strategy_weights[strategy.__class__.__name__]
        return score > 0.5

    def get_indicator_signal(self, indicator: str = None) -> Dict:
        if indicator and indicator in self._indicator_signals:
            return self._indicator_signals[indicator]
        return self._indicator_signals

    def set_indicator_signal(
        self,
        indicator: str,
        buy: float,
        sell: float,
        condition_buy: Any,
        condition_sell: Any,
        buy_max: float = None,
        sell_max: float = None,
        condition_buy_max: Any = None,
        condition_sell_max: Any = None,
    ) -> None:
        if indicator not in self._indicator_signals:
            self._indicator_signals[indicator] = {}
            self._indicators_key.append(indicator)
        self._indicator_signals[indicator]["buy"] = buy
        self._indicator_signals[indicator]["sell"] = sell
        self._indicator_signals[indicator]["buy_operator"] = condition_buy
        self._indicator_signals[indicator]["sell_operator"] = condition_sell
        self._indicator_signals[indicator]["buy_max"] = buy_max
        self._indicator_signals[indicator]["sell_max"] = sell_max
        self._indicator_signals[indicator]["buy_operator_max"] = condition_buy_max
        self._indicator_signals[indicator]["sell_operator_max"] = condition_sell_max

    def set_indicator_signal_compare(
        self,
        indicator_1: str,
        indicator_2: str,
        condition_buy: Any,
        condition_sell: Any,
    ) -> None:
        key = f"{indicator_1}_comp_{indicator_2}"
        if key not in self._indicator_signals:
            self._indicator_signals[key] = {}
            self._indicators_comp_key.append(key)
        indicator_dict = self._indicator_signals[key]
        indicator_dict["type"] = "comparison"
        indicator_dict["indicator_1"] = indicator_1
        indicator_dict["indicator_2"] = indicator_2
        indicator_dict["buy_operator"] = condition_buy
        indicator_dict["sell_operator"] = condition_sell

    @property
    def price_data_frame(self) -> pd.DataFrame:
        return self._frame

    @price_data_frame.setter
    def price_data_frame(self, price_data_frame: pd.DataFrame) -> None:
        if not isinstance(price_data_frame.index, pd.MultiIndex) or price_data_frame.index.names != ['symbol', 'timestamp']:
            price_data_frame = price_data_frame.reset_index().set_index(['symbol', 'timestamp'])
        self._frame = price_data_frame

    @property
    def is_multi_index(self) -> bool:
        return isinstance(self._frame.index, pd.MultiIndex) and self._frame.index.names == ['symbol', 'timestamp']

    def change_in_price(self, column_name: str = "change_in_price") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"column_name": column_name},
            "func": self.change_in_price
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in change_in_price")
            return self._frame
        self._frame[column_name] = self._frame.groupby('symbol')['close'].transform(lambda x: x.diff())
        return self._frame

    def rsi(self, period: int, method: str = "wilders", column_name: str = "rsi") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "method": method, "column_name": column_name},
            "func": self.rsi
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in rsi")
            return self._frame
        if "change_in_price" not in self._frame.columns:
            self.change_in_price()
        self._frame["up_day"] = self._frame.groupby('symbol')['change_in_price'].transform(lambda x: np.where(x >= 0, x, 0))
        self._frame["down_day"] = self._frame.groupby('symbol')['change_in_price'].transform(lambda x: np.where(x < 0, x.abs(), 0))
        self._frame["ewma_up"] = self._frame.groupby('symbol')["up_day"].transform(lambda x: x.ewm(span=period).mean())
        self._frame["ewma_down"] = self._frame.groupby('symbol')["down_day"].transform(lambda x: x.ewm(span=period).mean())
        relative_strength = self._frame["ewma_up"] / self._frame["ewma_down"]
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))
        self._frame["rsi"] = np.where(relative_strength_index == 0, 100, relative_strength_index)
        self._frame.drop(labels=["ewma_up", "ewma_down", "down_day", "up_day", "change_in_price"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def sma(self, period: int, column_name: str = "sma") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.sma
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in sma")
            return self._frame
        self._frame[column_name] = self._frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window=period).mean())
        return self._frame

    def ema(self, period: int, alpha: float = 0.0, column_name: str = "ema") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "alpha": alpha, "column_name": column_name},
            "func": self.ema
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in ema")
            return self._frame
        self._frame[column_name] = self._frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=period).mean())
        return self._frame

    def rate_of_change(self, period: int = 1, column_name: str = "rate_of_change") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.rate_of_change
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in rate_of_change")
            return self._frame
        self._frame[column_name] = self._frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(periods=period))
        return self._frame

    def bollinger_bands(self, period: int = 20, column_name: str = "bollinger_bands") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.bollinger_bands
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in bollinger_bands")
            return self._frame
        self._frame["moving_avg"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window=period).mean())
        self._frame["moving_std"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window=period).std())
        self._frame["band_upper"] = self._frame["moving_avg"] + (2 * self._frame["moving_std"])
        self._frame["band_lower"] = self._frame["moving_avg"] - (2 * self._frame["moving_std"])
        self._frame.drop(labels=["moving_avg", "moving_std"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def average_true_range(self, period: int = 14, column_name: str = "average_true_range") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.average_true_range
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in average_true_range")
            return self._frame
        self._frame["true_range_0"] = abs(self._frame["high"] - self._frame["low"])
        self._frame["true_range_1"] = abs(self._frame["high"] - self._frame["close"].shift())
        self._frame["true_range_2"] = abs(self._frame["low"] - self._frame["close"].shift())
        self._frame["true_range"] = self._frame[["true_range_0", "true_range_1", "true_range_2"]].max(axis=1)
        self._frame["average_true_range"] = self._frame.groupby('symbol')["true_range"].transform(lambda x: x.ewm(span=period, min_periods=period).mean())
        self._frame.drop(labels=["true_range_0", "true_range_1", "true_range_2", "true_range"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def stochastic_oscillator(self, column_name: str = "stochastic_oscillator") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"column_name": column_name},
            "func": self.stochastic_oscillator
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in stochastic_oscillator")
            return self._frame
        denominator = self._frame["high"] - self._frame["low"]
        self._frame["stochastic_oscillator"] = 100 * ((self._frame["close"] - self._frame["low"]) / denominator.where(denominator != 0, 1))
        return self._frame

    def macd(self, fast_period: int = 12, slow_period: int = 26, column_name: str = "macd") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"fast_period": fast_period, "slow_period": slow_period, "column_name": column_name},
            "func": self.macd
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in macd")
            return self._frame
        self._frame["macd_fast"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=fast_period, min_periods=fast_period).mean())
        self._frame["macd_slow"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=slow_period, min_periods=slow_period).mean())
        self._frame["macd_diff"] = self._frame["macd_fast"] - self._frame["macd_slow"]
        self._frame["macd"] = self._frame.groupby('symbol')["macd_diff"].transform(lambda x: x.ewm(span=9, min_periods=8).mean())
        self._frame["macd_signal"] = self._frame.groupby('symbol')["macd"].transform(lambda x: x.ewm(span=9).mean())
        self._frame.drop(labels=["macd_fast", "macd_slow", "macd_diff"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def mass_index(self, period: int = 9, column_name: str = "mass_index") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.mass_index
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in mass_index")
            return self._frame
        self._frame["diff"] = self._frame["high"] - self._frame["low"]
        self._frame["mass_index_1"] = self._frame.groupby('symbol')["diff"].transform(lambda x: x.ewm(span=period, min_periods=period - 1).mean())
        self._frame["mass_index_2"] = self._frame.groupby('symbol')["mass_index_1"].transform(lambda x: x.ewm(span=period, min_periods=period - 1).mean())
        self._frame["mass_index_raw"] = self._frame["mass_index_1"] / self._frame["mass_index_2"]
        self._frame["mass_index"] = self._frame.groupby('symbol')["mass_index_raw"].transform(lambda x: x.rolling(window=25).sum())
        self._frame.drop(labels=["diff", "mass_index_1", "mass_index_2", "mass_index_raw"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def force_index(self, period: int, column_name: str = "force_index") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.force_index
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in force_index")
            return self._frame
        self._frame[column_name] = self._frame.groupby('symbol')['close'].transform(lambda x: x.diff(period)) * self._frame.groupby('symbol')['volume'].transform(lambda x: x.diff(period))
        return self._frame

    def ease_of_movement(self, period: int, column_name: str = "ease_of_movement") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.ease_of_movement
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in ease_of_movement")
            return self._frame
        high_plus_low = self._frame["high"].diff(1) + self._frame["low"].diff(1)
        diff_divi_vol = (self._frame["high"] - self._frame["low"]) / (2 * self._frame["volume"].where(self._frame["volume"] != 0, 1))
        self._frame["ease_of_movement_raw"] = high_plus_low * diff_divi_vol
        self._frame["ease_of_movement"] = self._frame.groupby('symbol')["ease_of_movement_raw"].transform(lambda x: x.rolling(window=period).mean())
        self._frame.drop(labels=["ease_of_movement_raw"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def commodity_channel_index(self, period: int, column_name: str = "commodity_channel_index") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.commodity_channel_index
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in commodity_channel_index")
            return self._frame
        self._frame["typical_price"] = (self._frame["high"] + self._frame["low"] + self._frame["close"]) / 3
        self._frame["typical_price_mean"] = self._frame.groupby('symbol')["typical_price"].transform(lambda x: x.rolling(window=period).mean())
        self._frame["mean_deviation"] = self._frame.groupby('symbol')["typical_price"].transform(lambda x: (x - x.rolling(window=period).mean()).abs().rolling(window=period).mean())
        self._frame[column_name] = (self._frame["typical_price"] - self._frame["typical_price_mean"]) / (0.015 * self._frame["mean_deviation"].where(self._frame["mean_deviation"] != 0, 1))
        self._frame.drop(labels=["typical_price", "typical_price_mean", "mean_deviation"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def standard_deviation(self, period: int, column_name: str = "standard_deviation") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.standard_deviation
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in standard_deviation")
            return self._frame
        self._frame[column_name] = self._frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=period).std())
        return self._frame

    def chaikin_oscillator(self, period: int, column_name: str = "chaikin_oscillator") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"period": period, "column_name": column_name},
            "func": self.chaikin_oscillator
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in chaikin_oscillator")
            return self._frame
        money_flow_multiplier = ((self._frame["close"] - self._frame["low"]) - (self._frame["high"] - self._frame["close"])) / self._frame["high"].where(self._frame["high"] != self._frame["low"], 1)
        self._frame["money_flow_volume"] = money_flow_multiplier * self._frame["volume"]
        self._frame["money_flow_volume_3"] = self._frame.groupby('symbol')["money_flow_volume"].transform(lambda x: x.ewm(span=3, min_periods=2).mean())
        self._frame["money_flow_volume_10"] = self._frame.groupby('symbol')["money_flow_volume"].transform(lambda x: x.ewm(span=10, min_periods=9).mean())
        self._frame["chaikin_oscillator"] = self._frame["money_flow_volume_3"] - self._frame["money_flow_volume_10"]
        self._frame.drop(labels=["money_flow_volume", "money_flow_volume_3", "money_flow_volume_10"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def kst_oscillator(self, r1: int, r2: int, r3: int, r4: int, n1: int, n2: int, n3: int, n4: int, column_name: str = "kst_oscillator") -> pd.DataFrame:
        self._current_indicators[column_name] = {
            "args": {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "n1": n1, "n2": n2, "n3": n3, "n4": n4, "column_name": column_name},
            "func": self.kst_oscillator
        }
        if self._frame.empty:
            self.logger.warning("Empty DataFrame in kst_oscillator")
            return self._frame
        self._frame["roc_1"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.diff(r1 - 1) / x.shift(r1 - 1))
        self._frame["roc_2"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.diff(r2 - 1) / x.shift(r2 - 1))
        self._frame["roc_3"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.diff(r3 - 1) / x.shift(r3 - 1))
        self._frame["roc_4"] = self._frame.groupby('symbol')['close'].transform(lambda x: x.diff(r4 - 1) / x.shift(r4 - 1))
        self._frame["roc_1_n"] = self._frame.groupby('symbol')["roc_1"].transform(lambda x: x.rolling(window=n1).sum())
        self._frame["roc_2_n"] = self._frame.groupby('symbol')["roc_2"].transform(lambda x: x.rolling(window=n2).sum())
        self._frame["roc_3_n"] = self._frame.groupby('symbol')["roc_3"].transform(lambda x: x.rolling(window=n3).sum())
        self._frame["roc_4_n"] = self._frame.groupby('symbol')["roc_4"].transform(lambda x: x.rolling(window=n4).sum())
        self._frame[column_name] = 100 * (self._frame["roc_1_n"] + 2 * self._frame["roc_2_n"] + 3 * self._frame["roc_3_n"] + 4 * self._frame["roc_4_n"])
        self._frame[column_name + "_signal"] = self._frame.groupby('symbol')[column_name].transform(lambda x: x.rolling(window=9).mean())
        self._frame.drop(labels=["roc_1", "roc_2", "roc_3", "roc_4", "roc_1_n", "roc_2_n", "roc_3_n", "roc_4_n"], axis=1, inplace=True, errors='ignore')
        return self._frame

    def refresh(self):
        try:
            self._price_groups = self.data_manager._symbol_groups
            self.logger.debug(f"Refreshed _price_groups: {list(self._price_groups.keys())}")
            # Rebuild _frame with updated data
            dfs = []
            for symbol, df in self.data_manager.data.items():
                if df.empty:
                    continue
                df = df.copy()
                df['symbol'] = symbol
                dfs.append(df)
            if dfs:
                self._frame = pd.concat(dfs)
                self._frame = self._frame.reset_index().set_index(['symbol', 'timestamp'])
                self.logger.debug(f"Refreshed DataFrame: shape={self._frame.shape}, index={self._frame.index.names}")
        except Exception as e:
            self.logger.error(f"Failed to refresh _price_groups: {e}")
            raise
        for indicator, details in self._current_indicators.items():
            details["func"](**details["args"])

    def check_signals(self) -> Union[pd.DataFrame, None]:
        return None

    async def optimize_parameters(self, symbol: str) -> None:
        for strategy in self.strategies:
            if hasattr(strategy, 'optimize_parameters'):
                params = await strategy.optimize_parameters(self.data_manager, symbol)
                self.logger.info(f"Optimized {strategy.__class__.__name__} parameters: {params}")

    def _record_trade(self, strategy_name: str, symbol: str, exit_price: float, entry_price: float) -> None:
        profit = (exit_price - entry_price) / entry_price
        pips = (exit_price - entry_price) / 0.0001
        self.performance_metrics[strategy_name]['trades'] += 1
        self.performance_metrics[strategy_name]['wins'] += 1 if profit > 0 else 0
        self.performance_metrics[strategy_name]['total_return'] += profit
        self.performance_metrics[strategy_name]['total_pips'] += pips
        self._log_performance(strategy_name, symbol)

    def _log_performance(self, strategy_name: str, symbol: str) -> None:
        win_rate = self.performance_metrics[strategy_name]['wins'] / self.performance_metrics[strategy_name]['trades'] if self.performance_metrics[strategy_name]['trades'] > 0 else 0
        avg_return = self.performance_metrics[strategy_name]['total_return'] / self.performance_metrics[strategy_name]['trades'] if self.performance_metrics[strategy_name]['trades'] > 0 else 0
        avg_pips = self.performance_metrics[strategy_name]['total_pips'] / self.performance_metrics[strategy_name]['trades'] if self.performance_metrics[strategy_name]['trades'] > 0 else 0
        self.logger.info(f"{strategy_name} {symbol} - Win Rate: {win_rate:.2f}, Avg Return: {avg_return:.2f}, Avg Pips: {avg_pips:.2f}")
        with open('performance.csv', 'a') as f:
            f.write(f"{strategy_name},{symbol},{win_rate:.2f},{avg_return:.2f},{avg_pips:.2f}\n")
        if win_rate < 0.3:
            self.logger.warning(f"Low win rate for {strategy_name} on {symbol}: {win_rate:.2f}")

class RSIStrategy(BaseStrategy):
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.rsi_period = getattr(config, 'RSI_PERIOD', 14)
        self.rsi_overbought = getattr(config, 'RSI_OVERBOUGHT', 70)
        self.rsi_oversold = getattr(config, 'RSI_OVERSOLD', 30)
        self.stop_loss = self._pips_to_percentage(config.STOP_LOSS_PIPS)
        self.take_profit = self._pips_to_percentage(config.TAKE_PROFIT_PIPS)
        self.trailing_stop = self._pips_to_percentage(config.TRAILING_STOP_PIPS)
        self.risk_percentage = config.RISK_PERCENTAGE
        self.last_trade_price = {}
        self.highest_price = {}

    def _pips_to_percentage(self, pips: int) -> float:
        return pips * 0.0001

    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.rsi_period:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
            return False
        try:
            sm = StrategyManager(data_manager, None, self.logger)
            df = sm.rsi(period=self.rsi_period)
            if 'rsi' not in df.columns or df['rsi'].isna().all():
                self.logger.error(f"RSI calculation failed for {symbol}")
                return False
            df_symbol = df.loc[symbol]
            current_price = df_symbol['close'].iloc[-1]
            rsi = df_symbol['rsi'].iloc[-1]
            if rsi < self.rsi_oversold:
                if current_price * (1 - self.stop_loss) < self._get_last_trade_price(symbol):
                    self.logger.debug(f"Stop-loss triggered for {symbol}")
                    return False
                self.last_trade_price[symbol] = current_price
                self.highest_price[symbol] = current_price
                return True
            return False
        except Exception as e:
            self.logger.error(f"RSI calculation error for {symbol}: {e}")
            return False

    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.rsi_period:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
            return False
        try:
            sm = StrategyManager(data_manager, None, self.logger)
            df = sm.rsi(period=self.rsi_period)
            if 'rsi' not in df.columns or df['rsi'].isna().all():
                self.logger.error(f"RSI calculation failed for {symbol}")
                return False
            df_symbol = df.loc[symbol]
            current_price = df_symbol['close'].iloc[-1]
            rsi = df_symbol['rsi'].iloc[-1]
            entry_price = self._get_last_trade_price(symbol)
            self.highest_price[symbol] = max(self.highest_price.get(symbol, current_price), current_price)
            if current_price >= entry_price * (1 + self.take_profit):
                sm._record_trade('RSIStrategy', symbol, current_price, entry_price)
                return True
            if current_price <= self.highest_price[symbol] * (1 - self.trailing_stop):
                sm._record_trade('RSIStrategy', symbol, current_price, entry_price)
                return True
            if rsi > self.rsi_overbought:
                sm._record_trade('RSIStrategy', symbol, current_price, entry_price)
                return True
            return False
        except Exception as e:
            self.logger.error(f"RSI exit calculation error for {symbol}: {e}")
            return False

    async def optimize_parameters(self, data_manager: DataManager, symbol: str) -> dict:
        best_score = -float('inf')
        best_params = {}
        for rsi_period in range(5, 30, 5):
            for rsi_overbought in range(60, 80, 5):
                for rsi_oversold in range(20, 40, 5):
                    self.rsi_period = rsi_period
                    self.rsi_overbought = rsi_overbought
                    self.rsi_oversold = rsi_oversold
                    score = await self._evaluate_performance(data_manager, symbol)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'rsi_period': rsi_period,
                            'rsi_overbought': rsi_overbought,
                            'rsi_oversold': rsi_oversold
                        }
        return best_params

    async def _evaluate_performance(self, data_manager: DataManager, symbol: str) -> float:
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.rsi_period:
            return 0
        equity = 10000
        for i in range(self.rsi_period, len(df)):
            temp_sm = StrategyManager(data_manager, None, self.logger)
            temp_df = temp_sm.rsi(period=self.rsi_period)
            if 'rsi' not in temp_df.columns or temp_df['rsi'].isna().all():
                continue
            df_symbol = temp_df.loc[symbol]
            if df_symbol['rsi'].iloc[i] < self.rsi_oversold:
                entry_price = df_symbol['close'].iloc[i]
                for j in range(i + 1, len(df_symbol)):
                    temp_df = temp_sm.rsi(period=self.rsi_period)
                    df_symbol_j = temp_df.loc[symbol]
                    if 'rsi' not in df_symbol_j.columns or df_symbol_j['rsi'].isna().all():
                        break
                    if df_symbol_j['rsi'].iloc[j] > self.rsi_overbought:
                        exit_price = df_symbol_j['close'].iloc[j]
                        equity *= (1 + (exit_price - entry_price) / entry_price)
                        break
        return equity / 10000 - 1

    def _get_last_trade_price(self, symbol: str) -> float:
        return self.last_trade_price.get(symbol, 1.0)

class MACDStrategy(BaseStrategy):
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.fast_period = getattr(config, 'MACD_FAST_PERIOD', 12)
        self.slow_period = getattr(config, 'MACD_SLOW_PERIOD', 26)
        self.signal_period = getattr(config, 'MACD_SIGNAL_PERIOD', 9)
        self.stop_loss = self._pips_to_percentage(config.STOP_LOSS_PIPS)
        self.take_profit = self._pips_to_percentage(config.TAKE_PROFIT_PIPS)
        self.trailing_stop = self._pips_to_percentage(config.TRAILING_STOP_PIPS)
        self.risk_percentage = config.RISK_PERCENTAGE
        self.last_trade_price = {}
        self.highest_price = {}

    def _pips_to_percentage(self, pips: int) -> float:
        return pips * 0.0001

    async def should_enter_trade(self, symbol: str, data_manager: DataManager) -> bool:
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.slow_period:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
            return False
        try:
            sm = StrategyManager(data_manager, None, self.logger)
            df = sm.macd(fast_period=self.fast_period, slow_period=self.slow_period)
            if 'macd' not in df.columns or 'macd_signal' not in df.columns or df['macd'].isna().all():
                self.logger.error(f"MACD calculation failed for {symbol}")
                return False
            df_symbol = df.loc[symbol]
            current_price = df_symbol['close'].iloc[-1]
            macd = df_symbol['macd'].iloc[-1]
            signal = df_symbol['macd_signal'].iloc[-1]
            macd_prev = df_symbol['macd'].iloc[-2]
            signal_prev = df_symbol['macd_signal'].iloc[-2]
            if macd > signal and macd_prev <= signal_prev:
                if current_price * (1 - self.stop_loss) < self._get_last_trade_price(symbol):
                    self.logger.debug(f"Stop-loss triggered for {symbol}")
                    return False
                self.last_trade_price[symbol] = current_price
                self.highest_price[symbol] = current_price
                return True
            return False
        except Exception as e:
            self.logger.error(f"MACD calculation error for {symbol}: {e}")
            return False

    async def should_exit_trade(self, symbol: str, data_manager: DataManager) -> bool:
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.slow_period:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
            return False
        try:
            sm = StrategyManager(data_manager, None, self.logger)
            df = sm.macd(fast_period=self.fast_period, slow_period=self.slow_period)
            if 'macd' not in df.columns or 'macd_signal' not in df.columns or df['macd'].isna().all():
                self.logger.error(f"MACD calculation failed for {symbol}")
                return False
            df_symbol = df.loc[symbol]
            current_price = df_symbol['close'].iloc[-1]
            macd = df_symbol['macd'].iloc[-1]
            signal = df_symbol['macd_signal'].iloc[-1]
            macd_prev = df_symbol['macd'].iloc[-2]
            signal_prev = df_symbol['macd_signal'].iloc[-2]
            entry_price = self._get_last_trade_price(symbol)
            self.highest_price[symbol] = max(self.highest_price.get(symbol, current_price), current_price)
            if current_price >= entry_price * (1 + self.take_profit):
                sm._record_trade('MACDStrategy', symbol, current_price, entry_price)
                return True
            if current_price <= self.highest_price[symbol] * (1 - self.trailing_stop):
                sm._record_trade('MACDStrategy', symbol, current_price, entry_price)
                return True
            if macd < signal and macd_prev >= signal_prev:
                sm._record_trade('MACDStrategy', symbol, current_price, entry_price)
                return True
            return False
        except Exception as e:
            self.logger.error(f"MACD exit calculation error for {symbol}: {e}")
            return False

    async def optimize_parameters(self, data_manager: DataManager, symbol: str) -> dict:
        best_score = -float('inf')
        best_params = {}
        for fast_period in range(8, 16, 2):
            for slow_period in range(20, 30, 2):
                for signal_period in range(7, 11):
                    self.fast_period = fast_period
                    self.slow_period = slow_period
                    self.signal_period = signal_period
                    score = await self._evaluate_performance(data_manager, symbol)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'signal_period': signal_period
                        }
        return best_params

    async def _evaluate_performance(self, data_manager: DataManager, symbol: str) -> float:
        df = data_manager.get_snapshot(symbol)
        if df is None or len(df) < self.slow_period:
            return 0
        equity = 10000
        for i in range(self.slow_period, len(df)):
            temp_sm = StrategyManager(data_manager, None, self.logger)
            temp_df = temp_sm.macd(fast_period=self.fast_period, slow_period=self.slow_period)
            if 'macd' not in temp_df.columns or 'macd_signal' not in temp_df.columns or temp_df['macd'].isna().all():
                continue
            df_symbol = temp_df.loc[symbol]
            if df_symbol['macd'].iloc[i] > df_symbol['macd_signal'].iloc[i] and df_symbol['macd'].iloc[i-1] <= df_symbol['macd_signal'].iloc[i-1]:
                entry_price = df_symbol['close'].iloc[i]
                for j in range(i + 1, len(df_symbol)):
                    temp_df = temp_sm.macd(fast_period=self.fast_period, slow_period=self.slow_period)
                    df_symbol_j = temp_df.loc[symbol]
                    if 'macd' not in df_symbol_j.columns or 'macd_signal' not in df_symbol_j.columns or df_symbol_j['macd'].isna().all():
                        break
                    if df_symbol_j['macd'].iloc[j] < df_symbol_j['macd_signal'].iloc[j] and df_symbol_j['macd'].iloc[j-1] >= df_symbol_j['macd_signal'].iloc[j-1]:
                        exit_price = df_symbol_j['close'].iloc[j]
                        equity *= (1 + (exit_price - entry_price) / entry_price)
                        break
        return equity / 10000 - 1

    def _get_last_trade_price(self, symbol: str) -> float:
        return self.last_trade_price.get(symbol, 1.0)
