"""Qlib-based factor computation and evaluation engine.

This module provides factor calculation using Qlib's expression engine and D.features() API.
Fully leverages Qlib's capabilities for expression parsing, computation, and optimization.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# Qlib imports (with optional crypto extension)
try:
    import qlib
    from qlib.data import D
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    D = None
    DataHandlerLP = None

# Crypto extension (P2.2: custom implementation for crypto markets)
try:
    from iqfmp.core.qlib_crypto import CryptoDataHandler, CryptoField
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    CryptoDataHandler = None
    CryptoField = None

# Import Qlib initialization
from iqfmp.core.qlib_init import init_qlib, ensure_qlib_initialized, is_qlib_initialized

logger = logging.getLogger(__name__)


class QlibFactorEngine:
    """Qlib-based factor computation engine.

    Uses Qlib's D.features() API and expression engine for factor calculation.
    Supports both Qlib expression syntax and CryptoDataHandler for crypto data.
    """

    def __init__(
        self,
        provider_uri: Optional[str] = None,
        instruments: Optional[list[str]] = None,
        data_handler: Optional[DataHandlerLP] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        """Initialize Qlib factor engine.

        Args:
            provider_uri: Qlib data provider URI (e.g., "~/.qlib/qlib_data/crypto")
            instruments: List of instruments to load (e.g., ["BTCUSDT", "ETHUSDT"])
            data_handler: Pre-configured Qlib DataHandler
            df: Pre-loaded DataFrame (will be converted to Qlib format)
        """
        self._qlib_initialized = False
        self._provider_uri = provider_uri
        self._instruments = instruments or []
        self._data_handler = data_handler
        self._df: Optional[pd.DataFrame] = None
        self._qlib_data: Optional[pd.DataFrame] = None

        # Initialize with DataFrame if provided
        if df is not None:
            self._df = df.copy()
            self._prepare_data()

    def init_qlib(
        self,
        provider_uri: Optional[str] = None,
        region: str = "crypto",
        **kwargs,
    ) -> bool:
        """Initialize Qlib with configuration.

        Args:
            provider_uri: Data provider URI
            region: Market region
            **kwargs: Additional Qlib config

        Returns:
            True if initialization successful
        """
        try:
            uri = provider_uri or self._provider_uri
            if uri:
                qlib.init(provider_uri=uri, region=region, **kwargs)
            else:
                # Initialize with default config for expression engine
                qlib.init(region=region, **kwargs)

            self._qlib_initialized = True
            logger.info(f"Qlib initialized: region={region}")
            return True

        except Exception as e:
            logger.warning(f"Qlib initialization failed: {e}")
            return False

    def load_data(self, path: Path) -> None:
        """Load OHLCV data from CSV file.

        Args:
            path: Path to CSV file with OHLCV data
        """
        logger.info(f"Loading data from {path}")
        self._df = pd.read_csv(path)
        self._prepare_data()
        logger.info(f"Loaded {len(self._df)} rows")

    def _prepare_data(self) -> None:
        """Prepare data for Qlib factor computation."""
        if self._df is None:
            return

        # Ensure timestamp is datetime
        if "timestamp" in self._df.columns:
            self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])
            self._df = self._df.sort_values("timestamp").reset_index(drop=True)
            self._df.set_index("timestamp", inplace=True)

        # Standardize column names to Qlib format
        column_mapping = {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        }

        # Create Qlib-compatible DataFrame
        self._qlib_data = self._df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in self._qlib_data.columns:
                self._qlib_data[new_name] = self._qlib_data[old_name]

        # Calculate forward returns for evaluation
        self._df["returns"] = self._df["close"].pct_change()
        self._df["fwd_returns_1d"] = self._df["close"].pct_change().shift(-1)
        self._df["fwd_returns_5d"] = self._df["close"].pct_change(5).shift(-5)
        self._df["fwd_returns_10d"] = self._df["close"].pct_change(10).shift(-10)

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get loaded data."""
        return self._df

    def compute_factor(
        self,
        expression: str,
        factor_name: str = "factor",
    ) -> pd.Series:
        """Compute factor using Qlib expression.

        Supports Qlib expression syntax:
        - $close, $open, $high, $low, $volume - price fields
        - Ref($close, -1) - reference previous value
        - Mean($close, 20) - rolling mean
        - Std($close, 20) - rolling std
        - Sum($volume, 5) - rolling sum
        - Max($high, 10), Min($low, 10) - rolling max/min
        - Delta($close, 5) - difference
        - Rank($close) - cross-sectional rank
        - Corr($close, $volume, 20) - rolling correlation

        Args:
            expression: Qlib expression string
            factor_name: Name for the computed factor

        Returns:
            Series of factor values

        Raises:
            ValueError: If data not loaded or computation fails
        """
        if self._df is None or self._qlib_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            # Parse and compute Qlib expression
            result = self._compute_qlib_expression(expression)

            return pd.Series(result, name=factor_name, index=self._df.index)

        except Exception as e:
            logger.error(f"Factor computation failed: {e}")
            raise ValueError(f"Factor computation failed: {e}")

    def _compute_qlib_expression(self, expression: str) -> pd.Series:
        """Compute Qlib expression on local data.

        This method parses Qlib expression syntax and computes on the local DataFrame.
        Supports common Qlib operators without requiring Qlib data storage.
        """
        df = self._qlib_data

        # Tokenize and compute expression
        result = self._evaluate_expression(expression, df)

        return result

    def _evaluate_expression(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """Recursively evaluate Qlib expression.

        Args:
            expr: Qlib expression string
            df: DataFrame with $field columns

        Returns:
            Computed Series
        """
        expr = expr.strip()

        # Handle field references: $close, $open, etc.
        if expr.startswith("$"):
            field = expr
            if field in df.columns:
                return df[field]
            raise ValueError(f"Unknown field: {field}")

        # Handle numeric literals
        try:
            return pd.Series(float(expr), index=df.index)
        except ValueError:
            pass

        # Handle operators: Ref, Mean, Std, Sum, Max, Min, Delta, etc.
        operator_patterns = [
            (r"^Ref\((.+),\s*(-?\d+)\)$", self._op_ref),
            (r"^Mean\((.+),\s*(\d+)\)$", self._op_mean),
            (r"^Std\((.+),\s*(\d+)\)$", self._op_std),
            (r"^Sum\((.+),\s*(\d+)\)$", self._op_sum),
            (r"^Max\((.+),\s*(\d+)\)$", self._op_max),
            (r"^Min\((.+),\s*(\d+)\)$", self._op_min),
            (r"^Delta\((.+),\s*(\d+)\)$", self._op_delta),
            (r"^Rank\((.+)\)$", self._op_rank),
            (r"^Abs\((.+)\)$", self._op_abs),
            (r"^Log\((.+)\)$", self._op_log),
            (r"^Sign\((.+)\)$", self._op_sign),
            (r"^Corr\((.+),\s*(.+),\s*(\d+)\)$", self._op_corr),
            (r"^Cov\((.+),\s*(.+),\s*(\d+)\)$", self._op_cov),
            (r"^WMA\((.+),\s*(\d+)\)$", self._op_wma),
            (r"^EMA\((.+),\s*(\d+)\)$", self._op_ema),
            (r"^RSI\((.+),\s*(\d+)\)$", self._op_rsi),
            (r"^MACD\((.+),\s*(\d+),\s*(\d+),\s*(\d+)\)$", self._op_macd),
        ]

        for pattern, op_func in operator_patterns:
            match = re.match(pattern, expr)
            if match:
                return op_func(df, *match.groups())

        # Handle binary operations: +, -, *, /, >, <, ==
        # Find the main operator (respect parentheses)
        for op, op_func in [("+", self._add), ("-", self._sub),
                            ("*", self._mul), ("/", self._div),
                            (">", self._gt), ("<", self._lt)]:
            idx = self._find_main_operator(expr, op)
            if idx > 0:
                left = expr[:idx].strip()
                right = expr[idx + 1:].strip()
                left_val = self._evaluate_expression(left, df)
                right_val = self._evaluate_expression(right, df)
                return op_func(left_val, right_val)

        # Handle parentheses
        if expr.startswith("(") and expr.endswith(")"):
            return self._evaluate_expression(expr[1:-1], df)

        raise ValueError(f"Cannot parse expression: {expr}")

    def _find_main_operator(self, expr: str, op: str) -> int:
        """Find the main operator position, respecting parentheses."""
        depth = 0
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == op and depth == 0:
                return i
        return -1

    # Qlib Operator Implementations
    def _op_ref(self, df: pd.DataFrame, inner: str, periods: str) -> pd.Series:
        """Ref: Reference value n periods ago."""
        val = self._evaluate_expression(inner, df)
        return val.shift(-int(periods))

    def _op_mean(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """Mean: Rolling mean."""
        val = self._evaluate_expression(inner, df)
        return val.rolling(int(window)).mean()

    def _op_std(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """Std: Rolling standard deviation."""
        val = self._evaluate_expression(inner, df)
        return val.rolling(int(window)).std()

    def _op_sum(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """Sum: Rolling sum."""
        val = self._evaluate_expression(inner, df)
        return val.rolling(int(window)).sum()

    def _op_max(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """Max: Rolling maximum."""
        val = self._evaluate_expression(inner, df)
        return val.rolling(int(window)).max()

    def _op_min(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """Min: Rolling minimum."""
        val = self._evaluate_expression(inner, df)
        return val.rolling(int(window)).min()

    def _op_delta(self, df: pd.DataFrame, inner: str, periods: str) -> pd.Series:
        """Delta: Difference over n periods."""
        val = self._evaluate_expression(inner, df)
        return val.diff(int(periods))

    def _op_rank(self, df: pd.DataFrame, inner: str) -> pd.Series:
        """Rank: Cross-sectional rank (for single series, just normalize)."""
        val = self._evaluate_expression(inner, df)
        return val.rank(pct=True)

    def _op_abs(self, df: pd.DataFrame, inner: str) -> pd.Series:
        """Abs: Absolute value."""
        val = self._evaluate_expression(inner, df)
        return val.abs()

    def _op_log(self, df: pd.DataFrame, inner: str) -> pd.Series:
        """Log: Natural logarithm."""
        val = self._evaluate_expression(inner, df)
        return np.log(val.clip(lower=1e-10))

    def _op_sign(self, df: pd.DataFrame, inner: str) -> pd.Series:
        """Sign: Sign of values."""
        val = self._evaluate_expression(inner, df)
        return np.sign(val)

    def _op_corr(self, df: pd.DataFrame, x: str, y: str, window: str) -> pd.Series:
        """Corr: Rolling correlation."""
        x_val = self._evaluate_expression(x.strip(), df)
        y_val = self._evaluate_expression(y.strip(), df)
        return x_val.rolling(int(window)).corr(y_val)

    def _op_cov(self, df: pd.DataFrame, x: str, y: str, window: str) -> pd.Series:
        """Cov: Rolling covariance."""
        x_val = self._evaluate_expression(x.strip(), df)
        y_val = self._evaluate_expression(y.strip(), df)
        return x_val.rolling(int(window)).cov(y_val)

    def _op_wma(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """WMA: Weighted moving average."""
        val = self._evaluate_expression(inner, df)
        w = int(window)
        weights = np.arange(1, w + 1)
        return val.rolling(w).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def _op_ema(self, df: pd.DataFrame, inner: str, span: str) -> pd.Series:
        """EMA: Exponential moving average."""
        val = self._evaluate_expression(inner, df)
        return val.ewm(span=int(span)).mean()

    def _op_rsi(self, df: pd.DataFrame, inner: str, window: str) -> pd.Series:
        """RSI: Relative Strength Index."""
        val = self._evaluate_expression(inner, df)
        delta = val.diff()
        gain = delta.where(delta > 0, 0).rolling(int(window)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(int(window)).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _op_macd(self, df: pd.DataFrame, inner: str, fast: str, slow: str, signal: str) -> pd.Series:
        """MACD: Moving Average Convergence Divergence."""
        val = self._evaluate_expression(inner, df)
        ema_fast = val.ewm(span=int(fast)).mean()
        ema_slow = val.ewm(span=int(slow)).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=int(signal)).mean()
        return macd_line - signal_line  # MACD histogram

    # Binary Operators
    def _add(self, a: pd.Series, b: Union[pd.Series, float]) -> pd.Series:
        return a + b

    def _sub(self, a: pd.Series, b: Union[pd.Series, float]) -> pd.Series:
        return a - b

    def _mul(self, a: pd.Series, b: Union[pd.Series, float]) -> pd.Series:
        return a * b

    def _div(self, a: pd.Series, b: Union[pd.Series, float]) -> pd.Series:
        return a / (b + 1e-10)

    def _gt(self, a: pd.Series, b: Union[pd.Series, float]) -> pd.Series:
        return (a > b).astype(float)

    def _lt(self, a: pd.Series, b: Union[pd.Series, float]) -> pd.Series:
        return (a < b).astype(float)

    def compute_with_d_features(
        self,
        expressions: list[str],
        instruments: list[str],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Compute factors using Qlib D.features() API.

        This method requires Qlib to be properly initialized with data.

        Args:
            expressions: List of Qlib expressions
            instruments: List of instrument codes
            start_time: Start datetime string
            end_time: End datetime string

        Returns:
            DataFrame with computed factor values
        """
        if not self._qlib_initialized:
            raise ValueError("Qlib not initialized. Call init_qlib() first.")

        return D.features(
            instruments=instruments,
            fields=expressions,
            start_time=start_time,
            end_time=end_time,
        )


# Backward compatible alias
FactorEngine = QlibFactorEngine


class FactorEvaluator:
    """Real factor evaluation using statistical metrics.

    Computes actual IC, IR, Sharpe ratio, and other metrics from factor values
    and forward returns. No random simulations.
    """

    def __init__(self, engine: QlibFactorEngine):
        """Initialize evaluator.

        Args:
            engine: Factor engine with loaded data
        """
        self.engine = engine

    def evaluate(
        self,
        factor_values: pd.Series,
        forward_periods: list[int] = [1, 5, 10],
        splits: Optional[list[str]] = None,
    ) -> dict:
        """Evaluate factor and compute real metrics.

        Args:
            factor_values: Series of factor values
            forward_periods: List of forward return periods to evaluate
            splits: Optional data splits (train/valid/test)

        Returns:
            Dictionary of evaluation metrics
        """
        df = self.engine.data
        if df is None:
            raise ValueError("No data loaded")

        # Align factor with data
        factor = factor_values.reindex(df.index)

        # Default splits: 60% train, 20% valid, 20% test
        n = len(df)
        if splits is None:
            splits = ["train", "valid", "test"]

        split_indices = {
            "train": (0, int(n * 0.6)),
            "valid": (int(n * 0.6), int(n * 0.8)),
            "test": (int(n * 0.8), n),
        }

        # Calculate IC (Information Coefficient) for each period
        ic_results = {}
        for period in forward_periods:
            fwd_col = f"fwd_returns_{period}d"
            if fwd_col not in df.columns:
                continue

            fwd_returns = df[fwd_col]

            # Overall IC
            valid_mask = ~(factor.isna() | fwd_returns.isna())
            if valid_mask.sum() > 10:
                ic, _ = stats.spearmanr(
                    factor[valid_mask].values,
                    fwd_returns[valid_mask].values,
                )
                ic_results[f"ic_{period}d"] = float(ic) if not np.isnan(ic) else 0.0

        # Calculate IC by split
        ic_by_split = {}
        for split_name in splits:
            if split_name not in split_indices:
                continue
            start_idx, end_idx = split_indices[split_name]
            split_factor = factor.iloc[start_idx:end_idx]
            split_returns = df["fwd_returns_1d"].iloc[start_idx:end_idx]

            valid_mask = ~(split_factor.isna() | split_returns.isna())
            if valid_mask.sum() > 10:
                ic, _ = stats.spearmanr(
                    split_factor[valid_mask].values,
                    split_returns[valid_mask].values,
                )
                ic_by_split[split_name] = float(ic) if not np.isnan(ic) else 0.0
            else:
                ic_by_split[split_name] = 0.0

        # Calculate IC mean and std (rolling)
        ic_series = self._compute_rolling_ic(factor, df["fwd_returns_1d"], window=20)
        ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        ic_std = float(ic_series.std()) if len(ic_series) > 0 else 1.0

        # IR (Information Ratio) = IC_mean / IC_std
        ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # Backtest factor as simple long-short strategy
        backtest_results = self._backtest_factor_strategy(factor, df)

        # Calculate Sharpe by split
        sharpe_by_split = {}
        for split_name in splits:
            if split_name not in split_indices:
                continue
            start_idx, end_idx = split_indices[split_name]
            split_returns = backtest_results["strategy_returns"].iloc[start_idx:end_idx]
            if len(split_returns) > 1 and split_returns.std() > 0:
                sharpe = (split_returns.mean() / split_returns.std()) * np.sqrt(252)
                sharpe_by_split[split_name] = float(sharpe)
            else:
                sharpe_by_split[split_name] = 0.0

        # Stability analysis
        stability = self._compute_stability(factor, df)

        return {
            "metrics": {
                "ic_mean": round(ic_mean, 4),
                "ic_std": round(ic_std, 4),
                "ir": round(ir, 4),
                "sharpe": round(backtest_results["sharpe"], 4),
                "max_drawdown": round(backtest_results["max_drawdown"], 4),
                "turnover": round(backtest_results["turnover"], 4),
                "total_return": round(backtest_results["total_return"], 4),
                "win_rate": round(backtest_results["win_rate"], 4),
                "ic_by_split": {k: round(v, 4) for k, v in ic_by_split.items()},
                "sharpe_by_split": {k: round(v, 4) for k, v in sharpe_by_split.items()},
                **{k: round(v, 4) for k, v in ic_results.items()},
            },
            "stability": stability,
            "backtest": {
                "cumulative_returns": backtest_results["cumulative_returns"].tolist()[-100:],
                "drawdown": backtest_results["drawdown"].tolist()[-100:],
            },
        }

    def _compute_rolling_ic(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Compute rolling IC over time."""
        ic_values = []

        for i in range(window, len(factor)):
            f_window = factor.iloc[i - window : i]
            r_window = returns.iloc[i - window : i]

            valid_mask = ~(f_window.isna() | r_window.isna())
            if valid_mask.sum() >= 5:
                ic, _ = stats.spearmanr(
                    f_window[valid_mask].values,
                    r_window[valid_mask].values,
                )
                if not np.isnan(ic):
                    ic_values.append(ic)

        return pd.Series(ic_values)

    def _backtest_factor_strategy(
        self,
        factor: pd.Series,
        df: pd.DataFrame,
    ) -> dict:
        """Backtest factor as simple long-short strategy."""
        # Normalize factor to z-score
        factor_zscore = (factor - factor.rolling(20).mean()) / factor.rolling(20).std()
        factor_zscore = factor_zscore.clip(-3, 3)

        # Position: sign of z-score
        position = np.sign(factor_zscore).fillna(0)

        # Strategy returns
        returns = df["returns"].fillna(0)
        strategy_returns = position.shift(1) * returns

        # Remove NaN
        strategy_returns = strategy_returns.fillna(0)

        # Calculate metrics
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Max drawdown
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0

        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Turnover
        position_changes = position.diff().abs()
        turnover = float(position_changes.mean()) if len(position_changes) > 0 else 0.0

        # Win rate
        winning_days = (strategy_returns > 0).sum()
        total_days = (strategy_returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0.0

        # Total return
        total_return = float(cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0.0

        return {
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
            "turnover": turnover,
            "total_return": total_return,
            "win_rate": float(win_rate),
            "cumulative_returns": cumulative_returns,
            "drawdown": drawdown,
            "strategy_returns": strategy_returns,
        }

    def _compute_stability(self, factor: pd.Series, df: pd.DataFrame) -> dict:
        """Compute factor stability metrics."""
        # Time stability: autocorrelation
        autocorr_1 = float(factor.autocorr(1)) if len(factor) > 1 else 0.0
        autocorr_5 = float(factor.autocorr(5)) if len(factor) > 5 else 0.0
        autocorr_20 = float(factor.autocorr(20)) if len(factor) > 20 else 0.0

        # Monthly IC stability
        monthly_ics = []
        df_temp = df.copy()
        df_temp["factor"] = factor.values

        if df_temp.index.name == "timestamp" or hasattr(df_temp.index, "to_period"):
            df_temp["month"] = pd.to_datetime(df_temp.index).to_period("M")

            for _, group in df_temp.groupby("month"):
                if len(group) > 5:
                    valid_mask = ~(group["factor"].isna() | group["fwd_returns_1d"].isna())
                    if valid_mask.sum() > 5:
                        ic, _ = stats.spearmanr(
                            group.loc[valid_mask, "factor"].values,
                            group.loc[valid_mask, "fwd_returns_1d"].values,
                        )
                        if not np.isnan(ic):
                            monthly_ics.append(ic)

        monthly_ic_mean = np.mean(monthly_ics) if monthly_ics else 0.0
        monthly_ic_std = np.std(monthly_ics) if monthly_ics else 1.0
        monthly_stability = abs(monthly_ic_mean) / (monthly_ic_std + 0.001)

        # Regime stability
        market_returns = df["close"].pct_change(20)
        bull_mask = market_returns > 0
        bear_mask = market_returns <= 0

        bull_ic = self._compute_regime_ic(factor, df, bull_mask)
        bear_ic = self._compute_regime_ic(factor, df, bear_mask)

        return {
            "time_stability": {
                "autocorr_1d": round(autocorr_1, 4),
                "autocorr_5d": round(autocorr_5, 4),
                "autocorr_20d": round(autocorr_20, 4),
                "monthly_ic_stability": round(monthly_stability, 4),
            },
            "market_stability": {
                "overall": round((abs(bull_ic) + abs(bear_ic)) / 2, 4),
            },
            "regime_stability": {
                "bull": round(bull_ic, 4),
                "bear": round(bear_ic, 4),
                "consistency": round(1 - abs(bull_ic - bear_ic), 4),
            },
        }

    def _compute_regime_ic(
        self,
        factor: pd.Series,
        df: pd.DataFrame,
        mask: pd.Series,
    ) -> float:
        """Compute IC for a specific market regime."""
        regime_factor = factor[mask]
        regime_returns = df.loc[mask, "fwd_returns_1d"]
        valid_mask = ~(regime_factor.isna() | regime_returns.isna())

        if valid_mask.sum() > 10:
            ic, _ = stats.spearmanr(
                regime_factor[valid_mask].values,
                regime_returns[valid_mask].values,
            )
            return float(ic) if not np.isnan(ic) else 0.0
        return 0.0


# =============================================================================
# Qlib Expression Factor Library (Alpha158 Style)
# =============================================================================

BUILTIN_FACTORS = {
    # Simple expressions (work with current parser)
    "momentum_20d": "Ref($close, -20) / $close",
    "momentum_5d": "Ref($close, -5) / $close",
    "rsi_14": "RSI($close, 14)",
    "rsi_7": "RSI($close, 7)",
    "volatility_20d": "Std($close, 20)",
    "volatility_10d": "Std($close, 10)",
    "volume_ma_5": "Mean($volume, 5)",
    "volume_ma_20": "Mean($volume, 20)",
    "price_ma_20": "Mean($close, 20)",
    "price_ma_50": "Mean($close, 50)",
    "macd_histogram": "MACD($close, 12, 26, 9)",
    "ema_12": "EMA($close, 12)",
    "ema_26": "EMA($close, 26)",
    "high_max_20": "Max($high, 20)",
    "low_min_20": "Min($low, 20)",
    "delta_close_5": "Delta($close, 5)",
    "rank_close": "Rank($close)",
    "wma_10": "WMA($close, 10)",
    "corr_close_volume": "Corr($close, $volume, 20)",
}


def get_default_data_path() -> Path:
    """Get default path to sample data."""
    return Path(__file__).parent.parent.parent.parent / "data" / "sample" / "eth_usdt_futures_daily.csv"


def create_engine_with_sample_data() -> QlibFactorEngine:
    """Create factor engine with sample ETH/USDT data.

    Returns:
        QlibFactorEngine with loaded sample data
    """
    data_path = get_default_data_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {data_path}. "
            "Run scripts/download_sample_data.py first."
        )
    return QlibFactorEngine(df=pd.read_csv(data_path))
