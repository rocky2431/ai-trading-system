"""Qlib-based factor computation and evaluation engine.

This module provides factor calculation using Qlib's expression engine and D.features() API.
Fully leverages Qlib's capabilities for expression parsing, computation, and optimization.

Integrates with:
- TimescaleDB for OHLCV data (via DataProvider)
- CryptoDataHandler for crypto-specific fields
- Qlib expression engine for factor computation
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Use Qlib-native statistical functions instead of scipy
from iqfmp.evaluation.qlib_stats import spearman_rank_correlation

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

# Qlib Ops module for native expression evaluation
try:
    from qlib.data.ops import (
        Ref, Mean, Std, Sum, Max, Min, Delta, Rank, Abs, Log, Sign,
        Corr, Cov, WMA, EMA
    )
    QLIB_OPS_AVAILABLE = True
except ImportError:
    QLIB_OPS_AVAILABLE = False

# Crypto extension (P2.2: custom implementation for crypto markets)
try:
    from iqfmp.core.qlib_crypto import CryptoDataHandler
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    CryptoDataHandler = None

# Import Qlib initialization
from iqfmp.core.qlib_init import init_qlib, ensure_qlib_initialized, is_qlib_initialized
from iqfmp.core.qlib_crypto import QlibExpressionEngine, QlibUnavailableError

# Import data provider for DB integration
from iqfmp.core.data_provider import load_ohlcv_sync, DataProvider

# C4: Import UnifiedMarketDataProvider for derivative fields
try:
    from iqfmp.data.provider import (
        UnifiedMarketDataProvider,
        DataLoadConfig,
        DataLoadResult,
        DerivativeType,
    )
    UNIFIED_PROVIDER_AVAILABLE = True
except ImportError:
    UNIFIED_PROVIDER_AVAILABLE = False
    UnifiedMarketDataProvider = None
    DataLoadConfig = None
    DataLoadResult = None
    DerivativeType = None

# Import Alpha158/360 factor libraries
try:
    from iqfmp.evaluation.alpha158 import ALPHA158_FACTORS
    ALPHA158_AVAILABLE = True
except ImportError:
    ALPHA158_FACTORS = {}
    ALPHA158_AVAILABLE = False

try:
    from iqfmp.evaluation.alpha360 import ALPHA360_FACTORS
    ALPHA360_AVAILABLE = True
except ImportError:
    ALPHA360_FACTORS = {}
    ALPHA360_AVAILABLE = False

logger = logging.getLogger(__name__)


class QlibFactorEngine:
    """Qlib-based factor computation engine.

    Uses Qlib's D.features() API and expression engine for factor calculation.
    Supports both Qlib expression syntax and CryptoDataHandler for crypto data.

    Data loading priority:
    1. Pre-loaded DataFrame (if provided)
    2. TimescaleDB via DataProvider (async)
    3. CSV files (fallback)
    """

    def __init__(
        self,
        provider_uri: Optional[str] = None,
        instruments: Optional[list[str]] = None,
        data_handler: Optional[DataHandlerLP] = None,
        df: Optional[pd.DataFrame] = None,
        use_crypto_handler: bool = True,
        symbol: str = "ETHUSDT",
        timeframe: str = "1d",
        require_qlib: bool = True,
        allow_python_factors: bool = False,
    ):
        """Initialize Qlib factor engine.

        Args:
            provider_uri: Qlib data provider URI (e.g., "~/.qlib/qlib_data/crypto")
            instruments: List of instruments to load (e.g., ["BTCUSDT", "ETHUSDT"])
            data_handler: Pre-configured Qlib DataHandler
            df: Pre-loaded DataFrame (will be converted to Qlib format)
            use_crypto_handler: Use CryptoDataHandler for crypto-specific fields
            symbol: Default symbol to load when df is not provided
            timeframe: Default timeframe to load when df is not provided
            require_qlib: Require Qlib backend; raise if unavailable
            allow_python_factors: Allow custom Python factor functions (non-Qlib)
        """
        self._qlib_initialized = False
        self._provider_uri = provider_uri
        self._instruments = instruments or []
        self._data_handler = data_handler
        self._df: Optional[pd.DataFrame] = None
        self._qlib_data: Optional[pd.DataFrame] = None
        self._use_crypto = use_crypto_handler and CRYPTO_AVAILABLE
        self._crypto_handler: Optional[CryptoDataHandler] = None
        self._default_symbol = symbol
        self._default_timeframe = timeframe
        self._require_qlib = require_qlib
        self._allow_python_factors = allow_python_factors
        self._expression_engine: Optional[QlibExpressionEngine] = None

        # Try to initialize Qlib globally
        if QLIB_AVAILABLE and not is_qlib_initialized():
            try:
                ensure_qlib_initialized()
                self._qlib_initialized = is_qlib_initialized()
                logger.info("Qlib initialized via ensure_qlib_initialized()")
            except Exception as e:
                logger.debug(f"Qlib initialization skipped: {e}")
        else:
            self._qlib_initialized = is_qlib_initialized()

        if not self._qlib_initialized and self._require_qlib:
            raise QlibUnavailableError(
                "Qlib initialization failed but Qlib is required for factor computation."
            )

        # Expression engine (strictly Qlib-backed)
        self._expression_engine = QlibExpressionEngine(require_qlib=self._require_qlib)

        # Initialize with DataFrame if provided
        if df is not None:
            self._df = df.copy()
            self._prepare_data()
            # Also initialize CryptoDataHandler if available
            if self._use_crypto:
                self._init_crypto_handler()
        else:
            # Attempt to load data from TimescaleDB (fallback to CSV)
            try:
                self.load_data_sync(symbol=self._default_symbol, timeframe=self._default_timeframe)
            except Exception as e:
                logger.warning(f"FactorEngine default data load failed: {e}")

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

    def _init_crypto_handler(self) -> None:
        """Initialize CryptoDataHandler with current data."""
        if not self._use_crypto or self._df is None:
            return

        try:
            self._crypto_handler = CryptoDataHandler(
                instruments=self._instruments,
                timeframe="1d",
            )
            self._crypto_handler.load_data(df=self._df)
            self._crypto_handler.add_technical_indicators()
            logger.info("CryptoDataHandler initialized with technical indicators")
        except Exception as e:
            logger.warning(f"Failed to initialize CryptoDataHandler: {e}")
            self._crypto_handler = None

    def load_data(self, data: Union[Path, pd.DataFrame]) -> None:
        """Load OHLCV data from CSV file or DataFrame.

        Args:
            data: Path to CSV file or DataFrame with OHLCV data
        """
        if isinstance(data, pd.DataFrame):
            logger.info(f"Loading data from DataFrame ({len(data)} rows)")
            self._df = data.copy()
        else:
            logger.info(f"Loading data from {data}")
            self._df = pd.read_csv(data)
        self._prepare_data()
        if self._use_crypto:
            self._init_crypto_handler()
        logger.info(f"Loaded {len(self._df)} rows")

    def load_data_sync(
        self,
        symbol: str = "ETHUSDT",
        timeframe: str = "1d",
    ) -> None:
        """Load data synchronously from CSV fallback.

        Args:
            symbol: Trading pair
            timeframe: Data timeframe
        """
        try:
            self._df = load_ohlcv_sync(symbol=symbol, timeframe=timeframe)
            self._prepare_data()
            if self._use_crypto:
                self._init_crypto_handler()
            logger.info(f"Loaded {len(self._df)} rows for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _prepare_data(self) -> None:
        """Prepare data for Qlib factor computation.

        Handles both OHLCV fields and derivative fields (C4).
        """
        if self._df is None:
            return

        # Ensure timestamp is datetime
        if "timestamp" in self._df.columns:
            self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])
            self._df = self._df.sort_values("timestamp").reset_index(drop=True)
            self._df.set_index("timestamp", inplace=True)

        # Standardize column names to Qlib format
        # C4: Extended mapping for OHLCV + derivative fields
        column_mapping = {
            # Standard OHLCV
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            # C4: Derivative fields
            "funding_rate": "$funding_rate",
            "open_interest": "$open_interest",
            "open_interest_change": "$open_interest_change",
            "long_short_ratio": "$long_short_ratio",
            "liquidation_long": "$liquidation_long",
            "liquidation_short": "$liquidation_short",
            "liquidation_total": "$liquidation_total",
            # C5: Derived features
            "funding_ma_8h": "$funding_ma_8h",
            "funding_ma_24h": "$funding_ma_24h",
            "funding_momentum": "$funding_momentum",
            "funding_zscore": "$funding_zscore",
            "funding_extreme": "$funding_extreme",
            "funding_annualized": "$funding_annualized",
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

        # Log available derivative fields for debugging
        derivative_fields = [
            col for col in self._qlib_data.columns
            if col.startswith("$") and col not in ["$open", "$high", "$low", "$close", "$volume"]
        ]
        if derivative_fields:
            logger.info(f"Derivative fields available: {derivative_fields}")

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get loaded data."""
        return self._df

    def get_qlib_status(self) -> dict:
        """Get Qlib integration status for debugging.

        Returns:
            Dictionary with Qlib integration information
        """
        # C4: Include derivative field status
        derivative_fields = []
        if self._qlib_data is not None:
            derivative_fields = [
                col for col in self._qlib_data.columns
                if col.startswith("$") and col not in ["$open", "$high", "$low", "$close", "$volume"]
            ]

        return {
            "qlib_available": QLIB_AVAILABLE,
            "qlib_ops_available": QLIB_OPS_AVAILABLE,
            "qlib_initialized": self._qlib_initialized,
            "crypto_handler_available": CRYPTO_AVAILABLE,
            "crypto_handler_initialized": self._crypto_handler is not None,
            "data_loaded": self._df is not None,
            "data_rows": len(self._df) if self._df is not None else 0,
            "qlib_data_prepared": self._qlib_data is not None,
            "available_indicators": list(self._get_crypto_precomputed_fields()) if CRYPTO_AVAILABLE else [],
            # C4: Derivative fields status
            "unified_provider_available": UNIFIED_PROVIDER_AVAILABLE,
            "derivative_fields": derivative_fields,
            "derivative_fields_count": len(derivative_fields),
        }

    async def load_unified_data_async(
        self,
        session,  # AsyncSession from SQLAlchemy
        symbol: str = "BTC/USDT:USDT",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1h",
        exchange: str = "binance",
        include_derivatives: bool = True,
    ) -> "DataLoadResult":
        """Load unified market data with derivatives asynchronously.

        C4: New method for loading OHLCV + derivative data via UnifiedMarketDataProvider.

        Args:
            session: SQLAlchemy AsyncSession
            symbol: Trading pair (e.g., "BTC/USDT:USDT")
            start_date: Start datetime (default: 180 days ago)
            end_date: End datetime (default: now)
            timeframe: OHLCV timeframe
            exchange: Exchange identifier
            include_derivatives: Whether to include derivative data

        Returns:
            DataLoadResult with merged DataFrame and metadata

        Raises:
            ImportError: If UnifiedMarketDataProvider not available
        """
        if not UNIFIED_PROVIDER_AVAILABLE:
            raise ImportError(
                "UnifiedMarketDataProvider not available. "
                "Install iqfmp.data.provider module."
            )

        from datetime import timedelta

        # Default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)

        # Create provider and load data
        provider = UnifiedMarketDataProvider(session, exchange=exchange)
        config = DataLoadConfig(
            include_derivatives=include_derivatives,
            calculate_features=True,
        )

        result = await provider.load_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            config=config,
        )

        # Update internal state with loaded data
        if not result.df.empty:
            self._df = result.df.copy()
            self._prepare_data()
            if self._use_crypto:
                self._init_crypto_handler()

            logger.info(
                f"Loaded unified data: {len(result.df)} rows, "
                f"derivatives: {result.derivative_columns}, "
                f"derived: {result.derived_columns}"
            )

        return result

    def get_available_derivative_fields(self) -> list[str]:
        """Get list of available derivative fields in current data.

        C4: Helper method to check which derivative fields are loaded.

        Returns:
            List of available derivative field names (with $ prefix)
        """
        if self._qlib_data is None:
            return []

        all_derivative_fields = [
            "$funding_rate",
            "$open_interest",
            "$open_interest_change",
            "$long_short_ratio",
            "$liquidation_long",
            "$liquidation_short",
            "$liquidation_total",
            "$funding_ma_8h",
            "$funding_ma_24h",
            "$funding_momentum",
            "$funding_zscore",
            "$funding_extreme",
            "$funding_annualized",
        ]

        return [f for f in all_derivative_fields if f in self._qlib_data.columns]

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
        Uses Qlib's native Ops module when available, falls back to local implementation.
        Also supports Python function definitions (def ...).
        """
        df = self._qlib_data

        # Enforce Qlib-only unless explicitly allowed
        stripped = expression.strip()
        is_python_code = (
            stripped.startswith("def ")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
            or "\ndef " in expression
        )
        if is_python_code and not self._allow_python_factors:
            raise ValueError(
                "Python function factors are disabled; provide Qlib expression instead"
            )
        if is_python_code:
            return self._execute_python_factor(expression, df)

        # Use pre-computed crypto indicators when available
        if self._crypto_handler and expression in self._get_crypto_precomputed_fields():
            return self._get_crypto_indicator(expression)

        # Strictly go through Qlib expression engine
        if not self._expression_engine:
            raise QlibUnavailableError("Qlib expression engine not initialized")

        try:
            return self._expression_engine.compute_expression(
                expression=expression,
                df=df,
                result_name="factor",  # Use default name since _compute_qlib_expression doesn't have factor_name param
            )
        except QlibUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Qlib expression computation failed: {e}")
            raise

    def _execute_python_factor(self, code: str, df: pd.DataFrame) -> pd.Series:
        """Execute Python function code to compute factor.

        Args:
            code: Python code containing a function definition
            df: DataFrame with price data

        Returns:
            Computed factor series

        Raises:
            ValueError: If code execution fails
        """
        import re

        # Preprocess code: remove import statements (np and pd are already available)
        # This allows LLM-generated code with imports to work in the sandbox
        code_lines = code.split('\n')
        filtered_lines = []
        for line in code_lines:
            stripped = line.strip()
            # Skip import statements - np/pd/numpy/pandas are already available
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            filtered_lines.append(line)
        code = '\n'.join(filtered_lines)

        # Extract function name
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if not match:
            raise ValueError("No function definition found in code")

        func_name = match.group(1)

        # Prepare safe execution environment
        safe_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "float": float,
                "int": int,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "zip": zip,
                "enumerate": enumerate,
                "sorted": sorted,
                "reversed": reversed,
                "True": True,
                "False": False,
                "None": None,
                "print": print,  # Allow print for debugging
            },
            "pd": pd,
            "np": np,
            "numpy": np,  # Also expose as numpy for compatibility
            "pandas": pd,  # Also expose as pandas for compatibility
        }

        local_vars: dict = {}

        try:
            # Execute the function definition
            exec(code, safe_globals, local_vars)

            # Get the function
            if func_name not in local_vars:
                raise ValueError(f"Function {func_name} not found after execution")

            factor_func = local_vars[func_name]

            # Prepare DataFrame with lowercase columns for compatibility
            df_copy = df.copy()
            # Map $column to column for compatibility
            col_mapping = {}
            for col in df_copy.columns:
                if col.startswith("$"):
                    col_mapping[col] = col[1:]  # Remove $ prefix
            if col_mapping:
                df_copy = df_copy.rename(columns=col_mapping)

            # Execute the factor function
            result = factor_func(df_copy)

            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, pd.DataFrame):
                return result.iloc[:, 0]
            else:
                return pd.Series(result, index=df.index)

        except Exception as e:
            raise ValueError(f"Python factor execution failed: {e}")

    def _get_crypto_precomputed_fields(self) -> set[str]:
        """Get set of pre-computed crypto indicator field names."""
        return {
            "RSI($close, 14)", "RSI_14", "$rsi_14",
            "MACD($close, 12, 26, 9)", "MACD_HIST", "$macd_hist",
            "$macd", "$macd_signal",
            "$bb_upper", "$bb_lower", "$atr_14",
            "$vwap", "$typical", "$dollar_volume",
        }

    def _get_crypto_indicator(self, expression: str) -> pd.Series:
        """Get pre-computed indicator from CryptoDataHandler."""
        if not self._crypto_handler or self._crypto_handler.data is None:
            raise ValueError("CryptoDataHandler not initialized")

        df = self._crypto_handler.data

        # Map common expressions to CryptoDataHandler fields
        field_map = {
            "RSI($close, 14)": "$rsi_14",
            "RSI_14": "$rsi_14",
            "$rsi_14": "$rsi_14",
            "MACD($close, 12, 26, 9)": "$macd_hist",
            "MACD_HIST": "$macd_hist",
            "$macd_hist": "$macd_hist",
            "$macd": "$macd",
            "$macd_signal": "$macd_signal",
            "$bb_upper": "$bb_upper",
            "$bb_lower": "$bb_lower",
            "$atr_14": "$atr_14",
            "$vwap": "$vwap",
            "$typical": "$typical",
            "$dollar_volume": "$dollar_volume",
        }

        field = field_map.get(expression, expression)
        if field in df.columns:
            return df[field]

        raise ValueError(f"Pre-computed indicator not found: {expression}")

    def _compute_with_qlib_ops(self, expression: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """Try to compute expression using Qlib's native Ops module.

        Note: This requires Qlib to be properly initialized and only works
        for expressions that can be parsed by Qlib's expression parser.
        """
        # For now, we use the local implementation as it's more reliable
        # for arbitrary DataFrame data (not stored in Qlib format).
        # Qlib's native Ops require specific data provider setup.
        #
        # This method is a placeholder for future Qlib integration when
        # data is stored in Qlib format.
        return None

    # =========================================================================
    # REMOVED: Local Pandas Implementation (P0-2 Fix)
    # =========================================================================
    # All local Pandas operator implementations have been removed.
    # Factor expressions MUST use Qlib's C++ engine via:
    # 1. QlibExpressionEngine (from qlib_crypto.py)
    # 2. D.features() API (from qlib.data)
    #
    # This ensures production-grade performance and correctness.
    # =========================================================================

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
                ic, _ = spearman_rank_correlation(
                    factor[valid_mask],
                    fwd_returns[valid_mask],
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
                ic, _ = spearman_rank_correlation(
                    split_factor[valid_mask],
                    split_returns[valid_mask],
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
                ic, _ = spearman_rank_correlation(
                    f_window[valid_mask],
                    r_window[valid_mask],
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
                        ic, _ = spearman_rank_correlation(
                            group.loc[valid_mask, "factor"],
                            group.loc[valid_mask, "fwd_returns_1d"],
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
            ic, _ = spearman_rank_correlation(
                regime_factor[valid_mask],
                regime_returns[valid_mask],
            )
            return float(ic) if not np.isnan(ic) else 0.0
        return 0.0

    # =========================================================================
    # Alpha158/360 Factor Library Integration
    # =========================================================================

    def get_alpha158_factor(self, factor_name: str) -> pd.Series:
        """Compute a single Alpha158 factor.

        Args:
            factor_name: Name of the Alpha158 factor (e.g., "KMID", "ROC5", "RSI14")

        Returns:
            Series of factor values

        Raises:
            ValueError: If factor not found or data not loaded
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not ALPHA158_AVAILABLE:
            raise ValueError("Alpha158 factors not available. Install dependencies.")

        if factor_name not in ALPHA158_FACTORS:
            available = ", ".join(sorted(ALPHA158_FACTORS.keys())[:10])
            raise ValueError(
                f"Factor '{factor_name}' not found. Available: {available}... "
                f"(Total: {len(ALPHA158_FACTORS)} factors)"
            )

        factor_func = ALPHA158_FACTORS[factor_name]
        return factor_func(self._df)

    def compute_alpha158_factors(
        self,
        factor_names: Optional[list[str]] = None,
        parallel: bool = False,
    ) -> pd.DataFrame:
        """Compute multiple Alpha158 factors in batch.

        Args:
            factor_names: List of factor names, or None for all 158 factors
            parallel: Use parallel computation (for large datasets)

        Returns:
            DataFrame with all computed factors

        Raises:
            ValueError: If data not loaded
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not ALPHA158_AVAILABLE:
            raise ValueError("Alpha158 factors not available. Install dependencies.")

        # Use all factors if none specified
        names = factor_names or list(ALPHA158_FACTORS.keys())

        results = {}
        failed = []

        for name in names:
            if name not in ALPHA158_FACTORS:
                logger.warning(f"Factor '{name}' not found, skipping")
                continue

            try:
                factor_func = ALPHA158_FACTORS[name]
                results[name] = factor_func(self._df)
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                failed.append(name)

        if failed:
            logger.warning(f"Failed factors: {', '.join(failed)}")

        logger.info(f"Computed {len(results)}/{len(names)} Alpha158 factors")
        return pd.DataFrame(results, index=self._df.index)

    def get_alpha360_factor(self, factor_name: str) -> pd.Series:
        """Compute a single Alpha360 factor.

        Args:
            factor_name: Name of the Alpha360 factor

        Returns:
            Series of factor values
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not ALPHA360_AVAILABLE:
            raise ValueError("Alpha360 factors not available. Install dependencies.")

        if factor_name not in ALPHA360_FACTORS:
            raise ValueError(f"Factor '{factor_name}' not found in Alpha360")

        factor_func = ALPHA360_FACTORS[factor_name]
        return factor_func(self._df)

    def compute_alpha360_factors(
        self,
        factor_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute multiple Alpha360 factors in batch.

        Args:
            factor_names: List of factor names, or None for all factors

        Returns:
            DataFrame with all computed factors
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not ALPHA360_AVAILABLE:
            raise ValueError("Alpha360 factors not available. Install dependencies.")

        names = factor_names or list(ALPHA360_FACTORS.keys())

        results = {}
        for name in names:
            if name in ALPHA360_FACTORS:
                try:
                    results[name] = ALPHA360_FACTORS[name](self._df)
                except Exception as e:
                    logger.warning(f"Failed to compute {name}: {e}")

        logger.info(f"Computed {len(results)}/{len(names)} Alpha360 factors")
        return pd.DataFrame(results, index=self._df.index)

    def list_alpha_factors(self) -> dict[str, list[str]]:
        """List all available Alpha factors.

        Returns:
            Dictionary with 'alpha158' and 'alpha360' factor lists
        """
        return {
            "alpha158": sorted(ALPHA158_FACTORS.keys()) if ALPHA158_AVAILABLE else [],
            "alpha360": sorted(ALPHA360_FACTORS.keys()) if ALPHA360_AVAILABLE else [],
            "alpha158_count": len(ALPHA158_FACTORS) if ALPHA158_AVAILABLE else 0,
            "alpha360_count": len(ALPHA360_FACTORS) if ALPHA360_AVAILABLE else 0,
        }

    def evaluate_alpha158_factors(
        self,
        factor_names: Optional[list[str]] = None,
        top_n: int = 20,
    ) -> list[dict]:
        """Evaluate Alpha158 factors and return top performers.

        Args:
            factor_names: Factors to evaluate, or None for all
            top_n: Number of top factors to return

        Returns:
            List of evaluation results sorted by IC
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Compute all factors
        factors_df = self.compute_alpha158_factors(factor_names)

        results = []
        for col in factors_df.columns:
            factor = factors_df[col]
            valid = ~(factor.isna() | self._df["fwd_returns_1d"].isna())

            if valid.sum() > 30:
                ic, p_value = spearman_rank_correlation(
                    factor[valid],
                    self._df.loc[valid, "fwd_returns_1d"],
                )
                if not np.isnan(ic):
                    results.append({
                        "factor_name": col,
                        "ic": float(ic),
                        "ic_abs": abs(float(ic)),
                        "p_value": float(p_value),
                        "valid_count": int(valid.sum()),
                    })

        # Sort by absolute IC
        results.sort(key=lambda x: x["ic_abs"], reverse=True)

        logger.info(f"Evaluated {len(results)} Alpha158 factors, top IC: {results[0]['ic']:.4f}" if results else "No valid factors")

        return results[:top_n]


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
