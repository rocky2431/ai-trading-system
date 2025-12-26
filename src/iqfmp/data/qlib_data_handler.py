"""Qlib Alpha DataHandler Integration for IQFMP.

This module provides integration with Qlib's official DataHandler infrastructure,
implementing spec requirements for:
- Alpha158Handler integration
- Alpha360Handler integration
- Custom crypto factor handlers
- Qlib dataset creation and caching

Migration from custom data loaders to Qlib DataHandler.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Qlib imports with fallback
QLIB_AVAILABLE = False
_qlib_initialized = False

try:
    import qlib
    from qlib.config import C
    QLIB_IMPORT_SUCCESS = True
except ImportError:
    QLIB_IMPORT_SUCCESS = False
    logger.warning("Qlib not available. Install with: pip install pyqlib")


def _ensure_qlib_initialized() -> bool:
    """Ensure Qlib is properly initialized."""
    global QLIB_AVAILABLE, _qlib_initialized

    if _qlib_initialized:
        return QLIB_AVAILABLE

    if not QLIB_IMPORT_SUCCESS:
        _qlib_initialized = True
        return False

    try:
        if hasattr(C, "provider_uri") and C.provider_uri is not None:
            QLIB_AVAILABLE = True
            _qlib_initialized = True
            return True

        data_dir = os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data")
        qlib.init(provider_uri=os.path.expanduser(data_dir))
        QLIB_AVAILABLE = True
        _qlib_initialized = True
        return True

    except Exception as e:
        logger.warning(f"Qlib initialization failed: {e}")
        QLIB_AVAILABLE = False
        _qlib_initialized = True
        return False


# =============================================================================
# Qlib DataHandler Configuration
# =============================================================================

@dataclass
class DataHandlerConfig:
    """Configuration for Qlib DataHandler."""

    # Time range
    start_time: str = "2015-01-01"
    end_time: str = "2024-12-31"

    # Data settings
    fit_start_time: Optional[str] = None
    fit_end_time: Optional[str] = None
    instruments: str = "csi300"  # or "csi500", "all", custom list

    # Feature settings
    handler_type: str = "Alpha158"  # "Alpha158", "Alpha360", "crypto"
    label_type: str = "Ref($close, -2) / Ref($close, -1) - 1"  # 2-day return
    normalize: bool = True
    fillna: bool = True
    drop_na: bool = True

    # Cache settings
    cache_dir: str = "~/.qlib/cache"
    use_cache: bool = True


# =============================================================================
# Qlib DataHandler Wrapper
# =============================================================================

class QlibDataHandler:
    """Wrapper for Qlib's official DataHandler.

    Provides unified interface for:
    - Alpha158Handler (158 technical indicators)
    - Alpha360Handler (360 comprehensive features)
    - Custom crypto handlers
    """

    def __init__(self, config: Optional[DataHandlerConfig] = None) -> None:
        """Initialize data handler.

        Args:
            config: Data handler configuration
        """
        self.config = config or DataHandlerConfig()
        self._handler = None
        self._dataset = None

    def setup(self) -> bool:
        """Setup the data handler.

        Returns:
            True if setup successful, False otherwise
        """
        if not _ensure_qlib_initialized():
            logger.warning("Qlib not available, using fallback handler")
            return False

        try:
            self._handler = self._create_handler()
            return True
        except Exception as e:
            logger.error(f"Failed to create handler: {e}")
            return False

    def _create_handler(self) -> Any:
        """Create appropriate handler based on config."""
        from qlib.data.dataset.handler import DataHandlerLP

        handler_type = self.config.handler_type.lower()

        if handler_type == "alpha158":
            return self._create_alpha158_handler()
        elif handler_type == "alpha360":
            return self._create_alpha360_handler()
        elif handler_type == "crypto":
            return self._create_crypto_handler()
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def _create_alpha158_handler(self) -> Any:
        """Create Alpha158 handler."""
        from qlib.contrib.data.handler import Alpha158

        return Alpha158(
            instruments=self.config.instruments,
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            fit_start_time=self.config.fit_start_time,
            fit_end_time=self.config.fit_end_time,
            drop_raw=True,
        )

    def _create_alpha360_handler(self) -> Any:
        """Create Alpha360 handler."""
        from qlib.contrib.data.handler import Alpha360

        return Alpha360(
            instruments=self.config.instruments,
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            fit_start_time=self.config.fit_start_time,
            fit_end_time=self.config.fit_end_time,
            drop_raw=True,
        )

    def _create_crypto_handler(self) -> Any:
        """Create crypto-specific handler.

        This is a custom handler for crypto markets with:
        - 24/7 trading features
        - Funding rate features
        - On-chain metrics
        """
        from qlib.data.dataset.handler import DataHandlerLP

        # Define crypto-specific features
        crypto_fields = [
            # Price features
            ("$close", "close"),
            ("$open", "open"),
            ("$high", "high"),
            ("$low", "low"),
            ("$volume", "volume"),

            # Technical features (crypto-specific)
            ("Ref($close, -1) / $close - 1", "return_1d"),
            ("Ref($close, -7) / $close - 1", "return_7d"),
            ("Mean($volume, 24)", "volume_ma24"),
            ("Std($close, 24)", "volatility_24h"),

            # Momentum
            ("$close / Ref($close, -1) - 1", "momentum_1d"),
            ("$close / Ref($close, -7) - 1", "momentum_7d"),
            ("$close / Mean($close, 20) - 1", "ma_deviation_20"),

            # Volume features
            ("$volume / Mean($volume, 20)", "volume_ratio_20"),
            ("Rank($volume)", "volume_rank"),

            # Volatility features
            ("Std($close, 7) / Mean($close, 7)", "cv_7d"),
            ("Std($close, 30) / Mean($close, 30)", "cv_30d"),
        ]

        fields = [
            (expr, name) for expr, name in crypto_fields
        ]

        return DataHandlerLP(
            instruments=self.config.instruments,
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            data_loader={
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": fields,
                        "label": [(self.config.label_type, "label")],
                    },
                },
            },
        )

    def get_data(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get feature data and labels.

        Args:
            start_time: Optional start time override
            end_time: Optional end time override

        Returns:
            Tuple of (features DataFrame, labels DataFrame)
        """
        if self._handler is None:
            if not self.setup():
                return self._get_fallback_data()

        try:
            # Get handler data
            df = self._handler.fetch(
                data_key="learn",
                start_time=start_time or self.config.start_time,
                end_time=end_time or self.config.end_time,
            )

            # Split features and labels
            if isinstance(df, pd.DataFrame):
                if "label" in df.columns:
                    features = df.drop("label", axis=1)
                    labels = df[["label"]]
                else:
                    features = df
                    labels = pd.DataFrame(index=df.index)
            else:
                features = df
                labels = pd.DataFrame()

            return features, labels

        except Exception as e:
            logger.error(f"Failed to get data: {e}")
            return self._get_fallback_data()

    def _get_fallback_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return empty DataFrames when Qlib unavailable."""
        logger.warning("Returning empty fallback data")
        return pd.DataFrame(), pd.DataFrame()

    def create_dataset(
        self,
        train_start: str,
        train_end: str,
        valid_start: str,
        valid_end: str,
        test_start: str,
        test_end: str,
    ) -> Any:
        """Create Qlib Dataset with train/valid/test splits.

        Args:
            train_start: Training start date
            train_end: Training end date
            valid_start: Validation start date
            valid_end: Validation end date
            test_start: Test start date
            test_end: Test end date

        Returns:
            Qlib DatasetH instance
        """
        if not _ensure_qlib_initialized():
            raise RuntimeError("Qlib not available")

        from qlib.data.dataset import DatasetH

        if self._handler is None:
            self.setup()

        return DatasetH(
            handler=self._handler,
            segments={
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        )


# =============================================================================
# Qlib Feature Extractor
# =============================================================================

class QlibFeatureExtractor:
    """Extract features using Qlib expressions.

    Provides high-level interface for feature extraction
    without full handler setup.
    """

    def __init__(self) -> None:
        """Initialize feature extractor."""
        self._initialized = False

    def extract(
        self,
        expressions: List[str],
        instruments: Union[str, List[str]],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Extract features using Qlib expressions.

        Args:
            expressions: List of Qlib expression strings
            instruments: Instrument or list of instruments
            start_time: Start date
            end_time: End date

        Returns:
            DataFrame with extracted features
        """
        if not _ensure_qlib_initialized():
            return pd.DataFrame()

        from qlib.data import D

        try:
            fields = [(expr, f"feature_{i}") for i, expr in enumerate(expressions)]

            data = D.features(
                instruments=instruments,
                fields=fields,
                start_time=start_time,
                end_time=end_time,
            )

            return data

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return pd.DataFrame()


# =============================================================================
# Crypto Data Handler
# =============================================================================

class CryptoDataHandler:
    """Crypto-specific data handler.

    Extends Qlib DataHandler with crypto market features:
    - Funding rate integration
    - Perpetual/spot spread
    - On-chain metrics (when available)
    - 24/7 market considerations
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1d",
    ) -> None:
        """Initialize crypto data handler.

        Args:
            symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            timeframe: Data timeframe
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.timeframe = timeframe
        self._ohlcv_data: Dict[str, pd.DataFrame] = {}
        self._funding_data: Dict[str, pd.DataFrame] = {}

    def load_from_db(
        self,
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Load data from TimescaleDB.

        Args:
            start_time: Start date
            end_time: End date

        Returns:
            Combined DataFrame with all features
        """
        try:
            from iqfmp.data.ohlcv_loader import load_ohlcv_data

            all_data = []

            for symbol in self.symbols:
                df = load_ohlcv_data(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    start_time=start_time,
                    end_time=end_time,
                )

                if not df.empty:
                    df = self._add_crypto_features(df)
                    df["symbol"] = symbol
                    all_data.append(df)

            if all_data:
                return pd.concat(all_data, axis=0)
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to load crypto data: {e}")
            return pd.DataFrame()

    def _add_crypto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crypto-specific features to OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with additional features
        """
        # Returns
        df["return_1d"] = df["close"].pct_change()
        df["return_7d"] = df["close"].pct_change(7)

        # Volatility
        df["volatility_7d"] = df["return_1d"].rolling(7).std()
        df["volatility_30d"] = df["return_1d"].rolling(30).std()

        # Volume features
        df["volume_ma7"] = df["volume"].rolling(7).mean()
        df["volume_ma30"] = df["volume"].rolling(30).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma30"]

        # Price features
        df["ma7"] = df["close"].rolling(7).mean()
        df["ma30"] = df["close"].rolling(30).mean()
        df["ma_cross"] = (df["ma7"] > df["ma30"]).astype(float)

        # OHLC features
        df["range"] = (df["high"] - df["low"]) / df["close"]
        df["body"] = (df["close"] - df["open"]) / df["open"]

        return df

    def to_qlib_format(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert DataFrame to Qlib format.

        Args:
            df: DataFrame with features

        Returns:
            Qlib-compatible DataFrame with MultiIndex
        """
        if df.empty:
            return df

        # Ensure datetime index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Create MultiIndex (datetime, instrument)
        if "symbol" in df.columns:
            df = df.reset_index()
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["instrument"] = df["symbol"]
            df = df.set_index(["datetime", "instrument"])
            df = df.drop(["timestamp", "symbol"], axis=1, errors="ignore")

        return df


# =============================================================================
# Factory Functions
# =============================================================================

def create_data_handler(
    handler_type: str = "Alpha158",
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2024-12-31",
    **kwargs,
) -> QlibDataHandler:
    """Create a data handler.

    Args:
        handler_type: Type of handler ("Alpha158", "Alpha360", "crypto")
        instruments: Instrument universe
        start_time: Start date
        end_time: End date
        **kwargs: Additional configuration options

    Returns:
        Configured data handler
    """
    config = DataHandlerConfig(
        handler_type=handler_type,
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        **{k: v for k, v in kwargs.items() if hasattr(DataHandlerConfig, k)},
    )

    handler = QlibDataHandler(config)
    handler.setup()
    return handler


def create_crypto_handler(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1d",
) -> CryptoDataHandler:
    """Create a crypto data handler.

    Args:
        symbols: List of trading pairs
        timeframe: Data timeframe

    Returns:
        Configured crypto handler
    """
    return CryptoDataHandler(symbols=symbols, timeframe=timeframe)


def get_alpha158_features(
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2024-12-31",
) -> pd.DataFrame:
    """Get Alpha158 features for instruments.

    Args:
        instruments: Instrument universe
        start_time: Start date
        end_time: End date

    Returns:
        DataFrame with Alpha158 features
    """
    handler = create_data_handler(
        handler_type="Alpha158",
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
    )

    features, _ = handler.get_data()
    return features


def get_alpha360_features(
    instruments: str = "csi300",
    start_time: str = "2020-01-01",
    end_time: str = "2024-12-31",
) -> pd.DataFrame:
    """Get Alpha360 features for instruments.

    Args:
        instruments: Instrument universe
        start_time: Start date
        end_time: End date

    Returns:
        DataFrame with Alpha360 features
    """
    handler = create_data_handler(
        handler_type="Alpha360",
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
    )

    features, _ = handler.get_data()
    return features
