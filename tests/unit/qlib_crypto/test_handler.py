"""Tests for CryptoDataHandler.

Six-dimensional test coverage:
1. Functional: Core data handling functionality
2. Boundary: Edge cases and limits
3. Exception: Error handling
4. Performance: Data processing efficiency
5. Security: Data validation and sanitization
6. Compatibility: Different data formats and exchanges
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any

from iqfmp.qlib_crypto.data.handler import (
    CryptoDataHandler,
    CryptoDataConfig,
    CryptoField,
    Exchange,
    TimeFrame,
)
from iqfmp.qlib_crypto.data.validator import (
    DataValidator,
    ValidationResult,
    ValidationError,
)


class TestCryptoFieldModel:
    """Test CryptoField enumeration."""

    def test_standard_ohlcv_fields(self) -> None:
        """Test standard OHLCV fields exist."""
        assert CryptoField.OPEN is not None
        assert CryptoField.HIGH is not None
        assert CryptoField.LOW is not None
        assert CryptoField.CLOSE is not None
        assert CryptoField.VOLUME is not None

    def test_crypto_specific_fields(self) -> None:
        """Test crypto-specific fields exist."""
        assert CryptoField.FUNDING_RATE is not None
        assert CryptoField.OPEN_INTEREST is not None
        assert CryptoField.BASIS is not None
        assert CryptoField.MARK_PRICE is not None
        assert CryptoField.INDEX_PRICE is not None

    def test_field_string_values(self) -> None:
        """Test field string representations."""
        assert CryptoField.FUNDING_RATE.value == "funding_rate"
        assert CryptoField.OPEN_INTEREST.value == "open_interest"
        assert CryptoField.BASIS.value == "basis"


class TestExchangeModel:
    """Test Exchange enumeration."""

    def test_supported_exchanges(self) -> None:
        """Test supported exchanges exist."""
        assert Exchange.BINANCE is not None
        assert Exchange.OKX is not None
        assert Exchange.BYBIT is not None

    def test_exchange_string_values(self) -> None:
        """Test exchange string representations."""
        assert Exchange.BINANCE.value == "binance"
        assert Exchange.OKX.value == "okx"


class TestTimeFrameModel:
    """Test TimeFrame enumeration."""

    def test_supported_timeframes(self) -> None:
        """Test supported timeframes exist."""
        assert TimeFrame.M1 is not None
        assert TimeFrame.M5 is not None
        assert TimeFrame.M15 is not None
        assert TimeFrame.H1 is not None
        assert TimeFrame.H4 is not None
        assert TimeFrame.D1 is not None

    def test_timeframe_minutes(self) -> None:
        """Test timeframe to minutes conversion."""
        assert TimeFrame.M1.to_minutes() == 1
        assert TimeFrame.M5.to_minutes() == 5
        assert TimeFrame.H1.to_minutes() == 60
        assert TimeFrame.H4.to_minutes() == 240
        assert TimeFrame.D1.to_minutes() == 1440


class TestCryptoDataConfigModel:
    """Test CryptoDataConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CryptoDataConfig()
        assert len(config.fields) > 0
        assert CryptoField.CLOSE in config.fields
        assert config.exchange is not None
        assert config.timeframe is not None

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CryptoDataConfig(
            fields=[CryptoField.CLOSE, CryptoField.FUNDING_RATE],
            exchange=Exchange.OKX,
            timeframe=TimeFrame.H4,
            symbols=["BTC-USDT", "ETH-USDT"],
        )
        assert CryptoField.FUNDING_RATE in config.fields
        assert config.exchange == Exchange.OKX
        assert config.timeframe == TimeFrame.H4
        assert "BTC-USDT" in config.symbols


class TestCryptoDataHandlerFunctional:
    """Functional tests for CryptoDataHandler."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        return pd.DataFrame({
            "datetime": dates,
            "symbol": "BTC-USDT",
            "open": np.random.uniform(40000, 45000, 100),
            "high": np.random.uniform(45000, 50000, 100),
            "low": np.random.uniform(35000, 40000, 100),
            "close": np.random.uniform(40000, 45000, 100),
            "volume": np.random.uniform(1000, 10000, 100),
            "funding_rate": np.random.uniform(-0.001, 0.001, 100),
            "open_interest": np.random.uniform(1e9, 2e9, 100),
        })

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        """Create a data handler instance."""
        return CryptoDataHandler()

    # === Basic Operations ===

    def test_handler_creation(self, handler: CryptoDataHandler) -> None:
        """Test creating a handler."""
        assert handler is not None
        assert handler.config is not None

    def test_handler_with_config(self) -> None:
        """Test creating handler with custom config."""
        config = CryptoDataConfig(
            fields=[CryptoField.CLOSE, CryptoField.FUNDING_RATE],
            exchange=Exchange.BINANCE,
        )
        handler = CryptoDataHandler(config=config)
        assert handler.config.exchange == Exchange.BINANCE

    def test_load_data(
        self, handler: CryptoDataHandler, sample_data: pd.DataFrame
    ) -> None:
        """Test loading data into handler."""
        handler.load(sample_data)
        assert handler.data is not None
        assert len(handler.data) == 100

    def test_get_field(
        self, handler: CryptoDataHandler, sample_data: pd.DataFrame
    ) -> None:
        """Test getting specific field data."""
        handler.load(sample_data)
        close_data = handler.get_field(CryptoField.CLOSE)
        assert close_data is not None
        assert len(close_data) == 100

    def test_get_crypto_fields(
        self, handler: CryptoDataHandler, sample_data: pd.DataFrame
    ) -> None:
        """Test getting crypto-specific fields."""
        handler.load(sample_data)
        funding = handler.get_field(CryptoField.FUNDING_RATE)
        oi = handler.get_field(CryptoField.OPEN_INTEREST)
        assert funding is not None
        assert oi is not None

    # === Data Transformation ===

    def test_calculate_returns(
        self, handler: CryptoDataHandler, sample_data: pd.DataFrame
    ) -> None:
        """Test calculating returns."""
        handler.load(sample_data)
        returns = handler.calculate_returns()
        assert returns is not None
        assert len(returns) == 99  # One less due to diff

    def test_calculate_funding_adjusted_returns(
        self, handler: CryptoDataHandler, sample_data: pd.DataFrame
    ) -> None:
        """Test calculating funding-adjusted returns."""
        handler.load(sample_data)
        adj_returns = handler.calculate_funding_adjusted_returns()
        assert adj_returns is not None

    def test_resample_data(
        self, handler: CryptoDataHandler, sample_data: pd.DataFrame
    ) -> None:
        """Test resampling to different timeframe."""
        handler.load(sample_data)
        daily = handler.resample(TimeFrame.D1)
        assert daily is not None
        assert len(daily) < len(sample_data)


class TestCryptoDataHandlerSecurity:
    """Security tests for data validation."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_validate_required_columns(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test validation of required columns."""
        incomplete_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": np.random.uniform(40000, 45000, 10),
            # Missing open, high, low, volume
        })
        with pytest.raises(ValidationError, match="missing.*columns"):
            handler.load(incomplete_data, validate=True)

    def test_validate_data_types(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test validation of data types."""
        bad_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": ["invalid"] * 10,  # Should be numeric
            "high": np.random.uniform(45000, 50000, 10),
            "low": np.random.uniform(35000, 40000, 10),
            "close": np.random.uniform(40000, 45000, 10),
            "volume": np.random.uniform(1000, 10000, 10),
        })
        with pytest.raises(ValidationError, match="numeric"):
            handler.load(bad_data, validate=True)

    def test_validate_price_consistency(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test validation of price consistency (high >= low)."""
        inconsistent_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [100] * 10,
            "high": [90] * 10,   # High < Low is invalid
            "low": [110] * 10,
            "close": [100] * 10,
            "volume": [1000] * 10,
        })
        with pytest.raises(ValidationError, match="high.*low"):
            handler.load(inconsistent_data, validate=True)

    def test_validate_funding_rate_range(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test validation of funding rate range."""
        extreme_funding = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [45000] * 10,
            "low": [35000] * 10,
            "close": [42000] * 10,
            "volume": [1000] * 10,
            "funding_rate": [0.5] * 10,  # 50% is unrealistic
        })
        result = handler.validate(extreme_funding)
        assert not result.is_valid or len(result.warnings) > 0


class TestCryptoDataHandlerBoundary:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_empty_dataframe(self, handler: CryptoDataHandler) -> None:
        """Test handling empty dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            handler.load(empty_df, validate=True)

    def test_single_row(self, handler: CryptoDataHandler) -> None:
        """Test handling single row of data."""
        single_row = pd.DataFrame({
            "datetime": [datetime.now()],
            "symbol": ["BTC-USDT"],
            "open": [40000],
            "high": [41000],
            "low": [39000],
            "close": [40500],
            "volume": [1000],
        })
        handler.load(single_row)
        assert len(handler.data) == 1

    def test_large_dataset(self, handler: CryptoDataHandler) -> None:
        """Test handling large dataset."""
        large_data = pd.DataFrame({
            "datetime": pd.date_range(start="2020-01-01", periods=100000, freq="h"),
            "symbol": "BTC-USDT",
            "open": np.random.uniform(40000, 45000, 100000),
            "high": np.random.uniform(45000, 50000, 100000),
            "low": np.random.uniform(35000, 40000, 100000),
            "close": np.random.uniform(40000, 45000, 100000),
            "volume": np.random.uniform(1000, 10000, 100000),
        })
        handler.load(large_data)
        assert len(handler.data) == 100000

    def test_missing_values(self, handler: CryptoDataHandler) -> None:
        """Test handling missing values."""
        data_with_nan = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000, np.nan, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000],
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        handler.load(data_with_nan, fill_na=True)
        assert not handler.data["open"].isna().any()


class TestCryptoDataHandlerException:
    """Exception handling tests."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_get_field_before_load(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test getting field before loading data."""
        with pytest.raises(RuntimeError, match="not loaded"):
            handler.get_field(CryptoField.CLOSE)

    def test_get_nonexistent_field(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test getting non-existent field."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        handler.load(data)
        # Funding rate column doesn't exist
        result = handler.get_field(CryptoField.FUNDING_RATE)
        assert result is None

    def test_invalid_resample_timeframe(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test resampling with invalid timeframe."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="D"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        handler.load(data)
        # Cannot resample daily to hourly (upsample)
        with pytest.raises(ValueError, match="Cannot resample"):
            handler.resample(TimeFrame.H1)


class TestCryptoDataHandlerPerformance:
    """Performance tests."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_load_performance(self, handler: CryptoDataHandler) -> None:
        """Test data loading performance."""
        import time

        large_data = pd.DataFrame({
            "datetime": pd.date_range(start="2020-01-01", periods=50000, freq="h"),
            "symbol": "BTC-USDT",
            "open": np.random.uniform(40000, 45000, 50000),
            "high": np.random.uniform(45000, 50000, 50000),
            "low": np.random.uniform(35000, 40000, 50000),
            "close": np.random.uniform(40000, 45000, 50000),
            "volume": np.random.uniform(1000, 10000, 50000),
        })

        start = time.time()
        handler.load(large_data)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should load within 1 second

    def test_validation_performance(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test validation performance."""
        import time

        data = pd.DataFrame({
            "datetime": pd.date_range(start="2020-01-01", periods=10000, freq="h"),
            "symbol": "BTC-USDT",
            "open": np.random.uniform(40000, 45000, 10000),
            "high": np.random.uniform(45000, 50000, 10000),
            "low": np.random.uniform(35000, 40000, 10000),
            "close": np.random.uniform(40000, 45000, 10000),
            "volume": np.random.uniform(1000, 10000, 10000),
        })

        start = time.time()
        result = handler.validate(data)
        elapsed = time.time() - start

        assert elapsed < 0.5  # Validation within 500ms


class TestCryptoDataHandlerCompatibility:
    """Compatibility tests for different exchanges and formats."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_binance_format(self, handler: CryptoDataHandler) -> None:
        """Test Binance data format compatibility."""
        binance_data = pd.DataFrame({
            "open_time": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTCUSDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
            "close_time": pd.date_range(start="2024-01-01 01:00:00", periods=10, freq="h"),
            "quote_volume": [40000000] * 10,
            "trades": [500] * 10,
        })
        handler.load(binance_data, exchange=Exchange.BINANCE)
        assert handler.data is not None

    def test_okx_format(self, handler: CryptoDataHandler) -> None:
        """Test OKX data format compatibility."""
        okx_data = pd.DataFrame({
            "ts": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "instId": "BTC-USDT-SWAP",
            "o": [40000] * 10,
            "h": [41000] * 10,
            "l": [39000] * 10,
            "c": [40500] * 10,
            "vol": [1000] * 10,
            "volCcy": [40000000] * 10,
        })
        handler.load(okx_data, exchange=Exchange.OKX)
        assert handler.data is not None

    def test_multiple_symbols(self, handler: CryptoDataHandler) -> None:
        """Test handling multiple symbols."""
        multi_symbol_data = pd.DataFrame({
            "datetime": list(pd.date_range(start="2024-01-01", periods=10, freq="h")) * 2,
            "symbol": ["BTC-USDT"] * 10 + ["ETH-USDT"] * 10,
            "open": np.random.uniform(40000, 45000, 20),
            "high": np.random.uniform(45000, 50000, 20),
            "low": np.random.uniform(35000, 40000, 20),
            "close": np.random.uniform(40000, 45000, 20),
            "volume": np.random.uniform(1000, 10000, 20),
        })
        handler.load(multi_symbol_data)
        symbols = handler.get_symbols()
        assert "BTC-USDT" in symbols
        assert "ETH-USDT" in symbols

    def test_qlib_format_export(self, handler: CryptoDataHandler) -> None:
        """Test exporting to Qlib-compatible format."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        handler.load(data)
        qlib_df = handler.to_qlib_format()
        assert qlib_df is not None
        assert "$close" in qlib_df.columns or "close" in qlib_df.columns


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        return DataValidator()

    def test_validate_success(self, validator: DataValidator) -> None:
        """Test successful validation."""
        valid_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        result = validator.validate(valid_data)
        assert result.is_valid

    def test_validate_with_warnings(
        self, validator: DataValidator
    ) -> None:
        """Test validation with warnings."""
        data_with_issues = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [0] * 10,  # Zero volume is suspicious
        })
        result = validator.validate(data_with_issues)
        assert len(result.warnings) > 0

    def test_validation_result_summary(
        self, validator: DataValidator
    ) -> None:
        """Test validation result summary."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        result = validator.validate(data)
        summary = result.get_summary()
        assert isinstance(summary, str)
