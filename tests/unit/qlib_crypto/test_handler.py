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

from iqfmp.qlib_crypto import (
    CryptoDataHandler,
    CryptoDataConfig,
    CryptoField,
    Exchange,
    TimeFrame,
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
            # Missing 'close' which is required by default
        })
        with pytest.raises(ValidationError, match="[Mm]issing.*columns"):
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
        with pytest.raises(ValidationError, match="High.*Low"):
            handler.load(inconsistent_data, validate=True)

    def test_validate_funding_rate_presence(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test validation accepts data with funding rate."""
        data_with_funding = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000] * 10,
            "high": [45000] * 10,
            "low": [35000] * 10,
            "close": [42000] * 10,
            "volume": [1000] * 10,
            "funding_rate": [0.0001] * 10,  # Typical funding rate
        })
        result = handler.validate(data_with_funding)
        assert result.is_valid


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
        """Test validation with missing values triggers warnings."""
        # Create data with many NaN values (>10% missing ratio)
        data_with_issues = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTC-USDT",
            "open": [40000, np.nan, np.nan] + [40000] * 7,  # 20% missing
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
        })
        result = validator.validate(data_with_issues)
        assert len(result.warnings) > 0  # Should warn about missing values

    def test_validation_result_structure(
        self, validator: DataValidator
    ) -> None:
        """Test validation result structure."""
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
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


# === P4.3: Order Book / Microstructure Tests ===


class TestOrderBookFields:
    """Test P4.3: Order book field definitions."""

    def test_orderbook_fields_exist(self) -> None:
        """Test orderbook-related CryptoField enums exist."""
        assert CryptoField.BID_PRICE is not None
        assert CryptoField.ASK_PRICE is not None
        assert CryptoField.BID_SIZE is not None
        assert CryptoField.ASK_SIZE is not None
        assert CryptoField.MID_PRICE is not None
        assert CryptoField.SPREAD is not None
        assert CryptoField.SPREAD_BPS is not None
        assert CryptoField.ORDER_BOOK_IMBALANCE is not None

    def test_depth_fields_exist(self) -> None:
        """Test depth aggregation fields exist."""
        assert CryptoField.DEPTH_BID_5 is not None
        assert CryptoField.DEPTH_ASK_5 is not None
        assert CryptoField.DEPTH_IMBALANCE_5 is not None
        assert CryptoField.VWAP_BID_5 is not None
        assert CryptoField.VWAP_ASK_5 is not None

    def test_field_values(self) -> None:
        """Test field string values are correct."""
        assert CryptoField.BID_PRICE.value == "bid_price"
        assert CryptoField.ASK_PRICE.value == "ask_price"
        assert CryptoField.ORDER_BOOK_IMBALANCE.value == "order_book_imbalance"


class TestOrderBookMicrostructure:
    """P4.3: Order book microstructure calculation tests."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    @pytest.fixture
    def l1_data(self) -> pd.DataFrame:
        """Create L1 order book data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        np.random.seed(42)
        bid_prices = np.random.uniform(39900, 40000, 100)
        ask_prices = bid_prices + np.random.uniform(10, 50, 100)  # Spread 10-50
        return pd.DataFrame({
            "datetime": dates,
            "symbol": "BTC-USDT",
            "open": np.random.uniform(39000, 41000, 100),
            "high": np.random.uniform(40000, 42000, 100),
            "low": np.random.uniform(38000, 40000, 100),
            "close": (bid_prices + ask_prices) / 2,
            "volume": np.random.uniform(1000, 10000, 100),
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": np.random.uniform(10, 100, 100),
            "ask_size": np.random.uniform(10, 100, 100),
        })

    # === Spread Calculation Tests ===

    def test_calculate_spread_absolute(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test absolute spread calculation."""
        handler.load(l1_data)
        spread = handler.calculate_spread(method="absolute")
        assert spread is not None
        assert len(spread) == 100
        # Spread should be positive (ask > bid)
        assert (spread > 0).all()
        # Verify calculation
        expected = l1_data["ask_price"] - l1_data["bid_price"]
        assert np.allclose(spread.values, expected.values)

    def test_calculate_spread_percentage(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test percentage spread calculation."""
        handler.load(l1_data)
        spread_pct = handler.calculate_spread(method="percentage")
        assert spread_pct is not None
        # Should be small percentage for liquid markets
        assert (spread_pct < 1).all()  # Less than 1%
        assert (spread_pct > 0).all()

    def test_calculate_spread_bps(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test spread in basis points."""
        handler.load(l1_data)
        spread_bps = handler.calculate_spread(method="bps")
        assert spread_bps is not None
        # Should be ~25-125 bps for 10-50 absolute spread on 40k price
        assert (spread_bps > 0).all()
        assert (spread_bps < 200).all()

    def test_calculate_spread_no_data(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test spread calculation without L1 data."""
        # Load OHLCV without bid/ask
        ohlcv_only = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
            "volume": [1000] * 10,
        })
        handler.load(ohlcv_only)
        spread = handler.calculate_spread()
        assert spread is None

    # === Mid Price Tests ===

    def test_calculate_mid_price(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test mid price calculation."""
        handler.load(l1_data)
        mid = handler.calculate_mid_price()
        assert mid is not None
        assert len(mid) == 100
        # Mid should be between bid and ask
        bid = l1_data["bid_price"]
        ask = l1_data["ask_price"]
        assert (mid >= bid).all()
        assert (mid <= ask).all()

    # === Order Book Imbalance Tests ===

    def test_calculate_order_book_imbalance(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test L1 order book imbalance calculation."""
        handler.load(l1_data)
        imbalance = handler.calculate_order_book_imbalance()
        assert imbalance is not None
        assert len(imbalance) == 100
        # Imbalance should be between -1 and 1
        assert (imbalance >= -1).all()
        assert (imbalance <= 1).all()

    def test_imbalance_with_equal_sizes(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test imbalance when bid/ask sizes are equal."""
        balanced_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "bid_price": [39990] * 10,
            "ask_price": [40010] * 10,
            "bid_size": [50] * 10,
            "ask_size": [50] * 10,
        })
        handler.load(balanced_data)
        imbalance = handler.calculate_order_book_imbalance()
        assert imbalance is not None
        # Equal sizes should give 0 imbalance
        assert np.allclose(imbalance.values, 0)

    def test_imbalance_with_all_bid(self, handler: CryptoDataHandler) -> None:
        """Test imbalance when only bid size exists."""
        bid_heavy_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "bid_price": [39990] * 10,
            "ask_price": [40010] * 10,
            "bid_size": [100] * 10,
            "ask_size": [0] * 10,  # No ask
        })
        handler.load(bid_heavy_data)
        imbalance = handler.calculate_order_book_imbalance()
        # Division by zero handling should return NaN
        assert imbalance is not None

    # === Microprice Tests ===

    def test_calculate_microprice(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test microprice calculation."""
        handler.load(l1_data)
        microprice = handler.calculate_microprice()
        assert microprice is not None
        assert len(microprice) == 100
        # Microprice should be between bid and ask
        bid = l1_data["bid_price"]
        ask = l1_data["ask_price"]
        assert (microprice >= bid).all()
        assert (microprice <= ask).all()

    def test_microprice_vs_midprice(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test microprice differs from midprice with asymmetric sizes."""
        asymmetric_data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "bid_price": [100] * 10,
            "ask_price": [102] * 10,
            "bid_size": [100] * 10,  # Large bid
            "ask_size": [10] * 10,   # Small ask
        })
        handler.load(asymmetric_data)
        mid = handler.calculate_mid_price()
        microprice = handler.calculate_microprice()
        # With larger bid size, microprice should be closer to ask
        assert microprice is not None and mid is not None
        assert (microprice > mid).all()

    # === Effective Spread Tests ===

    def test_calculate_effective_spread(
        self, handler: CryptoDataHandler, l1_data: pd.DataFrame
    ) -> None:
        """Test effective spread calculation."""
        handler.load(l1_data)
        eff_spread = handler.calculate_effective_spread()
        assert eff_spread is not None
        assert len(eff_spread) == 100
        # Effective spread should be non-negative
        assert (eff_spread >= 0).all()


class TestOrderBookSnapshot:
    """P4.3: Order book snapshot processing tests."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_load_orderbook_snapshot_basic(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test processing a single order book snapshot."""
        bids = [(40000, 1.5), (39990, 2.0), (39980, 3.0), (39970, 1.0), (39960, 0.5)]
        asks = [(40010, 1.0), (40020, 2.5), (40030, 1.5), (40040, 2.0), (40050, 1.0)]

        features = handler.load_orderbook_snapshot(bids, asks)

        # L1 data
        assert features["bid_price"] == 40000
        assert features["ask_price"] == 40010
        assert features["bid_size"] == 1.5
        assert features["ask_size"] == 1.0

        # Mid and spread
        assert features["mid_price"] == 40005
        assert features["spread"] == 10
        assert features["spread_bps"] == pytest.approx(2.499, rel=0.01)

        # Imbalance (1.5 - 1.0) / (1.5 + 1.0) = 0.2
        assert features["order_book_imbalance"] == pytest.approx(0.2)

    def test_load_orderbook_snapshot_depth(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test depth aggregation from snapshot."""
        bids = [(100, 10), (99, 20), (98, 30), (97, 40), (96, 50)]
        asks = [(101, 5), (102, 15), (103, 25), (104, 35), (105, 45)]

        features = handler.load_orderbook_snapshot(bids, asks, levels=5)

        # Depth sums
        assert features["depth_bid_5"] == 150  # 10+20+30+40+50
        assert features["depth_ask_5"] == 125  # 5+15+25+35+45

        # Depth imbalance: (150 - 125) / (150 + 125) = 0.0909
        assert features["depth_imbalance_5"] == pytest.approx(0.0909, rel=0.01)

    def test_load_orderbook_snapshot_vwap(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test VWAP calculation from snapshot."""
        bids = [(100, 10), (99, 20)]  # VWAP = (100*10 + 99*20) / 30 = 99.33
        asks = [(101, 5), (102, 15)]   # VWAP = (101*5 + 102*15) / 20 = 101.75

        features = handler.load_orderbook_snapshot(bids, asks, levels=2)

        assert features["vwap_bid_2"] == pytest.approx(99.333, rel=0.01)
        assert features["vwap_ask_2"] == pytest.approx(101.75)

    def test_load_orderbook_snapshot_with_timestamp(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test snapshot with timestamp."""
        bids = [(100, 10)]
        asks = [(101, 10)]
        ts = pd.Timestamp("2024-01-01 12:00:00")

        features = handler.load_orderbook_snapshot(bids, asks, timestamp=ts)
        assert features["datetime"] == ts

    def test_load_orderbook_snapshot_empty_sides(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test handling empty bid/ask sides."""
        features = handler.load_orderbook_snapshot([], [])

        assert "bid_price" not in features
        assert "ask_price" not in features
        assert features["depth_bid_5"] == 0
        assert features["depth_ask_5"] == 0


class TestOrderBookStream:
    """P4.3: Order book stream processing tests."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_process_orderbook_stream(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test processing a stream of order book snapshots."""
        snapshots = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "bids": [(100, 10), (99, 20)],
                "asks": [(101, 15), (102, 25)],
            },
            {
                "timestamp": "2024-01-01T00:01:00",
                "bids": [(100.5, 12), (99.5, 18)],
                "asks": [(101.5, 14), (102.5, 22)],
            },
            {
                "timestamp": "2024-01-01T00:02:00",
                "bids": [(101, 8), (100, 22)],
                "asks": [(102, 10), (103, 30)],
            },
        ]

        df = handler.process_orderbook_stream(snapshots, levels=2)

        assert len(df) == 3
        assert "bid_price" in df.columns
        assert "ask_price" in df.columns
        assert "spread" in df.columns
        assert "depth_bid_2" in df.columns
        assert "datetime" in df.columns

    def test_process_orderbook_stream_empty(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test processing empty stream."""
        df = handler.process_orderbook_stream([])
        assert len(df) == 0

    def test_process_orderbook_stream_performance(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test stream processing performance."""
        import time

        # Generate 1000 snapshots
        np.random.seed(42)
        snapshots = []
        for i in range(1000):
            base_price = 40000 + np.random.uniform(-100, 100)
            bids = [(base_price - j * 10, np.random.uniform(1, 10)) for j in range(5)]
            asks = [(base_price + j * 10 + 10, np.random.uniform(1, 10)) for j in range(5)]
            snapshots.append({
                "timestamp": f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00",
                "bids": bids,
                "asks": asks,
            })

        start = time.time()
        df = handler.process_orderbook_stream(snapshots)
        elapsed = time.time() - start

        assert len(df) == 1000
        assert elapsed < 2.0  # Should process 1000 snapshots in under 2 seconds


class TestDepthImbalance:
    """P4.3: Depth imbalance calculation tests."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_calculate_depth_imbalance(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test depth imbalance calculation from loaded data."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "depth_bid_5": [100, 120, 80, 150, 90, 110, 130, 70, 100, 95],
            "depth_ask_5": [90, 80, 120, 100, 110, 90, 70, 130, 100, 105],
        })
        handler.load(data)
        imbalance = handler.calculate_depth_imbalance(levels=5)

        assert imbalance is not None
        assert len(imbalance) == 10
        # Check bounds
        assert (imbalance >= -1).all()
        assert (imbalance <= 1).all()

        # First row: (100-90)/(100+90) = 0.0526
        assert imbalance.iloc[0] == pytest.approx(0.0526, rel=0.01)

    def test_calculate_depth_imbalance_no_data(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test depth imbalance without depth fields."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
        })
        handler.load(data)
        imbalance = handler.calculate_depth_imbalance()
        assert imbalance is None
