"""Tests for CryptoDataHandler Complete Functionality (Task 5).

Six-dimensional test coverage:
1. Functional: Funding rate alignment, basis calculation, OI processing
2. Boundary: Edge cases for time alignment
3. Exception: Error handling for invalid data
4. Performance: Large dataset processing
5. Security: Data validation
6. Compatibility: Multi-exchange formats
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
)


class TestFundingRateAlignment:
    """Tests for funding rate time alignment functionality."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    @pytest.fixture
    def hourly_data_with_sparse_funding(self) -> pd.DataFrame:
        """Create hourly OHLCV data with sparse funding rate (every 8 hours)."""
        # 48 hours of hourly data
        dates = pd.date_range(start="2024-01-01", periods=48, freq="h")

        # Funding rate only at 00:00, 08:00, 16:00 UTC (indices 0, 8, 16, 24, 32, 40)
        funding = [np.nan] * 48
        funding_times = [0, 8, 16, 24, 32, 40]
        funding_values = [0.0001, 0.0002, -0.0001, 0.0003, 0.0001, -0.0002]
        for i, t in enumerate(funding_times):
            funding[t] = funding_values[i]

        return pd.DataFrame({
            "datetime": dates,
            "symbol": "BTC-USDT",
            "open": np.random.uniform(40000, 45000, 48),
            "high": np.random.uniform(45000, 50000, 48),
            "low": np.random.uniform(35000, 40000, 48),
            "close": np.random.uniform(40000, 45000, 48),
            "volume": np.random.uniform(1000, 10000, 48),
            "funding_rate": funding,
        })

    def test_align_funding_rate_forward_fill(
        self, handler: CryptoDataHandler, hourly_data_with_sparse_funding: pd.DataFrame
    ) -> None:
        """Test funding rate alignment with forward fill method."""
        handler.load(hourly_data_with_sparse_funding)
        aligned = handler.align_funding_rate(method="ffill")

        # After ffill, no NaN values should remain
        assert not aligned["funding_rate"].isna().any()

        # First value should be preserved
        assert aligned.loc[0, "funding_rate"] == 0.0001

        # Values between settlements should be forward filled
        assert aligned.loc[7, "funding_rate"] == 0.0001  # Before second settlement
        assert aligned.loc[8, "funding_rate"] == 0.0002  # Second settlement

    def test_align_funding_rate_interpolate(
        self, handler: CryptoDataHandler, hourly_data_with_sparse_funding: pd.DataFrame
    ) -> None:
        """Test funding rate alignment with linear interpolation."""
        handler.load(hourly_data_with_sparse_funding)
        aligned = handler.align_funding_rate(method="interpolate")

        # After interpolation, no NaN values should remain
        assert not aligned["funding_rate"].isna().any()

        # Interpolated values should be between settlements
        # Between 0.0001 (t=0) and 0.0002 (t=8), midpoint (t=4) should be ~0.00015
        midpoint = aligned.loc[4, "funding_rate"]
        assert 0.0001 < midpoint < 0.0002

    def test_align_funding_rate_distribute(
        self, handler: CryptoDataHandler, hourly_data_with_sparse_funding: pd.DataFrame
    ) -> None:
        """Test funding rate alignment with distribution method (rate / 8)."""
        handler.load(hourly_data_with_sparse_funding)
        aligned = handler.align_funding_rate(method="distribute")

        # Distributed rate should be original rate / 8
        expected_distributed = 0.0001 / 8
        assert abs(aligned.loc[0, "funding_rate"] - expected_distributed) < 1e-10

    def test_align_funding_rate_no_funding_column(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test alignment when funding rate column is missing."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
        })
        handler.load(data)

        # Should return original data unchanged
        result = handler.align_funding_rate()
        assert "funding_rate" not in result.columns


class TestBasisCalculation:
    """Tests for basis calculation functionality."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    @pytest.fixture
    def data_with_mark_and_index(self) -> pd.DataFrame:
        """Create data with mark price and index price for basis calculation."""
        dates = pd.date_range(start="2024-01-01", periods=24, freq="h")

        # Index price (spot reference)
        index_price = np.array([40000 + i * 10 for i in range(24)])
        # Mark price (perpetual price, slightly above spot = positive basis)
        mark_price = index_price * 1.001  # 0.1% premium

        return pd.DataFrame({
            "datetime": dates,
            "symbol": "BTC-USDT",
            "close": mark_price,
            "mark_price": mark_price,
            "index_price": index_price,
            "volume": np.random.uniform(1000, 10000, 24),
        })

    def test_calculate_basis_absolute(
        self, handler: CryptoDataHandler, data_with_mark_and_index: pd.DataFrame
    ) -> None:
        """Test absolute basis calculation (mark - index)."""
        handler.load(data_with_mark_and_index)
        basis = handler.calculate_basis(method="absolute")

        assert basis is not None
        assert len(basis) == 24
        # All basis values should be positive (mark > index)
        assert (basis > 0).all()

    def test_calculate_basis_percentage(
        self, handler: CryptoDataHandler, data_with_mark_and_index: pd.DataFrame
    ) -> None:
        """Test percentage basis calculation ((mark - index) / index * 100)."""
        handler.load(data_with_mark_and_index)
        basis_pct = handler.calculate_basis(method="percentage")

        assert basis_pct is not None
        # Should be around 0.1% = 0.1
        assert abs(basis_pct.mean() - 0.1) < 0.01

    def test_calculate_annualized_basis(
        self, handler: CryptoDataHandler, data_with_mark_and_index: pd.DataFrame
    ) -> None:
        """Test annualized basis rate calculation."""
        handler.load(data_with_mark_and_index)
        annualized = handler.calculate_annualized_basis()

        assert annualized is not None
        # 0.1% per period * 365 days * 24 hours = ~876% annualized (very high)
        # But with hourly data, it should be reasonable
        assert annualized.mean() > 0

    def test_calculate_basis_missing_prices(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test basis calculation with missing mark/index prices."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
        })
        handler.load(data)

        basis = handler.calculate_basis()
        assert basis is None  # Cannot calculate without mark/index prices


class TestOpenInterestProcessing:
    """Tests for open interest processing functionality."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    @pytest.fixture
    def data_with_open_interest(self) -> pd.DataFrame:
        """Create data with open interest values."""
        dates = pd.date_range(start="2024-01-01", periods=24, freq="h")

        # Simulating OI that increases over time
        oi = np.array([100000 + i * 1000 + np.random.randint(-500, 500) for i in range(24)])

        return pd.DataFrame({
            "datetime": dates,
            "symbol": "BTC-USDT",
            "close": np.random.uniform(40000, 45000, 24),
            "volume": np.random.uniform(1000, 10000, 24),
            "open_interest": oi.astype(float),
        })

    def test_calculate_oi_change(
        self, handler: CryptoDataHandler, data_with_open_interest: pd.DataFrame
    ) -> None:
        """Test open interest change rate calculation."""
        handler.load(data_with_open_interest)
        oi_change = handler.calculate_oi_change()

        assert oi_change is not None
        assert len(oi_change) == 23  # One less due to diff

    def test_calculate_oi_change_percentage(
        self, handler: CryptoDataHandler, data_with_open_interest: pd.DataFrame
    ) -> None:
        """Test open interest percentage change calculation."""
        handler.load(data_with_open_interest)
        oi_pct_change = handler.calculate_oi_change(method="percentage")

        assert oi_pct_change is not None
        # Percentage changes should be small (around 1% per hour)
        assert abs(oi_pct_change.mean()) < 5

    def test_normalize_oi_by_price(
        self, handler: CryptoDataHandler, data_with_open_interest: pd.DataFrame
    ) -> None:
        """Test OI normalization by contract value (OI * price)."""
        handler.load(data_with_open_interest)
        normalized = handler.normalize_oi(method="contract_value")

        assert normalized is not None
        # Normalized OI should be OI * close price
        assert normalized.iloc[0] > data_with_open_interest["open_interest"].iloc[0]

    def test_oi_processing_missing_column(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test OI processing when open_interest column is missing."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
        })
        handler.load(data)

        oi_change = handler.calculate_oi_change()
        assert oi_change is None


class TestEnhancedMultiExchange:
    """Tests for enhanced multi-exchange data normalization."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_binance_futures_funding_columns(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test Binance Futures specific column mappings."""
        binance_data = pd.DataFrame({
            "openTime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTCUSDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
            "fundingRate": [0.0001] * 10,  # Binance uses camelCase
            "markPrice": [40510] * 10,
            "indexPrice": [40500] * 10,
        })

        handler.load(binance_data, exchange=Exchange.BINANCE)

        # Columns should be normalized to lowercase
        assert "funding_rate" in handler.data.columns or "fundingrate" in handler.data.columns

    def test_okx_swap_format(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test OKX Swap specific column mappings."""
        okx_data = pd.DataFrame({
            "ts": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "instId": "BTC-USDT-SWAP",
            "o": [40000] * 10,
            "h": [41000] * 10,
            "l": [39000] * 10,
            "c": [40500] * 10,
            "vol": [1000] * 10,
            "fundingRate": [0.0001] * 10,
            "markPx": [40510] * 10,
        })

        handler.load(okx_data, exchange=Exchange.OKX)

        assert handler.data is not None
        assert "close" in handler.data.columns

    def test_bybit_format(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test Bybit specific column mappings."""
        bybit_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "symbol": "BTCUSDT",
            "open": [40000] * 10,
            "high": [41000] * 10,
            "low": [39000] * 10,
            "close": [40500] * 10,
            "volume": [1000] * 10,
            "turnover": [40000000] * 10,
            "fundingRate": [0.0001] * 10,
            "openInterest": [50000000] * 10,
        })

        handler.load(bybit_data, exchange=Exchange.BYBIT)

        assert handler.data is not None
        assert "quote_volume" in handler.data.columns  # turnover mapped to quote_volume

    def test_auto_detect_exchange(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test automatic exchange format detection."""
        # OKX format has distinctive column names
        okx_data = pd.DataFrame({
            "ts": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "instId": "BTC-USDT-SWAP",
            "o": [40000] * 10,
            "h": [41000] * 10,
            "l": [39000] * 10,
            "c": [40500] * 10,
            "vol": [1000] * 10,
        })

        detected_exchange = handler.detect_exchange_format(okx_data)
        assert detected_exchange == Exchange.OKX


class TestBoundaryConditions:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_single_funding_rate(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test alignment with only one funding rate value."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
            "funding_rate": [0.0001] + [np.nan] * 9,
        })
        handler.load(data)

        aligned = handler.align_funding_rate(method="ffill")
        # All values should be filled with 0.0001
        assert (aligned["funding_rate"] == 0.0001).all()

    def test_all_nan_funding_rate(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test alignment when all funding rates are NaN."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
            "funding_rate": [np.nan] * 10,
        })
        handler.load(data)

        aligned = handler.align_funding_rate(method="ffill")
        # Original NaN values should remain (or be filled with 0)
        assert aligned["funding_rate"].isna().all() or (aligned["funding_rate"] == 0).all()

    def test_zero_open_interest(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test OI change with zero values."""
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "close": [40000] * 10,
            "open_interest": [0] * 5 + [100000] * 5,
        })
        handler.load(data)

        oi_change = handler.calculate_oi_change(method="percentage")
        # Should handle division by zero gracefully
        assert oi_change is not None
        assert not np.isinf(oi_change).any()


class TestPerformance:
    """Performance tests for large datasets."""

    @pytest.fixture
    def handler(self) -> CryptoDataHandler:
        return CryptoDataHandler()

    def test_funding_alignment_performance(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test funding rate alignment performance with large dataset."""
        import time

        # 1 year of hourly data = 8760 rows
        dates = pd.date_range(start="2023-01-01", periods=8760, freq="h")
        funding = [np.nan] * 8760
        for i in range(0, 8760, 8):
            funding[i] = np.random.uniform(-0.001, 0.001)

        data = pd.DataFrame({
            "datetime": dates,
            "close": np.random.uniform(20000, 70000, 8760),
            "funding_rate": funding,
        })

        handler.load(data)

        start = time.time()
        aligned = handler.align_funding_rate(method="ffill")
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete within 1 second
        assert len(aligned) == 8760

    def test_basis_calculation_performance(
        self, handler: CryptoDataHandler
    ) -> None:
        """Test basis calculation performance with large dataset."""
        import time

        n = 100000
        data = pd.DataFrame({
            "datetime": pd.date_range(start="2020-01-01", periods=n, freq="h"),
            "mark_price": np.random.uniform(20000, 70000, n),
            "index_price": np.random.uniform(20000, 70000, n),
        })

        handler.load(data)

        start = time.time()
        basis = handler.calculate_basis(method="percentage")
        elapsed = time.time() - start

        assert elapsed < 2.0  # Should complete within 2 seconds
