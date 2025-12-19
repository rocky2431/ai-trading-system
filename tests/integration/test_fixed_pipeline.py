"""Integration tests for the fixed IQFMP pipeline.

Tests the complete flow:
1. LLM generates factor code
2. Factor code is validated
3. Factor is computed
4. Signal is converted
5. Backtest is run (via Qlib ONLY)

This validates that the architecture fixes properly bridge
LLM-generated pandas factors to Qlib-compatible backtests.

NOTE: Qlib is REQUIRED for backtesting. Tests that require Qlib
will be skipped if Qlib is not available.
"""

import numpy as np
import pandas as pd
import pytest

# Check if Qlib is available
try:
    from iqfmp.agents.backtest_agent import QLIB_AVAILABLE
except ImportError:
    QLIB_AVAILABLE = False

# Skip marker for tests requiring Qlib
requires_qlib = pytest.mark.skipif(
    not QLIB_AVAILABLE,
    reason="Qlib is REQUIRED for backtesting. Please ensure PYTHONPATH includes vendor/qlib."
)


@pytest.fixture
def sample_crypto_data() -> pd.DataFrame:
    """Generate sample crypto OHLCV + derivatives data."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="1h")

    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, len(dates))
    close = 50000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": "BTCUSDT",
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.uniform(1e6, 1e8, len(dates)),
            "funding_rate": np.random.normal(0.0001, 0.0005, len(dates)),
            "open_interest": np.random.uniform(1e9, 2e9, len(dates)),
        }
    )

    return df.set_index(["datetime", "instrument"])


@pytest.fixture
def sample_factor_code() -> str:
    """Sample LLM-generated factor code."""
    return '''
def funding_momentum(df: pd.DataFrame) -> pd.Series:
    """Funding rate momentum factor.

    Captures momentum in funding rates as a signal for
    directional crypto trades.
    """
    funding = df["funding_rate"]
    short_ma = funding.rolling(8).mean()
    long_ma = funding.rolling(24).mean()
    return short_ma - long_ma
'''


@pytest.fixture
def invalid_factor_code() -> str:
    """Invalid factor code with security issues."""
    return '''
import os  # Forbidden import

def bad_factor(df: pd.DataFrame) -> pd.Series:
    os.system("echo hello")  # Forbidden operation
    return df["close"]
'''


class TestFactorValidation:
    """Test factor code validation."""

    def test_valid_factor_code(self, sample_factor_code: str) -> None:
        """Valid factor code should pass validation."""
        from iqfmp.core.factor_validator import FactorCodeValidator

        validator = FactorCodeValidator()
        result = validator.validate(sample_factor_code, "funding_momentum")

        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_factor_code(self, invalid_factor_code: str) -> None:
        """Invalid factor code should fail validation."""
        from iqfmp.core.factor_validator import FactorCodeValidator

        validator = FactorCodeValidator()
        result = validator.validate(invalid_factor_code, "bad_factor")

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("Forbidden" in e for e in result.errors)

    def test_extract_required_fields(self, sample_factor_code: str) -> None:
        """Should extract required data fields from factor code."""
        from iqfmp.core.factor_validator import FactorCodeValidator

        validator = FactorCodeValidator()
        fields = validator.extract_required_fields(sample_factor_code)

        assert "funding_rate" in fields


class TestSignalConversion:
    """Test signal conversion."""

    def test_normalize_zscore(self) -> None:
        """Z-score normalization should center and scale."""
        from iqfmp.core.signal_converter import SignalConverter, SignalConfig

        config = SignalConfig(normalize_method="zscore")
        converter = SignalConverter(config)

        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = converter.normalize(factor)

        assert abs(normalized.mean()) < 0.01  # Should be near zero
        assert abs(normalized.std() - 1.0) < 0.1  # Should be near 1

    def test_to_signal(self, sample_crypto_data: pd.DataFrame) -> None:
        """Factor values should convert to bounded signals."""
        from iqfmp.core.signal_converter import SignalConverter

        # Create sample factor values
        factor = pd.Series(
            np.random.randn(len(sample_crypto_data)),
            index=sample_crypto_data.index,
        )

        converter = SignalConverter()
        signal = converter.to_signal(factor)

        assert signal.min() >= -1
        assert signal.max() <= 1

    def test_to_signal_topk(self) -> None:
        """Top-k selection should produce discrete signals."""
        from iqfmp.core.signal_converter import SignalConverter, SignalConfig

        config = SignalConfig(top_k=2)
        converter = SignalConverter(config)

        factor = pd.Series([0.1, 0.5, 0.3, 0.8, 0.2])
        signal = converter.to_signal(factor, normalize=False)

        # Should have exactly 2 longs (top 2) and 2 shorts (bottom 2)
        assert (signal == 1.0).sum() == 2
        assert (signal == -1.0).sum() == 2
        assert (signal == 0.0).sum() == 1

    def test_to_qlib_format(self, sample_crypto_data: pd.DataFrame) -> None:
        """Should convert to Qlib MultiIndex format."""
        from iqfmp.core.signal_converter import SignalConverter

        converter = SignalConverter()

        # Reset index for conversion
        df = sample_crypto_data.reset_index()
        qlib_df = converter.to_qlib_format(df)

        assert isinstance(qlib_df.index, pd.MultiIndex)
        assert qlib_df.index.names == ["datetime", "instrument"]


@requires_qlib
class TestQlibBacktest:
    """Test Qlib backtest engine (ONLY allowed backtest engine)."""

    def test_run_backtest(self, sample_crypto_data: pd.DataFrame) -> None:
        """Should run backtest via Qlib and return metrics."""
        from iqfmp.agents.backtest_agent import QlibBacktestEngine

        # Create sample signal and returns
        price_data = sample_crypto_data.reset_index(level="instrument", drop=True)
        returns = price_data["close"].pct_change().fillna(0)
        signal = pd.Series(
            np.random.choice([-1.0, 0.0, 1.0], len(price_data)),
            index=price_data.index,
        )

        engine = QlibBacktestEngine()
        result = engine.run(signal, returns)

        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.total_return is not None

    def test_qlib_backtest_with_prices(self, sample_crypto_data: pd.DataFrame) -> None:
        """Should run backtest with price data."""
        from iqfmp.agents.backtest_agent import QlibBacktestEngine

        price_data = sample_crypto_data.reset_index(level="instrument", drop=True)
        returns = price_data["close"].pct_change().fillna(0)
        prices = price_data["close"]
        signal = pd.Series(
            np.random.choice([-1.0, 0.0, 1.0], len(price_data)),
            index=price_data.index,
        )

        engine = QlibBacktestEngine()
        result = engine.run(signal, returns, prices=prices)

        assert result.sharpe_ratio is not None
        assert result.n_trades >= 0


class TestEndToEnd:
    """Test complete pipeline."""

    @requires_qlib
    def test_full_pipeline(
        self, sample_crypto_data: pd.DataFrame, sample_factor_code: str
    ) -> None:
        """Test complete factor → signal → backtest pipeline using Qlib."""
        from iqfmp.agents.backtest_agent import QlibBacktestEngine
        from iqfmp.core.factor_validator import FactorCodeValidator
        from iqfmp.core.signal_converter import SignalConverter

        # 1. Validate factor code
        validator = FactorCodeValidator()
        validation = validator.validate(sample_factor_code, "funding_momentum")
        assert validation.is_valid

        # 2. Execute factor code
        local_namespace: dict = {"pd": pd}
        exec(sample_factor_code, local_namespace)  # noqa: S102
        factor_func = local_namespace["funding_momentum"]

        price_data = sample_crypto_data.reset_index(level="instrument", drop=True)
        factor_values = factor_func(price_data)

        # 3. Convert to signal
        converter = SignalConverter()
        signal = converter.to_signal(factor_values.dropna())

        # 4. Prepare returns for Qlib backtest
        returns = price_data["close"].pct_change().fillna(0)

        # 5. Run Qlib backtest (ONLY allowed backtest engine)
        engine = QlibBacktestEngine()
        # Align signal and returns
        common_idx = signal.index.intersection(returns.index)
        result = engine.run(signal.loc[common_idx], returns.loc[common_idx])

        # 6. Verify metrics
        assert result.sharpe_ratio is not None
        assert result.max_drawdown <= 0  # Drawdown is negative or zero

    def test_signal_converter_with_dataframe_input(self) -> None:
        """Signal converter should handle DataFrame input."""
        from iqfmp.core.signal_converter import SignalConverter

        converter = SignalConverter()

        # Create DataFrame with 'value' column (common LLM output format)
        df = pd.DataFrame(
            {"value": [0.1, -0.2, 0.3, -0.1, 0.5]},
            index=pd.date_range("2024-01-01", periods=5),
        )

        signal = converter.to_signal(df)

        assert len(signal) == 5
        assert signal.min() >= -1
        assert signal.max() <= 1


class TestQlibPredictionDataset:
    """Test Qlib-compatible prediction dataset."""

    def test_create_prediction_dataset(self) -> None:
        """Should create a Qlib-compatible dataset."""
        from iqfmp.core.signal_converter import (
            QlibPredictionDataset,
            SignalConverter,
        )

        signal = pd.Series([0.1, 0.2, -0.1, 0.3], name="score")
        converter = SignalConverter()

        dataset = converter.create_prediction_dataset(
            signal=signal,
            instruments=["BTCUSDT"],
            start_time="2024-01-01",
            end_time="2024-01-04",
        )

        assert isinstance(dataset, QlibPredictionDataset)

        # Test Qlib interface
        dataset.prepare()
        assert dataset.is_prepared

        segments = dataset.get_segments()
        assert "train" in segments
        assert "test" in segments

        df = dataset.to_dataframe()
        assert "score" in df.columns


class TestCryptoDataHandler:
    """Test CryptoDataHandler integration."""

    def test_load_from_dataframe(self, sample_crypto_data: pd.DataFrame) -> None:
        """Should load data from DataFrame."""
        from iqfmp.core.qlib_crypto import CryptoDataHandler

        df = sample_crypto_data.reset_index()
        handler = CryptoDataHandler(instruments=["BTCUSDT"])
        handler.load_data(df=df)

        assert handler.data is not None
        assert "$close" in handler.data.columns
        assert "$volume" in handler.data.columns

    def test_add_technical_indicators(
        self, sample_crypto_data: pd.DataFrame
    ) -> None:
        """Should add technical indicators."""
        from iqfmp.core.qlib_crypto import CryptoDataHandler

        df = sample_crypto_data.reset_index()
        handler = CryptoDataHandler(instruments=["BTCUSDT"])
        handler.load_data(df=df)
        handler.add_technical_indicators()

        assert "$rsi_14" in handler.data.columns
        assert "$macd" in handler.data.columns
        assert "$bb_upper" in handler.data.columns
