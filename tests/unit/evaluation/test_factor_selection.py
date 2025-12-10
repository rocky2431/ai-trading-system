"""Tests for Factor Selection (Task 14).

Six-dimensional test coverage for factor library and selection.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Any

from iqfmp.evaluation.factor_selection import (
    FactorLibrary,
    FactorEntry,
    CorrelationAnalyzer,
    CorrelationMatrix,
    FactorSelector,
    SelectionConfig,
    SelectionResult,
    HighCorrelationWarning,
    InvalidSelectionError,
)


@pytest.fixture
def sample_factors() -> list[FactorEntry]:
    """Create sample factor entries."""
    return [
        FactorEntry(
            name="momentum_20d",
            family="momentum",
            ic=0.05,
            ir=1.5,
            sharpe=1.2,
            stability_score=0.8,
        ),
        FactorEntry(
            name="momentum_60d",
            family="momentum",
            ic=0.04,
            ir=1.3,
            sharpe=1.0,
            stability_score=0.75,
        ),
        FactorEntry(
            name="value_pb",
            family="value",
            ic=0.03,
            ir=1.1,
            sharpe=0.9,
            stability_score=0.7,
        ),
        FactorEntry(
            name="volatility_30d",
            family="volatility",
            ic=0.06,
            ir=1.8,
            sharpe=1.5,
            stability_score=0.85,
        ),
    ]


@pytest.fixture
def library(sample_factors: list[FactorEntry]) -> FactorLibrary:
    """Create factor library with sample factors."""
    lib = FactorLibrary()
    for factor in sample_factors:
        lib.add(factor)
    return lib


@pytest.fixture
def factor_values() -> pd.DataFrame:
    """Create sample factor value matrix."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=100)

    return pd.DataFrame({
        "momentum_20d": np.random.randn(100),
        "momentum_60d": np.random.randn(100) * 0.9 + np.random.randn(100) * 0.1,
        "value_pb": np.random.randn(100),
        "volatility_30d": np.random.randn(100),
    }, index=dates)


class TestFactorEntry:
    """Tests for FactorEntry."""

    def test_create_entry(self) -> None:
        """Test creating factor entry."""
        entry = FactorEntry(
            name="test_factor",
            family="momentum",
            ic=0.05,
            ir=1.5,
        )
        assert entry.name == "test_factor"
        assert entry.family == "momentum"

    def test_entry_to_dict(self) -> None:
        """Test entry serialization."""
        entry = FactorEntry(name="test", family="momentum", ic=0.05)
        d = entry.to_dict()
        assert "name" in d
        assert "family" in d

    def test_entry_from_dict(self) -> None:
        """Test entry deserialization."""
        data = {"name": "test", "family": "momentum", "ic": 0.05}
        entry = FactorEntry.from_dict(data)
        assert entry.name == "test"


class TestFactorLibrary:
    """Tests for FactorLibrary."""

    def test_add_factor(self, library: FactorLibrary) -> None:
        """Test adding factor to library."""
        assert library.count() == 4

    def test_get_factor(self, library: FactorLibrary) -> None:
        """Test retrieving factor by name."""
        factor = library.get("momentum_20d")
        assert factor is not None
        assert factor.name == "momentum_20d"

    def test_get_all(self, library: FactorLibrary) -> None:
        """Test getting all factors."""
        all_factors = library.get_all()
        assert len(all_factors) == 4

    def test_filter_by_family(self, library: FactorLibrary) -> None:
        """Test filtering by family."""
        momentum = library.filter(family="momentum")
        assert len(momentum) == 2
        assert all(f.family == "momentum" for f in momentum)

    def test_filter_by_ic(self, library: FactorLibrary) -> None:
        """Test filtering by IC threshold."""
        high_ic = library.filter(min_ic=0.04)
        assert len(high_ic) == 3

    def test_filter_by_stability(self, library: FactorLibrary) -> None:
        """Test filtering by stability score."""
        stable = library.filter(min_stability=0.8)
        assert len(stable) == 2

    def test_sort_by_ic(self, library: FactorLibrary) -> None:
        """Test sorting by IC."""
        sorted_factors = library.sort_by("ic", descending=True)
        assert sorted_factors[0].ic >= sorted_factors[-1].ic

    def test_sort_by_ir(self, library: FactorLibrary) -> None:
        """Test sorting by IR."""
        sorted_factors = library.sort_by("ir", descending=True)
        assert sorted_factors[0].ir >= sorted_factors[-1].ir

    def test_remove_factor(self, library: FactorLibrary) -> None:
        """Test removing factor."""
        library.remove("momentum_20d")
        assert library.count() == 3
        assert library.get("momentum_20d") is None

    def test_update_factor(self, library: FactorLibrary) -> None:
        """Test updating factor."""
        library.update("momentum_20d", ic=0.06)
        factor = library.get("momentum_20d")
        assert factor.ic == 0.06


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzer."""

    def test_calculate_correlation_matrix(
        self, factor_values: pd.DataFrame
    ) -> None:
        """Test correlation matrix calculation."""
        analyzer = CorrelationAnalyzer()
        matrix = analyzer.calculate(factor_values)

        assert isinstance(matrix, CorrelationMatrix)
        assert matrix.shape[0] == 4
        assert matrix.shape[1] == 4

    def test_get_correlation_pair(
        self, factor_values: pd.DataFrame
    ) -> None:
        """Test getting correlation between two factors."""
        analyzer = CorrelationAnalyzer()
        matrix = analyzer.calculate(factor_values)

        corr = matrix.get_correlation("momentum_20d", "value_pb")
        assert -1 <= corr <= 1

    def test_find_high_correlations(
        self, factor_values: pd.DataFrame
    ) -> None:
        """Test finding high correlation pairs."""
        analyzer = CorrelationAnalyzer()
        matrix = analyzer.calculate(factor_values)

        high_pairs = matrix.find_high_correlations(threshold=0.3)
        assert isinstance(high_pairs, list)

    def test_correlation_warning(
        self, factor_values: pd.DataFrame
    ) -> None:
        """Test high correlation warning."""
        # Create highly correlated factors
        df = factor_values.copy()
        df["momentum_copy"] = df["momentum_20d"] * 0.99 + np.random.randn(100) * 0.01

        analyzer = CorrelationAnalyzer()
        matrix = analyzer.calculate(df)

        high_pairs = matrix.find_high_correlations(threshold=0.9)
        assert len(high_pairs) > 0

    def test_correlation_matrix_to_dataframe(
        self, factor_values: pd.DataFrame
    ) -> None:
        """Test converting matrix to DataFrame."""
        analyzer = CorrelationAnalyzer()
        matrix = analyzer.calculate(factor_values)

        df = matrix.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 4)


class TestFactorSelector:
    """Tests for FactorSelector."""

    def test_select_top_n(
        self, library: FactorLibrary, factor_values: pd.DataFrame
    ) -> None:
        """Test selecting top N factors."""
        selector = FactorSelector(library)
        result = selector.select_top_n(n=2, sort_by="ic")

        assert len(result.selected) == 2

    def test_select_by_family(
        self, library: FactorLibrary
    ) -> None:
        """Test selecting one per family."""
        selector = FactorSelector(library)
        result = selector.select_by_family(one_per_family=True)

        families = [f.family for f in result.selected]
        assert len(families) == len(set(families))

    def test_select_with_correlation_filter(
        self, library: FactorLibrary, factor_values: pd.DataFrame
    ) -> None:
        """Test selection with correlation filtering."""
        config = SelectionConfig(
            max_correlation=0.7,
            factor_values=factor_values,
        )
        selector = FactorSelector(library, config)
        result = selector.select_uncorrelated(n=3)

        assert len(result.selected) <= 3

    def test_select_returns_warnings(
        self, library: FactorLibrary, factor_values: pd.DataFrame
    ) -> None:
        """Test that selection returns warnings."""
        # Add correlated factor
        library.add(FactorEntry(
            name="momentum_copy",
            family="momentum",
            ic=0.045,
            ir=1.4,
        ))

        config = SelectionConfig(factor_values=factor_values)
        selector = FactorSelector(library, config)
        result = selector.select_all()

        # Should have warnings about correlation
        assert isinstance(result.warnings, list)

    def test_manual_selection(
        self, library: FactorLibrary
    ) -> None:
        """Test manual factor selection."""
        selector = FactorSelector(library)
        result = selector.select_manual(["momentum_20d", "value_pb"])

        assert len(result.selected) == 2
        names = [f.name for f in result.selected]
        assert "momentum_20d" in names
        assert "value_pb" in names

    def test_validate_selection(
        self, library: FactorLibrary, factor_values: pd.DataFrame
    ) -> None:
        """Test selection validation."""
        config = SelectionConfig(
            min_factors=2,
            max_factors=5,
            factor_values=factor_values,
        )
        selector = FactorSelector(library, config)

        # Valid selection
        result = selector.select_manual(["momentum_20d", "value_pb"])
        assert result.is_valid

    def test_invalid_selection_too_few(
        self, library: FactorLibrary
    ) -> None:
        """Test invalid selection with too few factors."""
        config = SelectionConfig(min_factors=3)
        selector = FactorSelector(library, config)

        result = selector.select_manual(["momentum_20d"])
        assert not result.is_valid


class TestSelectionResult:
    """Tests for SelectionResult."""

    def test_result_to_dict(
        self, library: FactorLibrary
    ) -> None:
        """Test result serialization."""
        selector = FactorSelector(library)
        result = selector.select_top_n(n=2)

        d = result.to_dict()
        assert "selected" in d
        assert "warnings" in d

    def test_result_get_names(
        self, library: FactorLibrary
    ) -> None:
        """Test getting selected factor names."""
        selector = FactorSelector(library)
        result = selector.select_top_n(n=2)

        names = result.get_names()
        assert isinstance(names, list)
        assert len(names) == 2


class TestSelectionBoundary:
    """Boundary tests."""

    def test_empty_library(self) -> None:
        """Test with empty library."""
        library = FactorLibrary()
        selector = FactorSelector(library)

        result = selector.select_top_n(n=5)
        assert len(result.selected) == 0

    def test_select_more_than_available(
        self, library: FactorLibrary
    ) -> None:
        """Test selecting more factors than available."""
        selector = FactorSelector(library)
        result = selector.select_top_n(n=100)

        assert len(result.selected) == 4  # All available

    def test_single_factor(self) -> None:
        """Test with single factor."""
        library = FactorLibrary()
        library.add(FactorEntry(name="only_factor", family="test", ic=0.05))

        selector = FactorSelector(library)
        result = selector.select_top_n(n=1)

        assert len(result.selected) == 1


class TestSelectionException:
    """Exception handling tests."""

    def test_invalid_factor_name(
        self, library: FactorLibrary
    ) -> None:
        """Test selecting nonexistent factor."""
        selector = FactorSelector(library)

        with pytest.raises(InvalidSelectionError, match="not found"):
            selector.select_manual(["nonexistent_factor"])

    def test_invalid_sort_field(
        self, library: FactorLibrary
    ) -> None:
        """Test invalid sort field."""
        with pytest.raises(ValueError, match="sort"):
            library.sort_by("invalid_field")


class TestSelectionPerformance:
    """Performance tests."""

    def test_large_library_selection(self) -> None:
        """Test selection with large library."""
        import time

        library = FactorLibrary()
        for i in range(1000):
            library.add(FactorEntry(
                name=f"factor_{i}",
                family=f"family_{i % 10}",
                ic=np.random.uniform(0.01, 0.1),
                ir=np.random.uniform(0.5, 2.0),
            ))

        selector = FactorSelector(library)

        start = time.time()
        result = selector.select_top_n(n=50)
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert len(result.selected) == 50


class TestSelectionCompatibility:
    """Compatibility tests."""

    def test_library_export_import(
        self, library: FactorLibrary
    ) -> None:
        """Test library export and import."""
        exported = library.export_to_dict()

        new_library = FactorLibrary()
        new_library.import_from_dict(exported)

        assert new_library.count() == library.count()
