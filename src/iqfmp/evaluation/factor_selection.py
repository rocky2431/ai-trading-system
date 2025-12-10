"""Factor Selection for portfolio construction.

This module provides:
- FactorLibrary: Storage and retrieval of factors
- CorrelationAnalyzer: Factor correlation analysis
- FactorSelector: Factor selection strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd


class HighCorrelationWarning(UserWarning):
    """Warning for high factor correlation."""

    pass


class InvalidSelectionError(Exception):
    """Raised when factor selection is invalid."""

    pass


@dataclass
class FactorEntry:
    """Entry representing a factor in the library."""

    name: str
    family: str
    ic: float = 0.0
    ir: float = 0.0
    sharpe: float = 0.0
    stability_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "family": self.family,
            "ic": self.ic,
            "ir": self.ir,
            "sharpe": self.sharpe,
            "stability_score": self.stability_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactorEntry:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            family=data["family"],
            ic=data.get("ic", 0.0),
            ir=data.get("ir", 0.0),
            sharpe=data.get("sharpe", 0.0),
            stability_score=data.get("stability_score", 0.0),
            metadata=data.get("metadata", {}),
        )


class FactorLibrary:
    """Library for storing and managing factors."""

    def __init__(self) -> None:
        """Initialize empty library."""
        self._factors: dict[str, FactorEntry] = {}

    def add(self, factor: FactorEntry) -> None:
        """Add factor to library."""
        self._factors[factor.name] = factor

    def get(self, name: str) -> Optional[FactorEntry]:
        """Get factor by name."""
        return self._factors.get(name)

    def get_all(self) -> list[FactorEntry]:
        """Get all factors."""
        return list(self._factors.values())

    def count(self) -> int:
        """Get number of factors."""
        return len(self._factors)

    def remove(self, name: str) -> None:
        """Remove factor from library."""
        if name in self._factors:
            del self._factors[name]

    def update(self, name: str, **kwargs: Any) -> None:
        """Update factor attributes."""
        if name in self._factors:
            factor = self._factors[name]
            for key, value in kwargs.items():
                if hasattr(factor, key):
                    setattr(factor, key, value)

    def filter(
        self,
        family: Optional[str] = None,
        min_ic: Optional[float] = None,
        min_ir: Optional[float] = None,
        min_stability: Optional[float] = None,
    ) -> list[FactorEntry]:
        """Filter factors by criteria."""
        result = list(self._factors.values())

        if family is not None:
            result = [f for f in result if f.family == family]

        if min_ic is not None:
            result = [f for f in result if abs(f.ic) >= min_ic]

        if min_ir is not None:
            result = [f for f in result if f.ir >= min_ir]

        if min_stability is not None:
            result = [f for f in result if f.stability_score >= min_stability]

        return result

    def sort_by(
        self, field: str, descending: bool = True
    ) -> list[FactorEntry]:
        """Sort factors by field."""
        valid_fields = ["ic", "ir", "sharpe", "stability_score", "name"]
        if field not in valid_fields:
            raise ValueError(f"Invalid sort field: {field}")

        factors = list(self._factors.values())

        if field == "ic":
            factors.sort(key=lambda f: abs(f.ic), reverse=descending)
        else:
            factors.sort(key=lambda f: getattr(f, field), reverse=descending)

        return factors

    def export_to_dict(self) -> dict[str, Any]:
        """Export library to dictionary."""
        return {
            "factors": [f.to_dict() for f in self._factors.values()],
        }

    def import_from_dict(self, data: dict[str, Any]) -> None:
        """Import factors from dictionary."""
        for factor_data in data.get("factors", []):
            factor = FactorEntry.from_dict(factor_data)
            self.add(factor)


@dataclass
class CorrelationPair:
    """A pair of correlated factors."""

    factor1: str
    factor2: str
    correlation: float


class CorrelationMatrix:
    """Matrix of factor correlations."""

    def __init__(self, matrix: pd.DataFrame) -> None:
        """Initialize with correlation matrix."""
        self._matrix = matrix

    @property
    def shape(self) -> tuple[int, int]:
        """Get matrix shape."""
        return self._matrix.shape

    def get_correlation(self, factor1: str, factor2: str) -> float:
        """Get correlation between two factors."""
        if factor1 not in self._matrix.columns:
            return 0.0
        if factor2 not in self._matrix.columns:
            return 0.0
        return float(self._matrix.loc[factor1, factor2])

    def find_high_correlations(
        self, threshold: float = 0.7
    ) -> list[CorrelationPair]:
        """Find pairs with correlation above threshold."""
        pairs = []
        factors = self._matrix.columns.tolist()

        for i, f1 in enumerate(factors):
            for f2 in factors[i + 1 :]:
                corr = abs(self._matrix.loc[f1, f2])
                if corr >= threshold:
                    pairs.append(CorrelationPair(f1, f2, float(corr)))

        return pairs

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return self._matrix.copy()


class CorrelationAnalyzer:
    """Analyzer for factor correlations."""

    def calculate(self, factor_values: pd.DataFrame) -> CorrelationMatrix:
        """Calculate correlation matrix from factor values.

        Args:
            factor_values: DataFrame with factors as columns

        Returns:
            CorrelationMatrix
        """
        corr_matrix = factor_values.corr()
        return CorrelationMatrix(corr_matrix)


@dataclass
class SelectionConfig:
    """Configuration for factor selection."""

    min_factors: int = 1
    max_factors: int = 20
    max_correlation: float = 0.7
    factor_values: Optional[pd.DataFrame] = None


@dataclass
class SelectionResult:
    """Result of factor selection."""

    selected: list[FactorEntry]
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected": [f.to_dict() for f in self.selected],
            "warnings": self.warnings,
            "is_valid": self.is_valid,
        }

    def get_names(self) -> list[str]:
        """Get names of selected factors."""
        return [f.name for f in self.selected]


class FactorSelector:
    """Selector for choosing factors from library."""

    def __init__(
        self,
        library: FactorLibrary,
        config: Optional[SelectionConfig] = None,
    ) -> None:
        """Initialize with library and config."""
        self.library = library
        self.config = config or SelectionConfig()
        self.correlation_analyzer = CorrelationAnalyzer()

    def select_top_n(
        self, n: int, sort_by: str = "ic"
    ) -> SelectionResult:
        """Select top N factors by metric.

        Args:
            n: Number of factors to select
            sort_by: Metric to sort by

        Returns:
            SelectionResult
        """
        sorted_factors = self.library.sort_by(sort_by, descending=True)
        selected = sorted_factors[:n]

        return self._create_result(selected)

    def select_by_family(
        self, one_per_family: bool = True
    ) -> SelectionResult:
        """Select factors by family.

        Args:
            one_per_family: If True, select only one per family

        Returns:
            SelectionResult
        """
        all_factors = self.library.sort_by("ic", descending=True)

        if one_per_family:
            selected = []
            seen_families: set[str] = set()

            for factor in all_factors:
                if factor.family not in seen_families:
                    selected.append(factor)
                    seen_families.add(factor.family)
        else:
            selected = all_factors

        return self._create_result(selected)

    def select_uncorrelated(self, n: int) -> SelectionResult:
        """Select uncorrelated factors.

        Args:
            n: Maximum number of factors

        Returns:
            SelectionResult
        """
        if self.config.factor_values is None:
            # Fall back to top N if no factor values
            return self.select_top_n(n)

        sorted_factors = self.library.sort_by("ic", descending=True)
        selected: list[FactorEntry] = []

        # Greedy selection avoiding high correlation
        for factor in sorted_factors:
            if len(selected) >= n:
                break

            # Check correlation with already selected
            is_correlated = False
            for selected_factor in selected:
                if self._are_correlated(factor.name, selected_factor.name):
                    is_correlated = True
                    break

            if not is_correlated:
                selected.append(factor)

        return self._create_result(selected)

    def select_all(self) -> SelectionResult:
        """Select all factors with warnings."""
        selected = self.library.get_all()
        return self._create_result(selected)

    def select_manual(self, names: list[str]) -> SelectionResult:
        """Manually select factors by name.

        Args:
            names: List of factor names to select

        Returns:
            SelectionResult

        Raises:
            InvalidSelectionError: If factor not found
        """
        selected = []

        for name in names:
            factor = self.library.get(name)
            if factor is None:
                raise InvalidSelectionError(f"Factor not found: {name}")
            selected.append(factor)

        return self._create_result(selected)

    def _create_result(
        self, selected: list[FactorEntry]
    ) -> SelectionResult:
        """Create result with validation and warnings."""
        warnings = []

        # Check for high correlations
        if self.config.factor_values is not None and len(selected) > 1:
            warnings.extend(self._check_correlations(selected))

        # Validate selection
        is_valid = True
        if len(selected) < self.config.min_factors:
            is_valid = False
            warnings.append(
                f"Selection has {len(selected)} factors, "
                f"minimum is {self.config.min_factors}"
            )

        if len(selected) > self.config.max_factors:
            is_valid = False
            warnings.append(
                f"Selection has {len(selected)} factors, "
                f"maximum is {self.config.max_factors}"
            )

        return SelectionResult(
            selected=selected,
            warnings=warnings,
            is_valid=is_valid,
        )

    def _check_correlations(
        self, selected: list[FactorEntry]
    ) -> list[str]:
        """Check for high correlations among selected factors."""
        warnings = []

        if self.config.factor_values is None:
            return warnings

        names = [f.name for f in selected]
        available_cols = [
            n for n in names if n in self.config.factor_values.columns
        ]

        if len(available_cols) < 2:
            return warnings

        factor_df = self.config.factor_values[available_cols]
        matrix = self.correlation_analyzer.calculate(factor_df)

        high_pairs = matrix.find_high_correlations(self.config.max_correlation)
        for pair in high_pairs:
            warnings.append(
                f"High correlation ({pair.correlation:.2f}) between "
                f"{pair.factor1} and {pair.factor2}"
            )

        return warnings

    def _are_correlated(self, name1: str, name2: str) -> bool:
        """Check if two factors are highly correlated."""
        if self.config.factor_values is None:
            return False

        if name1 not in self.config.factor_values.columns:
            return False
        if name2 not in self.config.factor_values.columns:
            return False

        corr = self.config.factor_values[name1].corr(
            self.config.factor_values[name2]
        )

        return abs(corr) >= self.config.max_correlation
