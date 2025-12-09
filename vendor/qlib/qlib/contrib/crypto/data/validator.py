"""Data Validator for Cryptocurrency Data.

Provides validation utilities for cryptocurrency market data.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when data validation fails."""

    pass


@dataclass
class ValidationResult:
    """Result of data validation.

    Attributes:
        is_valid: Whether the data passed validation
        errors: List of error messages
        warnings: List of warning messages
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Validator for cryptocurrency market data.

    Validates OHLCV data and crypto-specific fields for correctness
    and completeness.

    Example:
        >>> validator = DataValidator(required_columns=["open", "close"])
        >>> result = validator.validate(df)
        >>> if not result.is_valid:
        ...     print(result.errors)
    """

    def __init__(
        self,
        required_columns: Optional[list[str]] = None,
        check_ohlc_consistency: bool = True,
        max_missing_ratio: float = 0.1,
    ) -> None:
        """Initialize the validator.

        Args:
            required_columns: Columns that must be present
            check_ohlc_consistency: Whether to validate OHLC relationships
            max_missing_ratio: Maximum allowed ratio of missing values
        """
        self.required_columns = required_columns or ["close"]
        self.check_ohlc_consistency = check_ohlc_consistency
        self.max_missing_ratio = max_missing_ratio

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate cryptocurrency data.

        Performs the following checks:
        1. Required columns present
        2. Data types are numeric for price/volume columns
        3. OHLC consistency (High >= Open, Close, Low; Low <= Open, Close, High)
        4. No negative prices
        5. Missing value ratio within threshold

        Args:
            data: DataFrame to validate

        Returns:
            ValidationResult with validation status and messages
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check for empty DataFrame
        if data.empty:
            return ValidationResult(
                is_valid=False,
                errors=["DataFrame is empty"],
            )

        # Check required columns
        missing_cols = [
            col for col in self.required_columns
            if col.lower() not in [c.lower() for c in data.columns]
        ]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Get lowercase column mapping
        col_map = {c.lower(): c for c in data.columns}

        # Check data types for price/volume columns
        type_errors = False
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in col_map:
                actual_col = col_map[col]
                if not np.issubdtype(data[actual_col].dtype, np.number):
                    errors.append(f"Column '{col}' must be numeric")
                    type_errors = True

        # Check OHLC consistency (only if no type errors to avoid comparison issues)
        if self.check_ohlc_consistency and not type_errors:
            ohlc_errors = self._check_ohlc_consistency(data, col_map)
            errors.extend(ohlc_errors)

        # Check for negative prices (only if no type errors)
        if not type_errors:
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in col_map:
                    actual_col = col_map[col]
                    if (data[actual_col] < 0).any():
                        errors.append(f"Column '{col}' contains negative values")

        # Check missing value ratio
        missing_warnings = self._check_missing_values(data)
        warnings.extend(missing_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_ohlc_consistency(
        self, data: pd.DataFrame, col_map: dict[str, str]
    ) -> list[str]:
        """Check OHLC price consistency."""
        errors: list[str] = []

        has_ohlc = all(
            col in col_map for col in ["open", "high", "low", "close"]
        )
        if not has_ohlc:
            return errors

        o = data[col_map["open"]]
        h = data[col_map["high"]]
        l = data[col_map["low"]]
        c = data[col_map["close"]]

        # High should be >= Open and Close
        if (h < o).any() or (h < c).any():
            errors.append("OHLC consistency error: High < Open or High < Close")

        # Low should be <= Open and Close
        if (l > o).any() or (l > c).any():
            errors.append("OHLC consistency error: Low > Open or Low > Close")

        # High should be >= Low
        if (h < l).any():
            errors.append("OHLC consistency error: High < Low")

        return errors

    def _check_missing_values(self, data: pd.DataFrame) -> list[str]:
        """Check for excessive missing values."""
        warnings: list[str] = []

        for col in data.columns:
            missing_ratio = data[col].isna().sum() / len(data)
            if missing_ratio > self.max_missing_ratio:
                warnings.append(
                    f"Column '{col}' has {missing_ratio:.1%} missing values "
                    f"(threshold: {self.max_missing_ratio:.1%})"
                )

        return warnings
