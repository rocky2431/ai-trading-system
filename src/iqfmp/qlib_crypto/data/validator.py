"""Data validation for cryptocurrency data.

This module provides validation utilities to ensure data quality
and consistency for cryptocurrency market data.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


class ValidationError(Exception):
    """Exception raised when data validation fails."""

    pass


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get a summary of validation results.

        Returns:
            Human-readable summary string.
        """
        if self.is_valid and not self.warnings:
            return "Validation passed with no issues."

        lines = []
        if self.is_valid:
            lines.append("Validation passed with warnings:")
        else:
            lines.append("Validation failed:")

        for error in self.errors:
            lines.append(f"  ERROR: {error}")
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")

        return "\n".join(lines)


class DataValidator:
    """Validator for cryptocurrency market data."""

    # Required columns for basic OHLCV data
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    # Datetime column names (various formats)
    DATETIME_COLUMNS = {"datetime", "timestamp", "date", "time", "open_time", "ts"}

    # Price columns that should be numeric
    PRICE_COLUMNS = {"open", "high", "low", "close", "mark_price", "index_price"}

    def __init__(self) -> None:
        """Initialize the validator."""
        pass

    def validate(
        self,
        data: pd.DataFrame,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate a dataframe of cryptocurrency data.

        Args:
            data: DataFrame to validate.
            strict: If True, treat warnings as errors.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check for empty dataframe
        if data.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)

        # Normalize column names
        columns = {col.lower() for col in data.columns}

        # Check for required columns
        missing = self.REQUIRED_COLUMNS - columns
        if missing:
            errors.append(f"Missing required columns: {missing}")

        # Check for datetime column
        datetime_col = self._find_datetime_column(data)
        if datetime_col is None:
            warnings.append("No datetime column found")

        # Check data types for price columns
        for col in self.PRICE_COLUMNS & columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column '{col}' should be numeric, got {data[col].dtype}")

        # Check price consistency (high >= low)
        if "high" in columns and "low" in columns:
            if (data["high"] < data["low"]).any():
                errors.append("Found rows where high < low (price inconsistency)")

        # Check for negative volumes
        if "volume" in columns:
            if (data["volume"] < 0).any():
                errors.append("Found negative volume values")
            if (data["volume"] == 0).all():
                warnings.append("All volume values are zero")

        # Check for NaN values
        nan_cols = [col for col in data.columns if data[col].isna().any()]
        if nan_cols:
            warnings.append(f"Found NaN values in columns: {nan_cols}")

        # Check funding rate range if present
        if "funding_rate" in columns:
            funding = data["funding_rate"]
            if (funding.abs() > 0.01).any():  # > 1% per period is unusual
                warnings.append(
                    "Funding rate values exceed 1% - please verify data"
                )

        # Determine validity
        is_valid = len(errors) == 0
        if strict and warnings:
            is_valid = False
            errors.extend(warnings)
            warnings = []

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def _find_datetime_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the datetime column in the dataframe.

        Args:
            data: DataFrame to search.

        Returns:
            Column name if found, None otherwise.
        """
        columns = {col.lower(): col for col in data.columns}
        for dt_col in self.DATETIME_COLUMNS:
            if dt_col in columns:
                return columns[dt_col]
        return None
