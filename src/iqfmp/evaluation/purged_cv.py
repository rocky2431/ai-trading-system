"""Purged K-Fold Cross-Validation with Embargo Period.

Implements Lopez de Prado's Purged K-Fold CV to prevent data leakage
in time-series data by:
1. Purging: Removing samples between train/test that could leak information
2. Embargo: Adding buffer period after train to prevent leakage

Reference: Advances in Financial Machine Learning, Chapter 7
         Lopez de Prado, M. (2018)

This is a P1 fix for IQFMP - addressing the lack of proper time-series
cross-validation that could lead to overfitting.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


@dataclass
class PurgedCVConfig:
    """Configuration for Purged K-Fold CV.

    Attributes:
        n_splits: Number of folds (default: 5)
        purge_gap: Number of periods to purge around test boundaries (default: 5)
        embargo_pct: Percentage of data to embargo after training (default: 0.01 = 1%)
        min_train_size: Minimum training set size as fraction (default: 0.5 = 50%)
    """
    n_splits: int = 5
    purge_gap: int = 5
    embargo_pct: float = 0.01
    min_train_size: float = 0.5


@dataclass
class PurgedSplit:
    """Single split result from Purged K-Fold CV.

    Attributes:
        train_idx: Indices for training set
        test_idx: Indices for test set
        purge_start: Start index of purge zone
        purge_end: End index of purge zone
        embargo_end: End index of embargo period
        fold_number: Current fold number (0-indexed)
    """
    train_idx: np.ndarray
    test_idx: np.ndarray
    purge_start: int
    purge_end: int
    embargo_end: int
    fold_number: int = 0

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.test_idx)

    @property
    def n_purged(self) -> int:
        """Number of purged samples."""
        return self.purge_end - self.purge_start


class PurgedKFoldCV:
    """Purged K-Fold Cross-Validation for time-series data.

    Ensures no information leakage by:
    1. Sorting data by time
    2. Purging samples within purge_gap of test boundaries
    3. Adding embargo period after training ends

    This prevents:
    - Look-ahead bias: Training on future information
    - Information leakage: Using overlapping windows
    - Overfitting: Multiple testing on similar data

    Example:
        >>> cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5, purge_gap=10))
        >>> for split in cv.split(X, time_index):
        ...     X_train, X_test = X[split.train_idx], X[split.test_idx]
        ...     model.fit(X_train)
        ...     scores.append(model.score(X_test))

    Reference:
        Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
        Chapter 7: Cross-Validation in Finance.
    """

    def __init__(self, config: Optional[PurgedCVConfig] = None):
        """Initialize Purged K-Fold CV.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or PurgedCVConfig()

        if self.config.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.config.purge_gap < 0:
            raise ValueError("purge_gap must be non-negative")
        if not 0 <= self.config.embargo_pct < 0.5:
            raise ValueError("embargo_pct must be between 0 and 0.5")

    def split(
        self,
        X: pd.DataFrame,
        t: Optional[pd.Series] = None,
    ) -> Iterator[PurgedSplit]:
        """Generate purged train/test splits.

        Args:
            X: Feature matrix (DataFrame with time index or index column)
            t: Time series index. If None, uses X.index

        Yields:
            PurgedSplit for each fold

        Raises:
            ValueError: If data is too small for configuration
        """
        n = len(X)
        n_splits = self.config.n_splits
        purge_gap = self.config.purge_gap
        embargo_n = int(n * self.config.embargo_pct)
        min_train_n = int(n * self.config.min_train_size)

        # Get time index
        if t is None:
            if isinstance(X.index, pd.DatetimeIndex):
                t = X.index
            elif "datetime" in X.columns:
                t = X["datetime"]
            elif "timestamp" in X.columns:
                t = X["timestamp"]
            else:
                # Use row order as time
                t = pd.Series(range(n))

        # Sort indices by time
        sorted_idx = np.argsort(t.values)

        # Calculate test fold size
        test_size = n // n_splits

        if test_size < 2:
            raise ValueError(f"Data too small for {n_splits} splits")

        # Generate folds
        for fold in range(n_splits):
            # Test set boundaries
            test_start = fold * test_size
            test_end = (fold + 1) * test_size if fold < n_splits - 1 else n

            # Test indices
            test_idx = sorted_idx[test_start:test_end]

            # Purge zone (around test set)
            purge_start = max(0, test_start - purge_gap)
            purge_end = min(n, test_end + purge_gap)

            # Train set: everything before purge zone
            train_end_raw = purge_start

            # Apply embargo (after train)
            if embargo_n > 0:
                train_end = max(0, train_end_raw - embargo_n)
            else:
                train_end = train_end_raw

            # Get training indices (before purge zone with embargo)
            train_idx = sorted_idx[:train_end].copy()

            # Also include samples after purge zone (if any)
            if purge_end < n:
                after_purge_idx = sorted_idx[purge_end:]
                train_idx = np.concatenate([train_idx, after_purge_idx])

            # Skip if training set too small
            if len(train_idx) < min_train_n:
                logger.warning(
                    f"Fold {fold}: Training set too small ({len(train_idx)} < {min_train_n}), skipping"
                )
                continue

            yield PurgedSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                purge_start=purge_start,
                purge_end=purge_end,
                embargo_end=train_end + embargo_n if embargo_n > 0 else train_end,
                fold_number=fold,
            )

    def get_n_splits(self) -> int:
        """Get number of splits (folds).

        Note: Actual number may be less due to data size constraints.
        """
        return self.config.n_splits


class TimeSeriesPurgedCV:
    """Time-series specific purged cross-validation.

    Extends PurgedKFoldCV with time-series specific features:
    - Forward-only splits (no future data in training)
    - Expanding or rolling window support
    - Gap-aware splitting for panel data
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_pct: float = 0.01,
        expanding: bool = True,
    ):
        """Initialize time-series purged CV.

        Args:
            n_splits: Number of test folds
            purge_gap: Gap between train and test
            embargo_pct: Embargo period percentage
            expanding: If True, use expanding window; if False, use rolling
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.expanding = expanding

    def split(
        self,
        X: pd.DataFrame,
        t: Optional[pd.Series] = None,
    ) -> Iterator[PurgedSplit]:
        """Generate forward-only purged splits.

        In forward-only mode, training never includes data from after the test period.
        This is more realistic for production scenarios.

        Args:
            X: Feature matrix
            t: Time index

        Yields:
            PurgedSplit for each fold
        """
        n = len(X)
        embargo_n = int(n * self.embargo_pct)

        # Get time index
        if t is None:
            if isinstance(X.index, pd.DatetimeIndex):
                t = X.index
            else:
                t = pd.Series(range(n))

        # Sort by time
        sorted_idx = np.argsort(t.values)

        # Calculate test fold size
        test_size = n // (self.n_splits + 1)

        # Reserve first portion for initial training
        initial_train_size = test_size

        for fold in range(self.n_splits):
            # Test set boundaries (move forward each fold)
            test_start = initial_train_size + fold * test_size
            test_end = min(test_start + test_size, n)

            if test_end <= test_start:
                continue

            test_idx = sorted_idx[test_start:test_end]

            # Training: everything before test_start with purge and embargo
            train_end_raw = max(0, test_start - self.purge_gap)
            train_end = max(0, train_end_raw - embargo_n)

            if self.expanding:
                # Expanding window: use all data up to train_end
                train_idx = sorted_idx[:train_end]
            else:
                # Rolling window: use fixed window size
                window_size = initial_train_size
                train_start = max(0, train_end - window_size)
                train_idx = sorted_idx[train_start:train_end]

            if len(train_idx) == 0:
                continue

            yield PurgedSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                purge_start=train_end,
                purge_end=test_start,
                embargo_end=train_end + embargo_n,
                fold_number=fold,
            )


def validate_cv_splits(
    cv: PurgedKFoldCV,
    X: pd.DataFrame,
    t: Optional[pd.Series] = None,
) -> dict:
    """Validate CV splits for data leakage.

    Args:
        cv: Cross-validator instance
        X: Feature matrix
        t: Time index

    Returns:
        Validation report with any issues found
    """
    issues = []
    splits_info = []

    for split in cv.split(X, t):
        # Check for index overlap
        train_set = set(split.train_idx)
        test_set = set(split.test_idx)
        overlap = train_set & test_set

        if overlap:
            issues.append(f"Fold {split.fold_number}: {len(overlap)} overlapping indices")

        # Check purge zone
        if split.purge_end <= split.purge_start:
            issues.append(f"Fold {split.fold_number}: Invalid purge zone")

        splits_info.append({
            "fold": split.fold_number,
            "n_train": split.n_train,
            "n_test": split.n_test,
            "n_purged": split.n_purged,
            "has_overlap": len(overlap) > 0,
        })

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "splits": splits_info,
        "n_splits_generated": len(splits_info),
    }
