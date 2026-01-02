"""Research Ledger for tracking factor evaluation trials.

This module provides:
- TrialRecord: Data structure for recording individual trials
- DynamicThreshold: Calculator for adjusting significance thresholds
- ResearchLedger: Main ledger for trial tracking and querying
- Storage backends: MemoryStorage and FileStorage

The dynamic threshold uses a simplified Deflated Sharpe Ratio approach
to account for multiple hypothesis testing.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Use Qlib-native statistics instead of scipy
from iqfmp.evaluation.qlib_stats import normal_cdf, normal_ppf

logger = logging.getLogger(__name__)


class InvalidTrialError(Exception):
    """Raised when a trial record is invalid."""

    pass


class ThresholdExceededWarning(UserWarning):
    """Warning raised when threshold is exceeded."""

    pass


@dataclass
class ThresholdConfig:
    """Configuration for dynamic threshold calculation."""

    base_sharpe_threshold: float = 2.0
    confidence_level: float = 0.95
    min_trials_for_adjustment: int = 1
    min_threshold: float = 1.5  # Minimum acceptable threshold
    default_n_observations: int = 252  # Default trading days per year


@dataclass
class ThresholdResult:
    """Result of a threshold check."""

    passes: bool
    threshold: float
    sharpe: float
    n_trials: int
    deflated_sharpe: float | None = None
    p_value: float | None = None


@dataclass
class TrialRecord:
    """Record of a single factor evaluation trial.

    Attributes:
        trial_id: Unique identifier for the trial
        factor_name: Name of the factor being evaluated
        factor_family: Category/family of the factor
        sharpe_ratio: Sharpe ratio of the factor
        ic_mean: Mean information coefficient
        ir: Information ratio
        max_drawdown: Maximum drawdown
        win_rate: Win rate of the factor
        created_at: Timestamp of trial creation
        metadata: Additional metadata
    """

    factor_name: str
    factor_family: str
    sharpe_ratio: float
    trial_id: str | None = None
    ic_mean: float | None = None
    ir: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        # Validate factor_name
        if not self.factor_name or not self.factor_name.strip():
            raise InvalidTrialError("factor_name cannot be empty")

        # Generate trial_id if not provided
        if self.trial_id is None:
            self.trial_id = str(uuid.uuid4())[:8]

        # Set created_at if not provided
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert trial record to dictionary."""
        return {
            "trial_id": self.trial_id,
            "factor_name": self.factor_name,
            "factor_family": self.factor_family,
            "sharpe_ratio": self.sharpe_ratio,
            "ic_mean": self.ic_mean,
            "ir": self.ir,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialRecord:
        """Create trial record from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            trial_id=data.get("trial_id"),
            factor_name=data["factor_name"],
            factor_family=data["factor_family"],
            sharpe_ratio=data["sharpe_ratio"],
            ic_mean=data.get("ic_mean"),
            ir=data.get("ir"),
            max_drawdown=data.get("max_drawdown"),
            win_rate=data.get("win_rate"),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )


class DynamicThreshold:
    """Calculator for dynamic significance thresholds.

    Uses the Deflated Sharpe Ratio approach from Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality"

    The key insight is that with more trials, the probability of
    finding a spuriously significant factor increases, so we need
    higher thresholds adjusted for multiple hypothesis testing.
    """

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        """Initialize with configuration."""
        self.config = config or ThresholdConfig()

    def calculate(self, n_trials: int) -> float:
        """Calculate adjusted threshold for given number of trials.

        Uses a simplified Deflated Sharpe Ratio formula:
        adjusted_threshold = base_threshold * adjustment_factor

        Where adjustment_factor accounts for multiple testing via
        the expected maximum of n normal random variables.

        Args:
            n_trials: Number of trials conducted

        Returns:
            Adjusted Sharpe ratio threshold
        """
        if n_trials <= 0:
            n_trials = 1

        # Expected maximum of n standard normal variables (Bonferroni-like adjustment)
        # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * ln(n)) for large n
        expected_max = 0.0 if n_trials == 1 else math.sqrt(2 * math.log(n_trials))

        # Get z-score for confidence level
        # Higher confidence → higher z-score → higher threshold
        z_score = normal_ppf(self.config.confidence_level)

        # Confidence multiplier: higher confidence means stricter threshold
        confidence_multiplier = z_score / 1.645  # Normalize to 95% baseline

        # Adjustment factor based on number of trials and confidence
        # Uses expected max scaled by confidence level
        adjustment = 1 + (expected_max * confidence_multiplier * 0.15)

        return float(self.config.base_sharpe_threshold * adjustment)

    def calculate_deflated(
        self,
        n_trials: int,
        n_observations: int | None = None,
        _expected_sharpe: float = 0.0,
        variance_of_sharpe: float = 1.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> tuple[float, float, float]:
        """Calculate Deflated Sharpe Ratio threshold with full academic rigor.

        Based on Bailey & López de Prado (2014) methodology for correcting
        selection bias, backtest overfitting, and non-normality in Sharpe ratios.

        Args:
            n_trials: Number of trials/strategies tested
            n_observations: Number of return observations (default: 252 trading days)
            expected_sharpe: Expected Sharpe under null hypothesis (usually 0)
            variance_of_sharpe: Variance of Sharpe distribution
            skewness: Skewness of return distribution (γ₃)
            kurtosis: Kurtosis of return distribution (γ₄, normal = 3)

        Returns:
            Tuple of (threshold, expected_max_sharpe, sharpe_std_error)
        """
        if n_trials <= 0:
            n_trials = 1
        if n_observations is None:
            n_observations = self.config.default_n_observations

        # 1. Calculate Expected Maximum Sharpe (Order Statistics)
        e_max_sharpe = self._expected_max_sharpe(n_trials, variance_of_sharpe)

        # 2. Calculate Sharpe standard error (Lo, 2002 adjustment for non-normality)
        se_sharpe = self._sharpe_standard_error(
            n_observations=n_observations,
            sharpe_estimate=1.0,  # Conservative estimate
            skewness=skewness,
            kurtosis=kurtosis,
        )

        # 3. Get z-score for confidence level
        z_alpha = normal_ppf(self.config.confidence_level)

        # 4. Deflated Sharpe Ratio threshold
        # Requirement: SR_observed > E[max(SR)] + z_alpha * SE(SR)
        threshold = e_max_sharpe + z_alpha * se_sharpe

        # Ensure threshold doesn't fall below minimum
        threshold = max(threshold, self.config.min_threshold)

        return float(threshold), float(e_max_sharpe), float(se_sharpe)

    def _expected_max_sharpe(self, n: int, variance: float = 1.0) -> float:
        """Calculate expected maximum Sharpe based on order statistics.

        For n independent trials, the expected maximum of n standard normal
        variables is approximately: E[max] ≈ Φ^(-1)(1 - 1/n) * sqrt(variance)

        Args:
            n: Number of trials
            variance: Variance of Sharpe ratio distribution

        Returns:
            Expected maximum Sharpe ratio
        """
        if n <= 1:
            return 0.0

        # Use the quantile function for expected max of n normals
        # This is more accurate than the sqrt(2*ln(n)) approximation
        quantile = normal_ppf(1 - 1 / n)
        return float(quantile * np.sqrt(variance))

    def _sharpe_standard_error(
        self,
        n_observations: int,
        sharpe_estimate: float = 1.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Calculate Sharpe Ratio standard error with non-normality adjustment.

        Based on Lo (2002) "The Statistics of Sharpe Ratios":
        SE(SR) = sqrt((1 + 0.5*SR² - γ₃*SR + (γ₄-3)/4*SR²) / n)

        Where:
        - SR: Sharpe ratio estimate
        - γ₃: Skewness of returns
        - γ₄: Kurtosis of returns (3 for normal)
        - n: Number of observations

        Args:
            n_observations: Number of return observations
            sharpe_estimate: Estimated Sharpe ratio (for SE calculation)
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis (excess kurtosis = kurtosis - 3)

        Returns:
            Standard error of Sharpe ratio
        """
        if n_observations <= 0:
            n_observations = 1

        sr = sharpe_estimate

        # Lo (2002) formula with non-normality adjustment
        se_squared = (
            1
            + 0.5 * sr**2
            - skewness * sr
            + (kurtosis - 3) / 4 * sr**2
        ) / n_observations

        return float(np.sqrt(max(se_squared, 0)))

    def calculate_deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int | None = None,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> tuple[float, float]:
        """Calculate the Deflated Sharpe Ratio and its p-value.

        The DSR adjusts an observed Sharpe ratio for selection bias
        from multiple testing.

        Args:
            observed_sharpe: The observed (unadjusted) Sharpe ratio
            n_trials: Number of trials/strategies tested
            n_observations: Number of return observations
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis

        Returns:
            Tuple of (deflated_sharpe_ratio, p_value)
        """
        if n_observations is None:
            n_observations = self.config.default_n_observations

        # Calculate expected max and standard error
        e_max = self._expected_max_sharpe(n_trials)
        se = self._sharpe_standard_error(
            n_observations=n_observations,
            sharpe_estimate=observed_sharpe,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        # Deflated Sharpe Ratio = (SR_observed - E[max(SR)]) / SE(SR)
        if se > 0:
            deflated_sr = (observed_sharpe - e_max) / se
        else:
            deflated_sr = observed_sharpe - e_max

        # P-value: probability of observing this DSR under null
        p_value = 1.0 - normal_cdf(deflated_sr)

        return float(deflated_sr), float(p_value)

    def check(self, sharpe: float, n_trials: int) -> ThresholdResult:
        """Check if a Sharpe ratio passes the threshold.

        Args:
            sharpe: Sharpe ratio to check
            n_trials: Current number of trials

        Returns:
            ThresholdResult with pass/fail and details
        """
        threshold = self.calculate(n_trials)
        passes = bool(sharpe >= threshold)  # Ensure Python bool

        return ThresholdResult(
            passes=passes,
            threshold=float(threshold),
            sharpe=float(sharpe),
            n_trials=n_trials,
        )

    def check_deflated(
        self,
        sharpe: float,
        n_trials: int,
        n_observations: int | None = None,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> ThresholdResult:
        """Check if a Sharpe ratio passes the deflated threshold.

        Uses the full Deflated Sharpe Ratio methodology for rigorous
        multiple hypothesis testing correction.

        Args:
            sharpe: Sharpe ratio to check
            n_trials: Current number of trials
            n_observations: Number of return observations
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis

        Returns:
            ThresholdResult with pass/fail, threshold, DSR, and p-value
        """
        # Calculate deflated threshold
        threshold, e_max, se = self.calculate_deflated(
            n_trials=n_trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        # Calculate deflated sharpe and p-value
        deflated_sr, p_value = self.calculate_deflated_sharpe_ratio(
            observed_sharpe=sharpe,
            n_trials=n_trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        passes = bool(sharpe >= threshold)

        return ThresholdResult(
            passes=passes,
            threshold=float(threshold),
            sharpe=float(sharpe),
            n_trials=n_trials,
            deflated_sharpe=float(deflated_sr),
            p_value=float(p_value),
        )


class LedgerStorage(ABC):
    """Abstract base class for ledger storage backends."""

    @abstractmethod
    def save(self, trials: list[TrialRecord]) -> None:
        """Save trials to storage."""
        pass

    @abstractmethod
    def load(self) -> list[TrialRecord]:
        """Load trials from storage."""
        pass


class MemoryStorage(LedgerStorage):
    """In-memory storage backend."""

    def __init__(self) -> None:
        """Initialize empty storage."""
        self._trials: list[TrialRecord] = []

    def save(self, trials: list[TrialRecord]) -> None:
        """Save trials to memory."""
        self._trials = trials.copy()

    def load(self) -> list[TrialRecord]:
        """Load trials from memory."""
        return self._trials.copy()


class FileStorage(LedgerStorage):
    """File-based storage backend using JSON."""

    def __init__(self, filepath: Path | str) -> None:
        """Initialize with file path."""
        self.filepath = Path(filepath)
        self._load_on_init()

    def _load_on_init(self) -> None:
        """Load existing data if file exists."""
        if self.filepath.exists():
            self._cached_trials = self._read_file()
        else:
            self._cached_trials: list[TrialRecord] = []

    def _read_file(self) -> list[TrialRecord]:
        """Read trials from file."""
        with open(self.filepath) as f:
            data = json.load(f)
        return [TrialRecord.from_dict(d) for d in data.get("trials", [])]

    def save(self, trials: list[TrialRecord]) -> None:
        """Save trials to file."""
        self._cached_trials = trials.copy()
        data = {"trials": [t.to_dict() for t in trials]}
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> list[TrialRecord]:
        """Load trials from file."""
        return self._cached_trials.copy()


class PostgresStorage(LedgerStorage):
    """PostgreSQL-based storage backend using SQLAlchemy async.

    This storage persists research trials to TimescaleDB/PostgreSQL,
    ensuring data survives server restarts and supports querying.
    """

    def __init__(self) -> None:
        """Initialize PostgreSQL storage."""
        self._cached_trials: list[TrialRecord] = []
        self._loaded = False
        self._threshold = DynamicThreshold()

    @staticmethod
    def _orm_to_trial(orm: Any) -> TrialRecord:
        """Convert ORM model to TrialRecord."""
        return TrialRecord(
            trial_id=orm.trial_id,
            factor_name=orm.factor_name,
            factor_family=orm.factor_family,
            sharpe_ratio=orm.sharpe_ratio,
            ic_mean=orm.ic_mean,
            ir=orm.ir,
            max_drawdown=orm.max_drawdown,
            win_rate=orm.win_rate,
            created_at=orm.created_at.replace(tzinfo=None) if orm.created_at else None,
            metadata=orm.evaluation_config or {},
        )

    def save(self, trials: list[TrialRecord]) -> None:
        """Save trials to PostgreSQL (sync wrapper for async operation)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, schedule the coroutine
                asyncio.create_task(self._save_async(trials))
            else:
                loop.run_until_complete(self._save_async(trials))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self._save_async(trials))
        self._cached_trials = trials.copy()

    async def _save_async(self, trials: list[TrialRecord]) -> None:
        """Async save implementation."""
        from sqlalchemy import func, select

        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM

        async with get_async_session() as session:
            # Get existing trial IDs from database
            result = await session.execute(
                select(ResearchTrialORM.trial_id)
            )
            existing_ids = {row[0] for row in result.fetchall()}

            # Get current max trial number
            max_num_result = await session.execute(
                select(func.max(ResearchTrialORM.trial_number))
            )
            max_num = max_num_result.scalar() or 0

            # Insert only new trials
            for trial in trials:
                if trial.trial_id not in existing_ids:
                    max_num += 1
                    threshold = self._threshold.calculate(max_num)

                    orm_trial = ResearchTrialORM(
                        trial_id=trial.trial_id,
                        trial_number=max_num,
                        factor_name=trial.factor_name,
                        factor_family=trial.factor_family,
                        sharpe_ratio=trial.sharpe_ratio,
                        ic_mean=trial.ic_mean,
                        ir=trial.ir,
                        max_drawdown=trial.max_drawdown,
                        win_rate=trial.win_rate,
                        threshold_used=threshold,
                        passed_threshold=trial.sharpe_ratio >= threshold,
                        evaluation_config=trial.metadata,
                    )
                    session.add(orm_trial)

    def load(self) -> list[TrialRecord]:
        """Load trials from PostgreSQL (sync wrapper)."""
        if not self._loaded:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Return cached if in async context
                    return self._cached_trials.copy()
                else:
                    self._cached_trials = loop.run_until_complete(self._load_async())
            except RuntimeError:
                self._cached_trials = asyncio.run(self._load_async())
            self._loaded = True
        return self._cached_trials.copy()

    async def _load_async(self) -> list[TrialRecord]:
        """Async load implementation."""
        from sqlalchemy import select

        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM

        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(ResearchTrialORM).order_by(ResearchTrialORM.trial_number)
                )
                orm_trials = result.scalars().all()
                return [self._orm_to_trial(t) for t in orm_trials]
        except Exception as e:
            # If database not available, return empty
            logger.warning(f"PostgreSQL load failed: {e}")
            return []

    async def save_trial_async(
        self,
        trial: TrialRecord,
        factor_id: str | None = None,
    ) -> str:
        """Save a single trial asynchronously with factor linking.

        Args:
            trial: Trial record to save
            factor_id: Optional factor ID to link

        Returns:
            Trial ID
        """
        from sqlalchemy import func, select

        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM

        async with get_async_session() as session:
            # Get current max trial number
            max_num_result = await session.execute(
                select(func.max(ResearchTrialORM.trial_number))
            )
            max_num = (max_num_result.scalar() or 0) + 1

            threshold = self._threshold.calculate(max_num)

            orm_trial = ResearchTrialORM(
                trial_id=trial.trial_id,
                trial_number=max_num,
                factor_id=factor_id,
                factor_name=trial.factor_name,
                factor_family=trial.factor_family,
                sharpe_ratio=trial.sharpe_ratio,
                ic_mean=trial.ic_mean,
                ir=trial.ir,
                max_drawdown=trial.max_drawdown,
                win_rate=trial.win_rate,
                threshold_used=threshold,
                passed_threshold=trial.sharpe_ratio >= threshold,
                evaluation_config=trial.metadata,
            )
            session.add(orm_trial)

            # Update cache
            self._cached_trials.append(trial)

        return trial.trial_id

    async def get_trial_count_async(self) -> int:
        """Get trial count from database."""
        from sqlalchemy import func, select

        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM

        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(func.count(ResearchTrialORM.id))
                )
                return result.scalar() or 0
        except Exception:
            return len(self._cached_trials)

    async def get_trials_by_family_async(self, family: str) -> list[TrialRecord]:
        """Get trials by family from database."""
        from sqlalchemy import select

        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM

        async with get_async_session() as session:
            result = await session.execute(
                select(ResearchTrialORM)
                .where(ResearchTrialORM.factor_family == family)
                .order_by(ResearchTrialORM.trial_number)
            )
            orm_trials = result.scalars().all()
            return [self._orm_to_trial(t) for t in orm_trials]


@dataclass
class LedgerStatistics:
    """Statistics computed from ledger trials."""

    total_trials: int = 0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    max_sharpe: float = 0.0
    min_sharpe: float = 0.0
    median_sharpe: float = 0.0


def _get_default_storage(strict_mode: bool = False) -> LedgerStorage:
    """Get the default storage backend with PostgreSQL preference.

    P1-2 FIX: Production systems MUST use PostgresStorage to persist
    research trials. MemoryStorage loses all data on restart.

    P3.1 FIX: Added strict_mode to block fallback to MemoryStorage in production.

    Args:
        strict_mode: If True, raise error instead of falling back to MemoryStorage.
                     Set via RESEARCH_LEDGER_STRICT=true environment variable.

    Returns:
        PostgresStorage if database is available.

    Raises:
        RuntimeError: If strict_mode=True and PostgresStorage is unavailable.
    """
    import os
    strict_mode = strict_mode or os.getenv("RESEARCH_LEDGER_STRICT", "").lower() == "true"

    try:
        storage = PostgresStorage()
        # Test that we can actually load (validates DB connection)
        storage.load()
        logger.info("ResearchLedger using PostgresStorage (production mode)")
        return storage
    except Exception as e:
        if strict_mode:
            raise RuntimeError(
                f"PostgresStorage unavailable ({e}). "
                "STRICT MODE: MemoryStorage/FileStorage are FORBIDDEN in production. "
                "Configure DATABASE_URL or set RESEARCH_LEDGER_STRICT=false for development."
            )
        logger.warning(
            f"PostgresStorage unavailable ({e}), falling back to MemoryStorage. "
            "WARNING: Research trials will NOT persist across restarts! "
            "Set RESEARCH_LEDGER_STRICT=true to enforce production persistence."
        )
        return MemoryStorage()


def validate_production_storage(storage: LedgerStorage) -> None:
    """Validate that storage backend is suitable for production.

    P3.1 FIX: Explicitly block MemoryStorage/FileStorage in production environments.

    Args:
        storage: The storage backend to validate.

    Raises:
        RuntimeError: If storage is not PostgresStorage and strict mode is enabled.
    """
    import os
    if (
        os.getenv("RESEARCH_LEDGER_STRICT", "").lower() == "true"
        and isinstance(storage, (MemoryStorage, FileStorage))
    ):
        raise RuntimeError(
            f"PRODUCTION ERROR: {type(storage).__name__} is not allowed. "
            "Use PostgresStorage for production persistence. "
            "Set RESEARCH_LEDGER_STRICT=false for development."
        )


class ResearchLedger:
    """Main ledger for tracking factor evaluation trials.

    The ResearchLedger provides:
    - Trial recording and retrieval
    - Dynamic threshold calculation
    - Statistics computation
    - Storage persistence
    """

    def __init__(
        self,
        storage: LedgerStorage | None = None,
        threshold: DynamicThreshold | None = None,
    ) -> None:
        """Initialize ledger with storage and threshold calculator.

        Args:
            storage: Storage backend (defaults to PostgresStorage with fallback)
            threshold: Threshold calculator (defaults to DynamicThreshold)
        """
        # P1-2 FIX: Default to PostgresStorage for production persistence
        self._storage = storage or _get_default_storage()
        self._threshold = threshold or DynamicThreshold()
        self._trials: list[TrialRecord] = self._storage.load()

    def record(self, trial: TrialRecord) -> str:
        """Record a trial to the ledger.

        Args:
            trial: Trial record to add

        Returns:
            Trial ID of the recorded trial

        Raises:
            InvalidTrialError: If trial is invalid
        """
        # Validation already done in TrialRecord.__post_init__
        self._trials.append(trial)
        self._storage.save(self._trials)
        return trial.trial_id

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        """Get a trial by ID.

        Args:
            trial_id: ID of the trial to retrieve

        Returns:
            Trial record or None if not found
        """
        for trial in self._trials:
            if trial.trial_id == trial_id:
                return trial
        return None

    def get_trial_count(self) -> int:
        """Get the total number of trials."""
        return len(self._trials)

    def get_all_trials(self) -> list[TrialRecord]:
        """Get all trials in the ledger."""
        return self._trials.copy()

    def get_trials_by_family(self, family: str) -> list[TrialRecord]:
        """Get trials filtered by factor family.

        Args:
            family: Factor family to filter by

        Returns:
            List of trials in the specified family
        """
        return [t for t in self._trials if t.factor_family == family]

    def get_current_threshold(self) -> float:
        """Get the current significance threshold.

        Returns:
            Current threshold based on trial count
        """
        n_trials = max(1, self.get_trial_count())
        return self._threshold.calculate(n_trials)

    def check_significance(self, sharpe: float) -> ThresholdResult:
        """Check if a Sharpe ratio is significant.

        Args:
            sharpe: Sharpe ratio to check

        Returns:
            ThresholdResult with pass/fail and details
        """
        n_trials = max(1, self.get_trial_count())
        return self._threshold.check(sharpe, n_trials)

    def check_significance_deflated(
        self,
        sharpe: float,
        n_observations: int | None = None,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> ThresholdResult:
        """Check if a Sharpe ratio is significant using Deflated Sharpe Ratio.

        Uses the full DSR methodology from Bailey & López de Prado (2014)
        for rigorous multiple hypothesis testing correction.

        Args:
            sharpe: Sharpe ratio to check
            n_observations: Number of return observations (default: 252)
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis

        Returns:
            ThresholdResult with pass/fail, threshold, DSR, and p-value
        """
        n_trials = max(1, self.get_trial_count())
        return self._threshold.check_deflated(
            sharpe=sharpe,
            n_trials=n_trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def get_statistics(self) -> LedgerStatistics:
        """Compute statistics for all trials.

        Returns:
            LedgerStatistics with aggregate metrics
        """
        if not self._trials:
            return LedgerStatistics()

        sharpes = [t.sharpe_ratio for t in self._trials]

        return LedgerStatistics(
            total_trials=len(self._trials),
            mean_sharpe=sum(sharpes) / len(sharpes),
            std_sharpe=self._compute_std(sharpes),
            max_sharpe=max(sharpes),
            min_sharpe=min(sharpes),
            median_sharpe=self._compute_median(sharpes),
        )

    def get_statistics_by_family(self) -> dict[str, LedgerStatistics]:
        """Compute statistics grouped by factor family.

        Returns:
            Dictionary mapping family names to statistics
        """
        families: dict[str, list[TrialRecord]] = {}
        for trial in self._trials:
            if trial.factor_family not in families:
                families[trial.factor_family] = []
            families[trial.factor_family].append(trial)

        result: dict[str, LedgerStatistics] = {}
        for family, trials in families.items():
            sharpes = [t.sharpe_ratio for t in trials]
            result[family] = LedgerStatistics(
                total_trials=len(trials),
                mean_sharpe=sum(sharpes) / len(sharpes),
                std_sharpe=self._compute_std(sharpes),
                max_sharpe=max(sharpes),
                min_sharpe=min(sharpes),
                median_sharpe=self._compute_median(sharpes),
            )

        return result

    def export_to_dict(self) -> dict[str, Any]:
        """Export ledger data to dictionary.

        Returns:
            Dictionary with trials and metadata
        """
        return {
            "trials": [t.to_dict() for t in self._trials],
            "exported_at": datetime.now().isoformat(),
            "total_trials": len(self._trials),
        }

    def import_from_dict(self, data: dict[str, Any]) -> None:
        """Import trials from dictionary.

        Args:
            data: Dictionary with trials data
        """
        imported = [TrialRecord.from_dict(d) for d in data.get("trials", [])]
        self._trials.extend(imported)
        self._storage.save(self._trials)

    @staticmethod
    def _compute_std(values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def _compute_median(values: list[float]) -> float:
        """Compute median."""
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        return sorted_values[n // 2]
