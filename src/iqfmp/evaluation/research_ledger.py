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
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from scipy import stats


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


@dataclass
class ThresholdResult:
    """Result of a threshold check."""

    passes: bool
    threshold: float
    sharpe: float
    n_trials: int


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
    trial_id: Optional[str] = None
    ic_mean: Optional[float] = None
    ir: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    created_at: Optional[datetime] = None
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

    Uses a simplified Deflated Sharpe Ratio approach to adjust
    thresholds based on the number of trials conducted.

    The key insight is that with more trials, the probability of
    finding a spuriously significant factor increases, so we need
    higher thresholds.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
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
        if n_trials == 1:
            expected_max = 0.0
        else:
            expected_max = math.sqrt(2 * math.log(n_trials))

        # Get z-score for confidence level
        # Higher confidence → higher z-score → higher threshold
        z_score = stats.norm.ppf(self.config.confidence_level)

        # Confidence multiplier: higher confidence means stricter threshold
        confidence_multiplier = z_score / 1.645  # Normalize to 95% baseline

        # Adjustment factor based on number of trials and confidence
        # Uses expected max scaled by confidence level
        adjustment = 1 + (expected_max * confidence_multiplier * 0.15)

        return float(self.config.base_sharpe_threshold * adjustment)

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
        with open(self.filepath, "r") as f:
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
        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM
        from sqlalchemy import select, func

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
        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM
        from sqlalchemy import select

        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(ResearchTrialORM).order_by(ResearchTrialORM.trial_number)
                )
                orm_trials = result.scalars().all()

                return [
                    TrialRecord(
                        trial_id=t.trial_id,
                        factor_name=t.factor_name,
                        factor_family=t.factor_family,
                        sharpe_ratio=t.sharpe_ratio,
                        ic_mean=t.ic_mean,
                        ir=t.ir,
                        max_drawdown=t.max_drawdown,
                        win_rate=t.win_rate,
                        created_at=t.created_at.replace(tzinfo=None) if t.created_at else None,
                        metadata=t.evaluation_config or {},
                    )
                    for t in orm_trials
                ]
        except Exception as e:
            # If database not available, return empty
            print(f"Warning: PostgreSQL load failed: {e}")
            return []

    async def save_trial_async(
        self,
        trial: TrialRecord,
        factor_id: Optional[str] = None,
    ) -> str:
        """Save a single trial asynchronously with factor linking.

        Args:
            trial: Trial record to save
            factor_id: Optional factor ID to link

        Returns:
            Trial ID
        """
        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM
        from sqlalchemy import select, func

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
        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM
        from sqlalchemy import select, func

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
        from iqfmp.db.database import get_async_session
        from iqfmp.db.models import ResearchTrialORM
        from sqlalchemy import select

        async with get_async_session() as session:
            result = await session.execute(
                select(ResearchTrialORM)
                .where(ResearchTrialORM.factor_family == family)
                .order_by(ResearchTrialORM.trial_number)
            )
            orm_trials = result.scalars().all()

            return [
                TrialRecord(
                    trial_id=t.trial_id,
                    factor_name=t.factor_name,
                    factor_family=t.factor_family,
                    sharpe_ratio=t.sharpe_ratio,
                    ic_mean=t.ic_mean,
                    ir=t.ir,
                    max_drawdown=t.max_drawdown,
                    win_rate=t.win_rate,
                    created_at=t.created_at.replace(tzinfo=None) if t.created_at else None,
                    metadata=t.evaluation_config or {},
                )
                for t in orm_trials
            ]


@dataclass
class LedgerStatistics:
    """Statistics computed from ledger trials."""

    total_trials: int = 0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    max_sharpe: float = 0.0
    min_sharpe: float = 0.0
    median_sharpe: float = 0.0


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
        storage: Optional[LedgerStorage] = None,
        threshold: Optional[DynamicThreshold] = None,
    ) -> None:
        """Initialize ledger with storage and threshold calculator.

        Args:
            storage: Storage backend (defaults to MemoryStorage)
            threshold: Threshold calculator (defaults to DynamicThreshold)
        """
        self._storage = storage or MemoryStorage()
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

    def get_trial(self, trial_id: str) -> Optional[TrialRecord]:
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
