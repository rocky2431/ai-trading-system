"""Research service for business logic.

This module provides PostgreSQL-backed research ledger management,
ensuring trial data persists across server restarts.
"""

import os
from datetime import datetime
from typing import Any, Optional

from iqfmp.evaluation.research_ledger import (
    DynamicThreshold,
    LedgerStatistics,
    MemoryStorage,
    PostgresStorage,
    ResearchLedger,
    ThresholdConfig,
    TrialRecord,
)


class ResearchService:
    """Service for research ledger management with PostgreSQL persistence."""

    def __init__(self, use_postgres: bool = True) -> None:
        """Initialize research service.

        P4 ARCHITECTURE FIX: Default to PostgreSQL (production-grade).
        Respects RESEARCH_LEDGER_STRICT env var for strict mode enforcement.

        Args:
            use_postgres: Whether to use PostgreSQL storage (default True).
                         In strict mode, raises error if DB is unavailable.
        """
        self._use_postgres = use_postgres
        self._postgres_storage: Optional[PostgresStorage] = None
        strict_mode = os.getenv("RESEARCH_LEDGER_STRICT", "").lower() == "true"

        # P4 FIX: Production-grade persistence - PostgreSQL required
        if use_postgres:
            try:
                self._postgres_storage = PostgresStorage()
                self._ledger = ResearchLedger(storage=self._postgres_storage)
            except Exception as e:
                if strict_mode:
                    raise RuntimeError(
                        f"P4 STRICT MODE: PostgresStorage required but initialization failed: {e}. "
                        "Set RESEARCH_LEDGER_STRICT=false to allow MemoryStorage fallback."
                    )
                print(f"Warning: PostgreSQL unavailable, using memory storage: {e}")
                self._ledger = ResearchLedger(storage=MemoryStorage())
        else:
            # Non-postgres mode still respects strict mode
            if strict_mode:
                raise RuntimeError(
                    "P4 STRICT MODE: use_postgres=False not allowed in strict mode. "
                    "PostgreSQL is required for production-grade persistence."
                )
            self._ledger = ResearchLedger(storage=MemoryStorage())

        self._threshold_config = ThresholdConfig()

    def add_trial(
        self,
        factor_name: str,
        factor_family: str,
        sharpe_ratio: float,
        ic_mean: Optional[float] = None,
        ir: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a trial to the ledger.

        Args:
            factor_name: Name of the factor
            factor_family: Factor family category
            sharpe_ratio: Sharpe ratio of the factor
            ic_mean: Mean information coefficient
            ir: Information ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate
            metadata: Additional metadata

        Returns:
            Trial ID
        """
        trial = TrialRecord(
            factor_name=factor_name,
            factor_family=factor_family,
            sharpe_ratio=sharpe_ratio,
            ic_mean=ic_mean,
            ir=ir,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            metadata=metadata or {},
        )
        return self._ledger.record(trial)

    async def add_trial_async(
        self,
        factor_name: str,
        factor_family: str,
        sharpe_ratio: float,
        factor_id: Optional[str] = None,
        ic_mean: Optional[float] = None,
        ir: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a trial to the ledger asynchronously with factor linking.

        This method directly saves to PostgreSQL with proper factor_id linking.

        Args:
            factor_name: Name of the factor
            factor_family: Factor family category
            sharpe_ratio: Sharpe ratio of the factor
            factor_id: Optional factor ID for database linking
            ic_mean: Mean information coefficient
            ir: Information ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate
            metadata: Additional metadata

        Returns:
            Trial ID
        """
        trial = TrialRecord(
            factor_name=factor_name,
            factor_family=factor_family,
            sharpe_ratio=sharpe_ratio,
            ic_mean=ic_mean,
            ir=ir,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            metadata=metadata or {},
        )

        # Use PostgresStorage async method if available
        if self._postgres_storage is not None:
            return await self._postgres_storage.save_trial_async(trial, factor_id=factor_id)
        else:
            # Fall back to sync method
            return self._ledger.record(trial)

    def list_trials(
        self,
        page: int = 1,
        page_size: int = 10,
        family: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[list[TrialRecord], int]:
        """List trials with pagination and filtering.

        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            family: Filter by family
            min_sharpe: Filter by minimum Sharpe ratio
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Tuple of (trials, total_count)
        """
        trials = self._ledger.get_all_trials()

        # Apply filters
        if family:
            trials = [t for t in trials if t.factor_family == family]

        if min_sharpe is not None:
            trials = [t for t in trials if t.sharpe_ratio >= min_sharpe]

        if start_date is not None:
            trials = [
                t
                for t in trials
                if t.created_at is not None and t.created_at >= start_date
            ]

        if end_date is not None:
            trials = [
                t
                for t in trials
                if t.created_at is not None and t.created_at <= end_date
            ]

        total = len(trials)

        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        trials = trials[start:end]

        return trials, total

    def get_statistics(self) -> LedgerStatistics:
        """Get overall ledger statistics.

        Returns:
            LedgerStatistics with aggregate metrics
        """
        return self._ledger.get_statistics()

    def get_statistics_by_family(self) -> dict[str, LedgerStatistics]:
        """Get statistics grouped by family.

        Returns:
            Dictionary mapping family names to statistics
        """
        return self._ledger.get_statistics_by_family()

    def get_current_threshold(self) -> float:
        """Get current significance threshold.

        Returns:
            Current threshold based on trial count
        """
        return self._ledger.get_current_threshold()

    def get_threshold_info(self) -> dict[str, Any]:
        """Get full threshold information.

        Returns:
            Dictionary with threshold details
        """
        n_trials = max(1, self._ledger.get_trial_count())
        current_threshold = self._ledger.get_current_threshold()

        # Generate threshold history at key points
        threshold_calc = DynamicThreshold(self._threshold_config)
        history_points = [1, 10, 25, 50, 100, 250, 500, 1000]
        history = []
        for n in history_points:
            if n <= n_trials or n <= 10:
                history.append(
                    {
                        "n_trials": n,
                        "threshold": threshold_calc.calculate(n),
                    }
                )

        return {
            "current_threshold": current_threshold,
            "n_trials": n_trials,
            "config": {
                "base_sharpe_threshold": self._threshold_config.base_sharpe_threshold,
                "confidence_level": self._threshold_config.confidence_level,
                "min_trials_for_adjustment": self._threshold_config.min_trials_for_adjustment,
            },
            "threshold_history": history,
        }


# Singleton instance
_research_service = ResearchService(
    use_postgres=os.environ.get("IQFMP_RESEARCH_USE_POSTGRES", "").lower()
    in {"1", "true", "yes"}
)


def get_research_service() -> ResearchService:
    """Get research service instance."""
    return _research_service
