"""Research service for business logic."""

from datetime import datetime
from typing import Any, Optional

from iqfmp.evaluation.research_ledger import (
    DynamicThreshold,
    LedgerStatistics,
    ResearchLedger,
    ThresholdConfig,
    TrialRecord,
)


class ResearchService:
    """Service for research ledger management."""

    def __init__(self) -> None:
        """Initialize research service."""
        self._ledger = ResearchLedger()
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
_research_service = ResearchService()


def get_research_service() -> ResearchService:
    """Get research service instance."""
    return _research_service
