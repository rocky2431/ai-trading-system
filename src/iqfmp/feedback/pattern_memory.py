"""Pattern memory for closed-loop factor mining.

This module implements the pattern storage and retrieval system that enables
the feedback loop to learn from past successes and failures.

Key concepts:
- Success patterns: Factors that passed evaluation, used as positive examples
- Failure patterns: Factors that failed, with classified reasons to avoid

The pattern memory uses both:
1. PostgreSQL (PatternRecordORM) for durable storage
2. Qdrant (via FactorVectorStore) for similarity-based retrieval
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeVar

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from iqfmp.feedback.structured_feedback import StructuredFeedback
    from iqfmp.vector.store import FactorVectorStore

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class PatternRecord:
    """A pattern record representing a success or failure.

    This is the in-memory representation of a pattern,
    independent of storage backend.
    """

    pattern_id: str
    pattern_type: Literal["success", "failure"]
    hypothesis: str
    factor_code: str
    factor_family: str
    metrics: dict[str, float]
    feedback: Optional[str] = None
    failure_reasons: list[str] = field(default_factory=list)
    trial_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = 0.0  # Similarity score from vector search

    def __post_init__(self) -> None:
        """Validate record fields."""
        if not self.factor_code or not self.factor_code.strip():
            raise ValueError("factor_code cannot be empty")
        if self.pattern_type not in ("success", "failure"):
            raise ValueError(
                f"pattern_type must be 'success' or 'failure', got '{self.pattern_type}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "hypothesis": self.hypothesis,
            "factor_code": self.factor_code,
            "factor_family": self.factor_family,
            "metrics": self.metrics,
            "feedback": self.feedback,
            "failure_reasons": self.failure_reasons,
            "trial_id": self.trial_id,
            "created_at": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else self.created_at
            ),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternRecord":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            hypothesis=data["hypothesis"],
            factor_code=data["factor_code"],
            factor_family=data["factor_family"],
            metrics=data.get("metrics", {}),
            feedback=data.get("feedback"),
            failure_reasons=data.get("failure_reasons", []),
            trial_id=data.get("trial_id"),
            created_at=created_at,
            score=data.get("score", 0.0),
        )


class PatternMemory:
    """Pattern memory for storing and retrieving success/failure patterns.

    Provides a dual-storage system:
    - PostgreSQL for durable storage and complex queries
    - Qdrant for fast similarity-based retrieval

    Usage:
        memory = PatternMemory(vector_store, session_factory)

        # Record outcomes
        memory.record_success(hypothesis, factor_code, family, metrics)
        memory.record_failure(hypothesis, factor_code, family, feedback)

        # Retrieve similar patterns
        successes = memory.retrieve_similar_successes(hypothesis, limit=3)
        failures = memory.retrieve_similar_failures(hypothesis, limit=3)
    """

    def __init__(
        self,
        vector_store: "FactorVectorStore",
        session_factory: Optional[Callable[[], "Session"]] = None,
        collection_name: str = "patterns",
    ):
        """Initialize pattern memory.

        Args:
            vector_store: FactorVectorStore instance for similarity search
            session_factory: Factory function that returns a SQLAlchemy session
            collection_name: Qdrant collection name for patterns
        """
        self.vector_store = vector_store
        self.session_factory = session_factory
        self.collection_name = collection_name

    def _execute_with_session(
        self,
        operation: Callable[["Session"], T],
        *,
        commit: bool = True,
        error_msg: str = "Database operation failed",
        suppress_error: bool = False,
    ) -> Optional[T]:
        """Execute operation with proper session management.

        Args:
            operation: Function that takes a session and returns a result
            commit: Whether to commit after operation (for writes)
            error_msg: Error message prefix for logging
            suppress_error: If True, log error and return None instead of raising

        Returns:
            Result of operation, or None if error suppressed
        """
        if self.session_factory is None:
            from iqfmp.db.database import sync_session

            try:
                with sync_session() as session:
                    result = operation(session)
                    # sync_session auto-commits on success
                    return result
            except Exception as e:
                logger.error(f"{error_msg}: {e}")
                if suppress_error:
                    return None
                raise

        # Fallback for DI/testing scenarios with manual management
        session = self.session_factory()
        try:
            result = operation(session)
            if commit:
                session.commit()
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"{error_msg}: {e}")
            if suppress_error:
                return None
            raise
        finally:
            session.close()

    def record_success(
        self,
        hypothesis: str,
        factor_code: str,
        factor_family: str,
        metrics: dict[str, float],
        trial_id: Optional[str] = None,
    ) -> str:
        """Record a successful factor pattern.

        Args:
            hypothesis: The research hypothesis
            factor_code: Generated factor code
            factor_family: Factor family name
            metrics: Evaluation metrics (ic, ir, sharpe, etc.)
            trial_id: Reference to evaluation trial

        Returns:
            Generated pattern ID
        """
        pattern_id = str(uuid.uuid4())

        # Store in vector database for similarity search
        self.vector_store.add_pattern(
            pattern_id=pattern_id,
            pattern_type="success",
            hypothesis=hypothesis,
            factor_code=factor_code,
            family=factor_family,
            metrics=metrics,
            feedback=None,
            failure_reasons=None,
            trial_id=trial_id,
            collection_name=self.collection_name,
        )

        # Store in PostgreSQL if session factory available
        if self.session_factory:
            self._persist_to_db(
                pattern_id=pattern_id,
                pattern_type="success",
                hypothesis=hypothesis,
                factor_code=factor_code,
                factor_family=factor_family,
                metrics=metrics,
                feedback=None,
                failure_reasons=None,
                trial_id=trial_id,
            )

        logger.info(f"Recorded success pattern: {pattern_id}")
        return pattern_id

    def record_failure(
        self,
        hypothesis: str,
        factor_code: str,
        factor_family: str,
        feedback: "StructuredFeedback",
    ) -> str:
        """Record a failed factor pattern.

        Args:
            hypothesis: The research hypothesis
            factor_code: Generated factor code
            factor_family: Factor family name
            feedback: StructuredFeedback from evaluation

        Returns:
            Generated pattern ID
        """
        pattern_id = str(uuid.uuid4())

        # Extract failure details
        failure_reasons = [r.value for r in feedback.failure_reasons]
        feedback_text = feedback.to_prompt_context()
        metrics = {
            "ic": feedback.ic,
            "ir": feedback.ir,
            "sharpe": feedback.sharpe,
            "max_drawdown": feedback.max_drawdown,
        }
        if feedback.win_rate is not None:
            metrics["win_rate"] = feedback.win_rate

        # Store in vector database
        self.vector_store.add_pattern(
            pattern_id=pattern_id,
            pattern_type="failure",
            hypothesis=hypothesis,
            factor_code=factor_code,
            family=factor_family,
            metrics=metrics,
            feedback=feedback_text,
            failure_reasons=failure_reasons,
            trial_id=feedback.trial_id,
            collection_name=self.collection_name,
        )

        # Store in PostgreSQL if session factory available
        if self.session_factory:
            self._persist_to_db(
                pattern_id=pattern_id,
                pattern_type="failure",
                hypothesis=hypothesis,
                factor_code=factor_code,
                factor_family=factor_family,
                metrics=metrics,
                feedback=feedback_text,
                failure_reasons=failure_reasons,
                trial_id=feedback.trial_id,
            )

        logger.info(f"Recorded failure pattern: {pattern_id}")
        return pattern_id

    def retrieve_similar_successes(
        self,
        hypothesis: str,
        family: Optional[str] = None,
        limit: int = 3,
    ) -> list[PatternRecord]:
        """Retrieve similar successful patterns.

        Args:
            hypothesis: Query hypothesis for similarity search
            family: Optional factor family filter
            limit: Maximum number of results

        Returns:
            List of similar success patterns
        """
        results = self.vector_store.search_patterns(
            hypothesis=hypothesis,
            pattern_type="success",
            family=family,
            limit=limit,
            collection_name=self.collection_name,
        )

        return [PatternRecord.from_dict(r) for r in results]

    def retrieve_similar_failures(
        self,
        hypothesis: str,
        family: Optional[str] = None,
        limit: int = 3,
    ) -> list[PatternRecord]:
        """Retrieve similar failed patterns.

        Args:
            hypothesis: Query hypothesis for similarity search
            family: Optional factor family filter
            limit: Maximum number of results

        Returns:
            List of similar failure patterns
        """
        results = self.vector_store.search_patterns(
            hypothesis=hypothesis,
            pattern_type="failure",
            family=family,
            limit=limit,
            collection_name=self.collection_name,
        )

        return [PatternRecord.from_dict(r) for r in results]

    def get_family_statistics(self, family: str) -> dict[str, Any]:
        """Get statistics for a factor family.

        Args:
            family: Factor family name

        Returns:
            Dictionary with family statistics
        """
        if not self.session_factory:
            return self._get_family_stats_from_vector_store(family)

        return self._get_family_stats_from_db(family)

    def _persist_to_db(
        self,
        pattern_id: str,
        pattern_type: str,
        hypothesis: str,
        factor_code: str,
        factor_family: str,
        metrics: dict[str, float],
        feedback: Optional[str],
        failure_reasons: Optional[list[str]],
        trial_id: Optional[str],
    ) -> None:
        """Persist pattern to PostgreSQL database."""
        from iqfmp.db.models import PatternRecordORM

        def _do_persist(session: "Session") -> None:
            record = PatternRecordORM(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                hypothesis=hypothesis,
                factor_code=factor_code,
                factor_family=factor_family,
                metrics=metrics,
                feedback=feedback,
                failure_reasons=failure_reasons,
                trial_id=trial_id,
            )
            session.add(record)

        self._execute_with_session(
            _do_persist,
            commit=True,
            error_msg=f"Failed to persist pattern {pattern_id} to DB",
        )

    def _get_family_stats_from_db(self, family: str) -> dict[str, Any]:
        """Get family statistics from PostgreSQL."""
        from sqlalchemy import func

        from iqfmp.db.models import PatternRecordORM

        def _query_stats(session: "Session") -> dict[str, Any]:
            # Count success and failure patterns
            success_count = (
                session.query(func.count(PatternRecordORM.id))
                .filter(
                    PatternRecordORM.factor_family == family,
                    PatternRecordORM.pattern_type == "success",
                )
                .scalar()
                or 0
            )

            failure_count = (
                session.query(func.count(PatternRecordORM.id))
                .filter(
                    PatternRecordORM.factor_family == family,
                    PatternRecordORM.pattern_type == "failure",
                )
                .scalar()
                or 0
            )

            total = success_count + failure_count
            success_rate = success_count / total if total > 0 else 0.0

            # Get common failure reasons
            failure_records = (
                session.query(PatternRecordORM.failure_reasons)
                .filter(
                    PatternRecordORM.factor_family == family,
                    PatternRecordORM.pattern_type == "failure",
                    PatternRecordORM.failure_reasons.isnot(None),
                )
                .limit(100)
                .all()
            )

            # Count failure reasons
            reason_counts: dict[str, int] = {}
            for (reasons,) in failure_records:
                if reasons:
                    for reason in reasons:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1

            common_failures = sorted(
                reason_counts.keys(), key=lambda x: reason_counts[x], reverse=True
            )[:3]

            # Get best IC from success patterns
            best_ic = 0.0
            success_metrics = (
                session.query(PatternRecordORM.metrics)
                .filter(
                    PatternRecordORM.factor_family == family,
                    PatternRecordORM.pattern_type == "success",
                )
                .all()
            )
            for (metrics,) in success_metrics:
                if metrics and "ic" in metrics:
                    best_ic = max(best_ic, metrics["ic"])

            return {
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_rate,
                "common_failures": common_failures,
                "best_ic": best_ic,
            }

        result = self._execute_with_session(
            _query_stats,
            commit=False,
            error_msg=f"Failed to get family stats for {family}",
        )
        return result or {
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0.0,
            "common_failures": [],
            "best_ic": 0.0,
        }

    def _get_family_stats_from_vector_store(self, family: str) -> dict[str, Any]:
        """Get family statistics from vector store (fallback)."""
        # This is a simplified version when DB is not available
        successes = self.vector_store.search_patterns(
            hypothesis="",  # Empty query to get all
            pattern_type="success",
            family=family,
            limit=100,
            collection_name=self.collection_name,
        )

        failures = self.vector_store.search_patterns(
            hypothesis="",
            pattern_type="failure",
            family=family,
            limit=100,
            collection_name=self.collection_name,
        )

        total = len(successes) + len(failures)
        success_rate = len(successes) / total if total > 0 else 0.0

        # Count failure reasons
        reason_counts: dict[str, int] = {}
        for f in failures:
            reasons = f.get("failure_reasons", [])
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        common_failures = sorted(
            reason_counts.keys(), key=lambda x: reason_counts[x], reverse=True
        )[:3]

        # Get best IC
        best_ic = 0.0
        for s in successes:
            metrics = s.get("metrics", {})
            if "ic" in metrics:
                best_ic = max(best_ic, metrics["ic"])

        return {
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": success_rate,
            "common_failures": common_failures,
            "best_ic": best_ic,
        }

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern from both stores.

        Args:
            pattern_id: Pattern ID to delete

        Returns:
            True if deleted from vector store (DB deletion is best-effort)
        """
        # Delete from vector store
        deleted = self.vector_store.delete_pattern(
            pattern_id=pattern_id,
            collection_name=self.collection_name,
        )

        # Delete from PostgreSQL (best-effort, log errors but don't fail)
        from iqfmp.db.models import PatternRecordORM

        def _do_delete(session: "Session") -> None:
            session.query(PatternRecordORM).filter(
                PatternRecordORM.pattern_id == pattern_id
            ).delete()

        self._execute_with_session(
            _do_delete,
            commit=True,
            error_msg=f"Failed to delete pattern {pattern_id} from DB",
            suppress_error=True,  # Don't fail if DB delete fails
        )

        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get overall pattern memory statistics."""
        return self.vector_store.get_pattern_stats(self.collection_name)
