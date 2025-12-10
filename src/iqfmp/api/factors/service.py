"""Factor service for business logic."""

import hashlib
import uuid
from datetime import datetime
from typing import Optional

from iqfmp.models.factor import Factor, FactorMetrics, FactorStatus, StabilityReport


class FactorNotFoundError(Exception):
    """Raised when factor is not found."""

    pass


class FactorService:
    """Service for factor management."""

    def __init__(self) -> None:
        """Initialize factor service."""
        self._factors: dict[str, Factor] = {}

    def _generate_code_hash(self, code: str) -> str:
        """Generate hash for factor code."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def _generate_name(self, description: str, family: list[str]) -> str:
        """Generate factor name from description."""
        # Simple name generation
        words = description.lower().split()[:3]
        base = "_".join(w for w in words if w.isalnum())
        if family:
            base = f"{family[0]}_{base}"
        return base[:50]

    def create_factor(
        self,
        name: str,
        family: list[str],
        code: str,
        target_task: str,
    ) -> Factor:
        """Create a new factor.

        Args:
            name: Factor name
            family: Factor family tags
            code: Factor computation code
            target_task: Target prediction task

        Returns:
            Created factor
        """
        factor_id = str(uuid.uuid4())
        code_hash = self._generate_code_hash(code)

        factor = Factor(
            id=factor_id,
            name=name,
            family=family,
            code=code,
            code_hash=code_hash,
            target_task=target_task,
            status=FactorStatus.CANDIDATE,
            created_at=datetime.now(),
        )

        self._factors[factor_id] = factor
        return factor

    def generate_factor(
        self,
        description: str,
        family: list[str],
        target_task: str,
    ) -> Factor:
        """Generate a new factor from description.

        Args:
            description: Natural language description
            family: Factor family tags
            target_task: Target prediction task

        Returns:
            Generated factor
        """
        # In production, this would call LLM to generate code
        # For now, generate a simple placeholder
        name = self._generate_name(description, family)
        code = f'''def {name}(df):
    """
    {description}
    """
    # Generated factor code
    import pandas as pd
    return pd.Series(index=df.index, data=0.0)
'''
        return self.create_factor(
            name=name,
            family=family,
            code=code,
            target_task=target_task,
        )

    def get_factor(self, factor_id: str) -> Optional[Factor]:
        """Get factor by ID.

        Args:
            factor_id: Factor ID

        Returns:
            Factor if found, None otherwise
        """
        return self._factors.get(factor_id)

    def list_factors(
        self,
        page: int = 1,
        page_size: int = 10,
        family: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[Factor], int]:
        """List factors with pagination and filtering.

        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            family: Filter by family
            status: Filter by status

        Returns:
            Tuple of (factors, total_count)
        """
        factors = list(self._factors.values())

        # Apply filters
        if family:
            factors = [f for f in factors if family in f.family]
        if status:
            factors = [
                f for f in factors
                if (f.status.value if hasattr(f.status, "value") else str(f.status)) == status
            ]

        total = len(factors)

        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        factors = factors[start:end]

        return factors, total

    def update_status(self, factor_id: str, status: str) -> Factor:
        """Update factor status.

        Args:
            factor_id: Factor ID
            status: New status

        Returns:
            Updated factor

        Raises:
            FactorNotFoundError: If factor not found
        """
        factor = self._factors.get(factor_id)
        if not factor:
            raise FactorNotFoundError(f"Factor {factor_id} not found")

        # Update status
        factor.status = FactorStatus(status)
        return factor

    def delete_factor(self, factor_id: str) -> bool:
        """Delete a factor.

        Args:
            factor_id: Factor ID

        Returns:
            True if deleted, False if not found
        """
        if factor_id in self._factors:
            del self._factors[factor_id]
            return True
        return False

    def evaluate_factor(
        self,
        factor_id: str,
        splits: list[str],
        market_splits: list[str] = None,
    ) -> tuple[FactorMetrics, StabilityReport, bool, int]:
        """Evaluate a factor.

        Args:
            factor_id: Factor ID
            splits: Data splits to evaluate on
            market_splits: Market splits to evaluate on

        Returns:
            Tuple of (metrics, stability, passed_threshold, experiment_number)

        Raises:
            FactorNotFoundError: If factor not found
        """
        factor = self._factors.get(factor_id)
        if not factor:
            raise FactorNotFoundError(f"Factor {factor_id} not found")

        # In production, this would run actual evaluation
        # For now, return mock results
        metrics = FactorMetrics(
            ic_mean=0.05,
            ic_std=0.02,
            ir=2.5,
            sharpe=1.8,
            max_drawdown=0.15,
            turnover=0.3,
            ic_by_split={s: 0.05 for s in splits},
            sharpe_by_split={s: 1.8 for s in splits},
        )

        stability = StabilityReport(
            time_stability={"monthly": 0.8, "quarterly": 0.75},
            market_stability={"btc": 0.9, "altcoins": 0.7},
            regime_stability={"bull": 0.85, "bear": 0.6},
        )

        # Update factor with metrics
        factor.metrics = metrics
        factor.stability = stability
        factor.experiment_number += 1

        # Check if passed threshold (simplified)
        passed = metrics.ir > 1.0 and metrics.sharpe > 1.0

        return metrics, stability, passed, factor.experiment_number


# Singleton instance
_factor_service = FactorService()


def get_factor_service() -> FactorService:
    """Get factor service instance."""
    return _factor_service
