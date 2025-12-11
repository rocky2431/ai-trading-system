"""Redundancy Detector for Factor Mining.

This module implements factor redundancy detection using hierarchical clustering
to identify and filter out highly correlated factors:
- Correlation matrix computation
- Hierarchical clustering
- Optimal factor selection from clusters
- Redundancy report generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class InsufficientFactorsError(Exception):
    """Raised when there are not enough factors for analysis."""

    pass


@dataclass
class RedundancyConfig:
    """Configuration for redundancy detection."""

    # Clustering settings
    correlation_threshold: float = 0.85  # Threshold for considering factors redundant
    linkage_method: str = "average"  # Linkage method: single, complete, average, ward
    distance_metric: str = "correlation"  # Distance metric

    # Selection criteria for keeping best factor in cluster
    selection_criterion: str = "sharpe"  # sharpe, ic, ir, or combined

    # Minimum requirements
    min_factors: int = 2  # Minimum factors needed for analysis
    min_samples: int = 30  # Minimum samples for correlation calculation

    # Output settings
    keep_cluster_info: bool = True  # Include detailed cluster information

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "correlation_threshold": self.correlation_threshold,
            "linkage_method": self.linkage_method,
            "distance_metric": self.distance_metric,
            "selection_criterion": self.selection_criterion,
            "min_factors": self.min_factors,
            "min_samples": self.min_samples,
            "keep_cluster_info": self.keep_cluster_info,
        }


@dataclass
class RedundantCluster:
    """Information about a cluster of redundant factors."""

    cluster_id: int
    factor_names: list[str]
    best_factor: str
    avg_correlation: float
    size: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "factor_names": self.factor_names,
            "best_factor": self.best_factor,
            "avg_correlation": round(self.avg_correlation, 4),
            "size": self.size,
        }


@dataclass
class RedundancyResult:
    """Result of redundancy detection."""

    # Summary
    total_factors: int = 0
    retained_count: int = 0
    removed_count: int = 0
    reduction_ratio: float = 0.0  # % of factors removed

    # Factor lists
    retained_factors: list[str] = field(default_factory=list)
    removed_factors: list[str] = field(default_factory=list)

    # Cluster information
    n_clusters: int = 0
    redundant_clusters: list[RedundantCluster] = field(default_factory=list)

    # Correlation matrix
    correlation_matrix: Optional[pd.DataFrame] = None

    # Metadata
    analysis_date: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API response."""
        result = {
            "total_factors": self.total_factors,
            "retained_factors": self.retained_factors,
            "removed_factors": self.removed_factors,
            "factor_reduction_ratio": round(self.reduction_ratio, 4),
            "n_clusters": self.n_clusters,
            "redundant_groups": [c.to_dict() for c in self.redundant_clusters],
        }
        return result


class RedundancyDetector:
    """Detector for identifying and filtering redundant factors.

    Uses hierarchical clustering on factor correlation matrix to identify
    groups of highly correlated factors, then selects the best factor
    from each group.
    """

    def __init__(self, config: Optional[RedundancyConfig] = None) -> None:
        """Initialize detector.

        Args:
            config: Detection configuration
        """
        self.config = config or RedundancyConfig()

    def detect(
        self,
        factor_data: pd.DataFrame,
        factor_metrics: Optional[dict[str, dict[str, float]]] = None,
    ) -> RedundancyResult:
        """Detect redundant factors and recommend which to keep.

        Args:
            factor_data: DataFrame where each column is a factor's values
            factor_metrics: Optional dict mapping factor names to their
                           performance metrics (sharpe, ic, ir)

        Returns:
            RedundancyResult with recommendations
        """
        # Validate inputs
        self._validate_inputs(factor_data)

        factor_names = list(factor_data.columns)
        n_factors = len(factor_names)

        # If only one factor, nothing to do
        if n_factors == 1:
            return RedundancyResult(
                total_factors=1,
                retained_count=1,
                removed_count=0,
                reduction_ratio=0.0,
                retained_factors=factor_names,
                removed_factors=[],
                n_clusters=1,
            )

        # Calculate correlation matrix
        corr_matrix = self._calculate_correlation_matrix(factor_data)

        # Perform hierarchical clustering
        clusters = self._cluster_factors(corr_matrix, factor_names)

        # Select best factor from each cluster
        retained, removed, cluster_info = self._select_best_factors(
            clusters, factor_names, factor_metrics, corr_matrix
        )

        # Build result
        reduction_ratio = len(removed) / n_factors if n_factors > 0 else 0.0

        return RedundancyResult(
            total_factors=n_factors,
            retained_count=len(retained),
            removed_count=len(removed),
            reduction_ratio=reduction_ratio,
            retained_factors=retained,
            removed_factors=removed,
            n_clusters=len(cluster_info),
            redundant_clusters=cluster_info,
            correlation_matrix=corr_matrix if self.config.keep_cluster_info else None,
        )

    def _validate_inputs(self, factor_data: pd.DataFrame) -> None:
        """Validate input data."""
        if factor_data.empty:
            raise InsufficientFactorsError("Factor data is empty")

        if len(factor_data.columns) < self.config.min_factors:
            raise InsufficientFactorsError(
                f"Need at least {self.config.min_factors} factors, got {len(factor_data.columns)}"
            )

        if len(factor_data) < self.config.min_samples:
            raise InsufficientFactorsError(
                f"Need at least {self.config.min_samples} samples, got {len(factor_data)}"
            )

    def _calculate_correlation_matrix(
        self, factor_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate pairwise Spearman correlation matrix."""
        # Use Spearman correlation for robustness
        corr_matrix = factor_data.corr(method="spearman")

        # Fill NaN with 0 (uncorrelated)
        corr_matrix = corr_matrix.fillna(0)

        return corr_matrix

    def _cluster_factors(
        self,
        corr_matrix: pd.DataFrame,
        factor_names: list[str],
    ) -> dict[int, list[str]]:
        """Cluster factors using hierarchical clustering.

        Returns:
            Dictionary mapping cluster ID to list of factor names
        """
        n = len(factor_names)

        # Convert correlation to distance (1 - |corr|)
        distance_matrix = 1 - np.abs(corr_matrix.values)

        # Ensure symmetry and no negative values
        distance_matrix = np.maximum(distance_matrix, 0)
        np.fill_diagonal(distance_matrix, 0)

        # Convert to condensed form for scipy
        condensed_dist = squareform(distance_matrix)

        # Perform hierarchical clustering
        linkage_matrix = linkage(
            condensed_dist,
            method=self.config.linkage_method,
        )

        # Cut tree at threshold (distance = 1 - correlation_threshold)
        distance_threshold = 1 - self.config.correlation_threshold
        cluster_labels = fcluster(
            linkage_matrix,
            t=distance_threshold,
            criterion="distance",
        )

        # Group factors by cluster
        clusters: dict[int, list[str]] = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(factor_names[i])

        return clusters

    def _select_best_factors(
        self,
        clusters: dict[int, list[str]],
        factor_names: list[str],
        factor_metrics: Optional[dict[str, dict[str, float]]],
        corr_matrix: pd.DataFrame,
    ) -> tuple[list[str], list[str], list[RedundantCluster]]:
        """Select best factor from each cluster.

        Returns:
            (retained_factors, removed_factors, cluster_info)
        """
        retained = []
        removed = []
        cluster_info = []

        for cluster_id, members in clusters.items():
            if len(members) == 1:
                # Single factor cluster - keep it
                retained.append(members[0])
                cluster_info.append(RedundantCluster(
                    cluster_id=cluster_id,
                    factor_names=members,
                    best_factor=members[0],
                    avg_correlation=1.0,
                    size=1,
                ))
            else:
                # Multiple factors - select best
                best = self._select_best_in_cluster(members, factor_metrics)
                retained.append(best)
                removed.extend([f for f in members if f != best])

                # Calculate average correlation within cluster
                avg_corr = self._calculate_cluster_correlation(members, corr_matrix)

                cluster_info.append(RedundantCluster(
                    cluster_id=cluster_id,
                    factor_names=members,
                    best_factor=best,
                    avg_correlation=avg_corr,
                    size=len(members),
                ))

        return retained, removed, cluster_info

    def _select_best_in_cluster(
        self,
        members: list[str],
        factor_metrics: Optional[dict[str, dict[str, float]]],
    ) -> str:
        """Select best factor within a cluster based on metrics."""
        if not factor_metrics:
            # No metrics provided - just return first one
            return members[0]

        criterion = self.config.selection_criterion
        best_factor = members[0]
        best_score = float("-inf")

        for factor in members:
            if factor not in factor_metrics:
                continue

            metrics = factor_metrics[factor]

            if criterion == "sharpe":
                score = metrics.get("sharpe", 0)
            elif criterion == "ic":
                score = abs(metrics.get("ic", 0))
            elif criterion == "ir":
                score = metrics.get("ir", 0)
            elif criterion == "combined":
                # Combined score: normalize and sum
                sharpe = metrics.get("sharpe", 0)
                ic = abs(metrics.get("ic", 0)) * 20  # Scale IC to similar range
                ir = metrics.get("ir", 0)
                score = sharpe + ic + ir
            else:
                score = metrics.get("sharpe", 0)

            if score > best_score:
                best_score = score
                best_factor = factor

        return best_factor

    def _calculate_cluster_correlation(
        self,
        members: list[str],
        corr_matrix: pd.DataFrame,
    ) -> float:
        """Calculate average absolute correlation within cluster."""
        if len(members) < 2:
            return 1.0

        correlations = []
        for i, f1 in enumerate(members):
            for f2 in members[i + 1:]:
                if f1 in corr_matrix.columns and f2 in corr_matrix.columns:
                    correlations.append(abs(corr_matrix.loc[f1, f2]))

        return float(np.mean(correlations)) if correlations else 1.0

    def get_pairwise_correlations(
        self,
        factor_data: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> list[tuple[str, str, float]]:
        """Get all pairwise correlations above threshold.

        Args:
            factor_data: DataFrame where each column is a factor
            threshold: Correlation threshold (default: config threshold)

        Returns:
            List of (factor1, factor2, correlation) tuples
        """
        threshold = threshold or self.config.correlation_threshold
        corr_matrix = self._calculate_correlation_matrix(factor_data)

        pairs = []
        factor_names = list(factor_data.columns)

        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i + 1:]:
                corr = abs(corr_matrix.loc[f1, f2])
                if corr >= threshold:
                    pairs.append((f1, f2, float(corr)))

        # Sort by correlation descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


class IncrementalRedundancyChecker:
    """Check if a new factor is redundant with existing factors.

    Useful for online/streaming scenarios where factors are added one at a time.
    """

    def __init__(
        self,
        existing_factors: pd.DataFrame,
        correlation_threshold: float = 0.85,
    ) -> None:
        """Initialize checker.

        Args:
            existing_factors: DataFrame of existing factor values
            correlation_threshold: Threshold for redundancy
        """
        self.existing_factors = existing_factors
        self.threshold = correlation_threshold
        self.factor_names = list(existing_factors.columns)

    def is_redundant(
        self,
        new_factor: pd.Series,
        new_factor_name: str,
    ) -> tuple[bool, Optional[str], float]:
        """Check if new factor is redundant with existing ones.

        Args:
            new_factor: Series of new factor values
            new_factor_name: Name of new factor

        Returns:
            (is_redundant, most_correlated_factor, max_correlation)
        """
        if len(self.existing_factors.columns) == 0:
            return False, None, 0.0

        max_corr = 0.0
        most_correlated = None

        for existing_name in self.factor_names:
            existing = self.existing_factors[existing_name]

            # Calculate Spearman correlation
            corr, _ = stats.spearmanr(new_factor, existing, nan_policy="omit")

            if not np.isnan(corr) and abs(corr) > max_corr:
                max_corr = abs(corr)
                most_correlated = existing_name

        is_redundant = max_corr >= self.threshold
        return is_redundant, most_correlated, max_corr

    def add_factor(self, new_factor: pd.Series, name: str) -> None:
        """Add a new factor to the existing set.

        Args:
            new_factor: Series of factor values
            name: Name of the factor
        """
        self.existing_factors[name] = new_factor.values
        self.factor_names.append(name)


class CorrelationHeatmapData:
    """Generate data for correlation heatmap visualization."""

    @staticmethod
    def generate(
        factor_data: pd.DataFrame,
        sort_by_cluster: bool = True,
    ) -> dict[str, Any]:
        """Generate heatmap data from factor DataFrame.

        Args:
            factor_data: DataFrame where each column is a factor
            sort_by_cluster: Whether to sort factors by cluster

        Returns:
            Dictionary with heatmap data for visualization
        """
        # Calculate correlation matrix
        corr_matrix = factor_data.corr(method="spearman")

        factor_names = list(corr_matrix.columns)

        if sort_by_cluster:
            # Cluster and reorder for better visualization
            detector = RedundancyDetector()
            clusters = detector._cluster_factors(corr_matrix, factor_names)

            # Flatten clusters into ordered list
            ordered_names = []
            for cluster_factors in clusters.values():
                ordered_names.extend(cluster_factors)

            # Reorder correlation matrix
            corr_matrix = corr_matrix.loc[ordered_names, ordered_names]
            factor_names = ordered_names

        # Convert to list format for visualization
        data = []
        for i, f1 in enumerate(factor_names):
            for j, f2 in enumerate(factor_names):
                data.append({
                    "x": f1,
                    "y": f2,
                    "value": float(corr_matrix.loc[f1, f2]),
                })

        return {
            "factors": factor_names,
            "data": data,
            "min_value": -1.0,
            "max_value": 1.0,
        }
