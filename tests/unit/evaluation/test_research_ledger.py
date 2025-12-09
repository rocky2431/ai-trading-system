"""Tests for ResearchLedger (Task 11).

Six-dimensional test coverage:
1. Functional: Trial recording, threshold calculation, statistics
2. Boundary: Edge cases for trial counts and metrics
3. Exception: Error handling for invalid inputs
4. Performance: Query and calculation time
5. Security: Data integrity and persistence
6. Compatibility: Different storage backends
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import time

from iqfmp.evaluation.research_ledger import (
    ResearchLedger,
    TrialRecord,
    DynamicThreshold,
    ThresholdConfig,
    LedgerStorage,
    MemoryStorage,
    FileStorage,
    LedgerStatistics,
    InvalidTrialError,
    ThresholdExceededWarning,
)


@pytest.fixture
def sample_trial() -> TrialRecord:
    """Create a sample trial record."""
    return TrialRecord(
        trial_id="trial_001",
        factor_name="momentum_20d",
        factor_family="momentum",
        sharpe_ratio=1.5,
        ic_mean=0.05,
        ir=0.8,
        created_at=datetime.now(),
        metadata={"user": "test"},
    )


@pytest.fixture
def ledger() -> ResearchLedger:
    """Create a ledger with memory storage."""
    return ResearchLedger(storage=MemoryStorage())


class TestTrialRecord:
    """Tests for TrialRecord data structure."""

    def test_create_trial_record(self) -> None:
        """Test creating a trial record."""
        record = TrialRecord(
            trial_id="trial_001",
            factor_name="test_factor",
            factor_family="momentum",
            sharpe_ratio=1.2,
            ic_mean=0.03,
            ir=0.5,
        )
        assert record.trial_id == "trial_001"
        assert record.factor_name == "test_factor"
        assert record.sharpe_ratio == 1.2

    def test_trial_record_auto_id(self) -> None:
        """Test automatic trial ID generation."""
        record = TrialRecord(
            factor_name="test_factor",
            factor_family="momentum",
            sharpe_ratio=1.0,
        )
        assert record.trial_id is not None
        assert len(record.trial_id) > 0

    def test_trial_record_timestamp(self) -> None:
        """Test automatic timestamp generation."""
        record = TrialRecord(
            factor_name="test_factor",
            factor_family="momentum",
            sharpe_ratio=1.0,
        )
        assert record.created_at is not None
        assert isinstance(record.created_at, datetime)

    def test_trial_record_to_dict(self, sample_trial: TrialRecord) -> None:
        """Test serialization to dictionary."""
        d = sample_trial.to_dict()
        assert "trial_id" in d
        assert "factor_name" in d
        assert "sharpe_ratio" in d
        assert "created_at" in d

    def test_trial_record_from_dict(self, sample_trial: TrialRecord) -> None:
        """Test deserialization from dictionary."""
        d = sample_trial.to_dict()
        record = TrialRecord.from_dict(d)
        assert record.trial_id == sample_trial.trial_id
        assert record.factor_name == sample_trial.factor_name
        assert record.sharpe_ratio == sample_trial.sharpe_ratio

    def test_trial_record_validation(self) -> None:
        """Test trial record validation."""
        with pytest.raises(InvalidTrialError, match="factor_name"):
            TrialRecord(
                factor_name="",  # Empty name
                factor_family="momentum",
                sharpe_ratio=1.0,
            )


class TestDynamicThreshold:
    """Tests for DynamicThreshold calculator."""

    def test_default_config(self) -> None:
        """Test default threshold configuration."""
        threshold = DynamicThreshold()
        assert threshold.config.base_sharpe_threshold == 2.0
        assert threshold.config.confidence_level == 0.95

    def test_custom_config(self) -> None:
        """Test custom threshold configuration."""
        config = ThresholdConfig(
            base_sharpe_threshold=1.5,
            confidence_level=0.99,
        )
        threshold = DynamicThreshold(config)
        assert threshold.config.base_sharpe_threshold == 1.5

    def test_threshold_increases_with_trials(self) -> None:
        """Test that threshold increases with more trials."""
        threshold = DynamicThreshold()

        t1 = threshold.calculate(n_trials=10)
        t2 = threshold.calculate(n_trials=100)
        t3 = threshold.calculate(n_trials=1000)

        assert t2 > t1, "More trials should increase threshold"
        assert t3 > t2, "More trials should increase threshold"

    def test_threshold_at_one_trial(self) -> None:
        """Test threshold with single trial."""
        threshold = DynamicThreshold()
        t = threshold.calculate(n_trials=1)

        # Should be close to base threshold
        assert t >= threshold.config.base_sharpe_threshold * 0.9
        assert t <= threshold.config.base_sharpe_threshold * 1.1

    def test_deflated_sharpe_formula(self) -> None:
        """Test deflated sharpe ratio calculation."""
        threshold = DynamicThreshold()

        # With many trials, threshold should be significantly higher
        t_many = threshold.calculate(n_trials=10000)
        assert t_many > threshold.config.base_sharpe_threshold * 1.5

    def test_threshold_with_different_confidence(self) -> None:
        """Test threshold at different confidence levels."""
        t1 = DynamicThreshold(ThresholdConfig(confidence_level=0.90))
        t2 = DynamicThreshold(ThresholdConfig(confidence_level=0.99))

        # Higher confidence should mean higher threshold
        assert t2.calculate(100) > t1.calculate(100)

    def test_check_passes(self) -> None:
        """Test threshold check that passes."""
        threshold = DynamicThreshold()
        result = threshold.check(sharpe=3.0, n_trials=10)

        assert result.passes is True
        assert result.threshold > 0

    def test_check_fails(self) -> None:
        """Test threshold check that fails."""
        threshold = DynamicThreshold()
        result = threshold.check(sharpe=1.0, n_trials=1000)

        assert result.passes is False


class TestResearchLedgerFunctional:
    """Functional tests for ResearchLedger."""

    def test_record_trial(self, ledger: ResearchLedger, sample_trial: TrialRecord) -> None:
        """Test recording a trial."""
        trial_id = ledger.record(sample_trial)

        assert trial_id == sample_trial.trial_id
        assert ledger.get_trial_count() == 1

    def test_record_multiple_trials(self, ledger: ResearchLedger) -> None:
        """Test recording multiple trials."""
        for i in range(10):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.0 + i * 0.1,
            )
            ledger.record(trial)

        assert ledger.get_trial_count() == 10

    def test_get_trial(self, ledger: ResearchLedger, sample_trial: TrialRecord) -> None:
        """Test retrieving a trial by ID."""
        ledger.record(sample_trial)
        retrieved = ledger.get_trial(sample_trial.trial_id)

        assert retrieved is not None
        assert retrieved.trial_id == sample_trial.trial_id
        assert retrieved.factor_name == sample_trial.factor_name

    def test_get_nonexistent_trial(self, ledger: ResearchLedger) -> None:
        """Test retrieving nonexistent trial."""
        result = ledger.get_trial("nonexistent_id")
        assert result is None

    def test_get_all_trials(self, ledger: ResearchLedger) -> None:
        """Test retrieving all trials."""
        for i in range(5):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.0,
            )
            ledger.record(trial)

        all_trials = ledger.get_all_trials()
        assert len(all_trials) == 5

    def test_get_trials_by_family(self, ledger: ResearchLedger) -> None:
        """Test filtering trials by factor family."""
        families = ["momentum", "value", "momentum", "volatility"]
        for i, family in enumerate(families):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family=family,
                sharpe_ratio=1.0,
            )
            ledger.record(trial)

        momentum_trials = ledger.get_trials_by_family("momentum")
        assert len(momentum_trials) == 2

    def test_get_current_threshold(self, ledger: ResearchLedger) -> None:
        """Test getting current threshold based on trial count."""
        # Record some trials
        for i in range(10):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.0,
            )
            ledger.record(trial)

        threshold = ledger.get_current_threshold()
        assert threshold > 0

    def test_check_factor_significance(self, ledger: ResearchLedger) -> None:
        """Test checking if a factor meets significance threshold."""
        # Record trials to establish baseline
        for i in range(10):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.5,
            )
            ledger.record(trial)

        # High sharpe should pass
        result = ledger.check_significance(sharpe=3.0)
        assert result.passes is True

        # Low sharpe should fail
        result = ledger.check_significance(sharpe=0.5)
        assert result.passes is False


class TestResearchLedgerBoundary:
    """Boundary tests for ResearchLedger."""

    def test_empty_ledger(self, ledger: ResearchLedger) -> None:
        """Test operations on empty ledger."""
        assert ledger.get_trial_count() == 0
        assert ledger.get_all_trials() == []
        assert ledger.get_current_threshold() > 0

    def test_very_high_trial_count(self, ledger: ResearchLedger) -> None:
        """Test with many trials."""
        n_trials = 1000
        for i in range(n_trials):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.0 + (i % 10) * 0.1,
            )
            ledger.record(trial)

        assert ledger.get_trial_count() == n_trials

        # Threshold should be quite high
        threshold = ledger.get_current_threshold()
        assert threshold > 2.0

    def test_extreme_sharpe_values(self, ledger: ResearchLedger) -> None:
        """Test with extreme sharpe ratios."""
        extreme_sharpes = [-10.0, -1.0, 0.0, 10.0, 100.0]
        for i, sharpe in enumerate(extreme_sharpes):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=sharpe,
            )
            ledger.record(trial)

        assert ledger.get_trial_count() == 5

    def test_duplicate_factor_names(self, ledger: ResearchLedger) -> None:
        """Test recording trials with duplicate factor names."""
        for i in range(3):
            trial = TrialRecord(
                factor_name="same_factor",  # Same name
                factor_family="momentum",
                sharpe_ratio=1.0 + i * 0.5,
            )
            ledger.record(trial)

        # All should be recorded (different trial IDs)
        assert ledger.get_trial_count() == 3


class TestResearchLedgerException:
    """Exception handling tests for ResearchLedger."""

    def test_record_invalid_trial(self, ledger: ResearchLedger) -> None:
        """Test recording invalid trial."""
        with pytest.raises(InvalidTrialError):
            trial = TrialRecord(
                factor_name="",  # Invalid empty name
                factor_family="momentum",
                sharpe_ratio=1.0,
            )
            ledger.record(trial)

    def test_invalid_family_filter(self, ledger: ResearchLedger) -> None:
        """Test filtering by nonexistent family."""
        trial = TrialRecord(
            factor_name="test",
            factor_family="momentum",
            sharpe_ratio=1.0,
        )
        ledger.record(trial)

        # Should return empty list, not error
        result = ledger.get_trials_by_family("nonexistent")
        assert result == []


class TestLedgerStatistics:
    """Tests for ledger statistics."""

    def test_get_statistics(self, ledger: ResearchLedger) -> None:
        """Test getting ledger statistics."""
        sharpes = [0.5, 1.0, 1.5, 2.0, 2.5]
        for i, sharpe in enumerate(sharpes):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=sharpe,
            )
            ledger.record(trial)

        stats = ledger.get_statistics()

        assert stats.total_trials == 5
        assert stats.mean_sharpe == pytest.approx(1.5, rel=0.01)
        assert stats.max_sharpe == 2.5
        assert stats.min_sharpe == 0.5

    def test_statistics_by_family(self, ledger: ResearchLedger) -> None:
        """Test statistics grouped by family."""
        families = ["momentum", "momentum", "value", "value"]
        sharpes = [1.0, 2.0, 1.5, 2.5]

        for family, sharpe in zip(families, sharpes):
            trial = TrialRecord(
                factor_name=f"factor_{sharpe}",
                factor_family=family,
                sharpe_ratio=sharpe,
            )
            ledger.record(trial)

        stats = ledger.get_statistics_by_family()

        assert "momentum" in stats
        assert "value" in stats
        assert stats["momentum"].mean_sharpe == pytest.approx(1.5, rel=0.01)
        assert stats["value"].mean_sharpe == pytest.approx(2.0, rel=0.01)

    def test_empty_statistics(self, ledger: ResearchLedger) -> None:
        """Test statistics on empty ledger."""
        stats = ledger.get_statistics()

        assert stats.total_trials == 0
        assert stats.mean_sharpe == 0.0


class TestLedgerStorage:
    """Tests for storage backends."""

    def test_memory_storage(self) -> None:
        """Test memory storage backend."""
        storage = MemoryStorage()
        ledger = ResearchLedger(storage=storage)

        trial = TrialRecord(
            factor_name="test",
            factor_family="momentum",
            sharpe_ratio=1.0,
        )
        ledger.record(trial)

        assert ledger.get_trial_count() == 1

    def test_file_storage(self) -> None:
        """Test file storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "ledger.json"
            storage = FileStorage(filepath)
            ledger = ResearchLedger(storage=storage)

            # Record trial
            trial = TrialRecord(
                factor_name="test",
                factor_family="momentum",
                sharpe_ratio=1.0,
            )
            ledger.record(trial)

            # Verify file exists
            assert filepath.exists()

            # Create new ledger from same file
            storage2 = FileStorage(filepath)
            ledger2 = ResearchLedger(storage=storage2)

            assert ledger2.get_trial_count() == 1

    def test_file_storage_persistence(self) -> None:
        """Test that file storage persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "ledger.json"

            # First instance
            storage1 = FileStorage(filepath)
            ledger1 = ResearchLedger(storage=storage1)

            for i in range(5):
                trial = TrialRecord(
                    factor_name=f"factor_{i}",
                    factor_family="momentum",
                    sharpe_ratio=1.0,
                )
                ledger1.record(trial)

            # Second instance
            storage2 = FileStorage(filepath)
            ledger2 = ResearchLedger(storage=storage2)

            assert ledger2.get_trial_count() == 5


class TestResearchLedgerPerformance:
    """Performance tests for ResearchLedger."""

    def test_recording_performance(self) -> None:
        """Test performance of trial recording."""
        ledger = ResearchLedger(storage=MemoryStorage())

        start = time.time()
        for i in range(1000):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.0,
            )
            ledger.record(trial)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Recording 1000 trials took {elapsed}s"

    def test_query_performance(self) -> None:
        """Test performance of trial queries."""
        ledger = ResearchLedger(storage=MemoryStorage())

        # Record trials
        for i in range(1000):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family=["momentum", "value", "volatility"][i % 3],
                sharpe_ratio=1.0,
            )
            ledger.record(trial)

        # Query performance
        start = time.time()
        for _ in range(100):
            ledger.get_trials_by_family("momentum")
        elapsed = time.time() - start

        assert elapsed < 1.0, f"100 queries took {elapsed}s"

    def test_threshold_calculation_performance(self) -> None:
        """Test performance of threshold calculation."""
        threshold = DynamicThreshold()

        start = time.time()
        for n in range(1, 10001):
            threshold.calculate(n)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"10000 calculations took {elapsed}s"


class TestResearchLedgerCompatibility:
    """Compatibility tests."""

    def test_json_serialization(self, ledger: ResearchLedger, sample_trial: TrialRecord) -> None:
        """Test JSON serialization of trials."""
        ledger.record(sample_trial)
        trial = ledger.get_trial(sample_trial.trial_id)

        # Should be JSON serializable
        json_str = json.dumps(trial.to_dict(), default=str)
        assert len(json_str) > 0

        # Should be deserializable
        data = json.loads(json_str)
        restored = TrialRecord.from_dict(data)
        assert restored.factor_name == sample_trial.factor_name

    def test_different_metric_combinations(self, ledger: ResearchLedger) -> None:
        """Test trials with different metric combinations."""
        # Minimal metrics
        trial1 = TrialRecord(
            factor_name="minimal",
            factor_family="momentum",
            sharpe_ratio=1.0,
        )
        ledger.record(trial1)

        # Full metrics
        trial2 = TrialRecord(
            factor_name="full",
            factor_family="momentum",
            sharpe_ratio=1.5,
            ic_mean=0.05,
            ir=0.8,
            max_drawdown=0.15,
            win_rate=0.55,
            metadata={"extra": "data"},
        )
        ledger.record(trial2)

        assert ledger.get_trial_count() == 2

    def test_ledger_export_import(self, ledger: ResearchLedger) -> None:
        """Test exporting and importing ledger data."""
        # Record trials
        for i in range(5):
            trial = TrialRecord(
                factor_name=f"factor_{i}",
                factor_family="momentum",
                sharpe_ratio=1.0 + i * 0.1,
            )
            ledger.record(trial)

        # Export
        export_data = ledger.export_to_dict()
        assert "trials" in export_data
        assert len(export_data["trials"]) == 5

        # Import to new ledger
        new_ledger = ResearchLedger(storage=MemoryStorage())
        new_ledger.import_from_dict(export_data)

        assert new_ledger.get_trial_count() == 5
