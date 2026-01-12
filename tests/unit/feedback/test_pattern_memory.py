"""Unit tests for PatternMemory."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from iqfmp.feedback.pattern_memory import PatternMemory, PatternRecord
from iqfmp.feedback.structured_feedback import FailureReason, StructuredFeedback


class TestPatternRecord:
    """Tests for PatternRecord dataclass."""

    def test_basic_creation(self):
        """Test basic PatternRecord creation."""
        record = PatternRecord(
            pattern_id="test-123",
            pattern_type="success",
            hypothesis="Test hypothesis",
            factor_code="Ref($close, -1)",
            factor_family="momentum",
            metrics={"ic": 0.05, "ir": 1.5},
        )

        assert record.pattern_id == "test-123"
        assert record.pattern_type == "success"
        assert record.hypothesis == "Test hypothesis"
        assert record.factor_code == "Ref($close, -1)"
        assert record.factor_family == "momentum"
        assert record.metrics == {"ic": 0.05, "ir": 1.5}

    def test_creation_with_failure_details(self):
        """Test PatternRecord with failure details."""
        record = PatternRecord(
            pattern_id="test-456",
            pattern_type="failure",
            hypothesis="Failed hypothesis",
            factor_code="$close / $open",
            factor_family="value",
            metrics={"ic": 0.01, "ir": 0.5},
            feedback="Low IC feedback",
            failure_reasons=["low_ic", "low_ir"],
            trial_id="trial-789",
        )

        assert record.pattern_type == "failure"
        assert record.feedback == "Low IC feedback"
        assert record.failure_reasons == ["low_ic", "low_ir"]
        assert record.trial_id == "trial-789"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = PatternRecord(
            pattern_id="test-123",
            pattern_type="success",
            hypothesis="Test hypothesis",
            factor_code="Ref($close, -1)",
            factor_family="momentum",
            metrics={"ic": 0.05},
            score=0.95,
        )

        result = record.to_dict()

        assert result["pattern_id"] == "test-123"
        assert result["pattern_type"] == "success"
        assert result["metrics"] == {"ic": 0.05}
        assert result["score"] == 0.95
        assert "created_at" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "pattern_id": "test-123",
            "pattern_type": "success",
            "hypothesis": "Test hypothesis",
            "factor_code": "Ref($close, -1)",
            "factor_family": "momentum",
            "metrics": {"ic": 0.05, "ir": 1.5},
            "feedback": None,
            "failure_reasons": [],
            "trial_id": None,
            "created_at": "2025-01-01T00:00:00",
            "score": 0.85,
        }

        record = PatternRecord.from_dict(data)

        assert record.pattern_id == "test-123"
        assert record.pattern_type == "success"
        assert record.metrics == {"ic": 0.05, "ir": 1.5}
        assert record.score == 0.85
        assert isinstance(record.created_at, datetime)

    def test_from_dict_without_created_at(self):
        """Test from_dict when created_at is missing."""
        data = {
            "pattern_id": "test-123",
            "pattern_type": "success",
            "hypothesis": "Test",
            "factor_code": "code",
            "factor_family": "momentum",
        }

        record = PatternRecord.from_dict(data)

        assert record.pattern_id == "test-123"
        assert isinstance(record.created_at, datetime)

    def test_roundtrip_serialization(self):
        """Test roundtrip dictionary serialization."""
        original = PatternRecord(
            pattern_id="test-roundtrip",
            pattern_type="failure",
            hypothesis="Roundtrip test",
            factor_code="$volume",
            factor_family="volume",
            metrics={"ic": 0.02, "ir": 0.8, "sharpe": 0.5},
            feedback="Some feedback",
            failure_reasons=["low_ic"],
            trial_id="trial-rt",
            score=0.75,
        )

        data = original.to_dict()
        restored = PatternRecord.from_dict(data)

        assert restored.pattern_id == original.pattern_id
        assert restored.pattern_type == original.pattern_type
        assert restored.hypothesis == original.hypothesis
        assert restored.metrics == original.metrics
        assert restored.failure_reasons == original.failure_reasons
        assert restored.score == original.score


class TestPatternMemory:
    """Tests for PatternMemory class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock = MagicMock()
        mock.add_pattern.return_value = "pattern-id"
        mock.search_patterns.return_value = []
        mock.delete_pattern.return_value = True
        mock.get_pattern_stats.return_value = {"points_count": 10}
        return mock

    @pytest.fixture
    def pattern_memory(self, mock_vector_store):
        """Create PatternMemory with mock vector store."""
        return PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,  # No DB for unit tests
            collection_name="test_patterns",
        )

    def test_record_success(self, pattern_memory, mock_vector_store):
        """Test recording a success pattern."""
        result = pattern_memory.record_success(
            hypothesis="Success hypothesis",
            factor_code="Ref($close, -5)",
            factor_family="momentum",
            metrics={"ic": 0.05, "ir": 1.5, "sharpe": 2.0},
            trial_id="trial-001",
        )

        # Should return a pattern ID
        assert result is not None

        # Should call vector store
        mock_vector_store.add_pattern.assert_called_once()
        call_kwargs = mock_vector_store.add_pattern.call_args.kwargs

        assert call_kwargs["pattern_type"] == "success"
        assert call_kwargs["hypothesis"] == "Success hypothesis"
        assert call_kwargs["factor_code"] == "Ref($close, -5)"
        assert call_kwargs["family"] == "momentum"
        assert call_kwargs["metrics"] == {"ic": 0.05, "ir": 1.5, "sharpe": 2.0}
        assert call_kwargs["trial_id"] == "trial-001"

    def test_record_failure(self, pattern_memory, mock_vector_store):
        """Test recording a failure pattern."""
        feedback = StructuredFeedback(
            factor_name="test_factor",
            hypothesis="Failure hypothesis",
            factor_code="$close / $open",
            ic=0.01,
            ir=0.5,
            sharpe=0.3,
            max_drawdown=0.25,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC, FailureReason.LOW_IR],
            failure_details={
                FailureReason.LOW_IC: "IC=0.01, need >= 0.03",
                FailureReason.LOW_IR: "IR=0.5, need >= 1.0",
            },
            suggestions=["Try different approach"],
            trial_id="trial-002",
            win_rate=0.42,
        )

        result = pattern_memory.record_failure(
            hypothesis="Failure hypothesis",
            factor_code="$close / $open",
            factor_family="value",
            feedback=feedback,
        )

        assert result is not None

        mock_vector_store.add_pattern.assert_called_once()
        call_kwargs = mock_vector_store.add_pattern.call_args.kwargs

        assert call_kwargs["pattern_type"] == "failure"
        assert call_kwargs["hypothesis"] == "Failure hypothesis"
        assert call_kwargs["family"] == "value"
        assert "low_ic" in call_kwargs["failure_reasons"]
        assert "low_ir" in call_kwargs["failure_reasons"]
        assert call_kwargs["metrics"]["ic"] == 0.01
        assert call_kwargs["metrics"]["win_rate"] == 0.42

    def test_retrieve_similar_successes(self, pattern_memory, mock_vector_store):
        """Test retrieving similar success patterns."""
        mock_vector_store.search_patterns.return_value = [
            {
                "pattern_id": "success-1",
                "pattern_type": "success",
                "hypothesis": "Similar success",
                "factor_code": "code1",
                "factor_family": "momentum",
                "metrics": {"ic": 0.05},
                "feedback": None,
                "failure_reasons": [],
                "trial_id": "t1",
                "created_at": "2025-01-01T00:00:00",
                "score": 0.95,
            },
            {
                "pattern_id": "success-2",
                "pattern_type": "success",
                "hypothesis": "Another success",
                "factor_code": "code2",
                "factor_family": "momentum",
                "metrics": {"ic": 0.04},
                "feedback": None,
                "failure_reasons": [],
                "trial_id": "t2",
                "created_at": "2025-01-02T00:00:00",
                "score": 0.85,
            },
        ]

        results = pattern_memory.retrieve_similar_successes(
            hypothesis="Query hypothesis",
            family="momentum",
            limit=5,
        )

        assert len(results) == 2
        assert all(isinstance(r, PatternRecord) for r in results)
        assert results[0].pattern_id == "success-1"
        assert results[0].score == 0.95
        assert results[1].pattern_id == "success-2"

        mock_vector_store.search_patterns.assert_called_once_with(
            hypothesis="Query hypothesis",
            pattern_type="success",
            family="momentum",
            limit=5,
            collection_name="test_patterns",
        )

    def test_retrieve_similar_failures(self, pattern_memory, mock_vector_store):
        """Test retrieving similar failure patterns."""
        mock_vector_store.search_patterns.return_value = [
            {
                "pattern_id": "failure-1",
                "pattern_type": "failure",
                "hypothesis": "Failed hypothesis",
                "factor_code": "bad_code",
                "factor_family": "value",
                "metrics": {"ic": 0.01},
                "feedback": "Low IC",
                "failure_reasons": ["low_ic"],
                "trial_id": "t3",
                "created_at": "2025-01-03T00:00:00",
                "score": 0.90,
            },
        ]

        results = pattern_memory.retrieve_similar_failures(
            hypothesis="Query",
            limit=3,
        )

        assert len(results) == 1
        assert results[0].pattern_type == "failure"
        assert results[0].failure_reasons == ["low_ic"]

        mock_vector_store.search_patterns.assert_called_once_with(
            hypothesis="Query",
            pattern_type="failure",
            family=None,
            limit=3,
            collection_name="test_patterns",
        )

    def test_get_family_statistics_from_vector_store(
        self, pattern_memory, mock_vector_store
    ):
        """Test getting family statistics without DB."""
        # Mock success patterns
        mock_vector_store.search_patterns.side_effect = [
            # First call for successes
            [
                {
                    "pattern_id": "s1",
                    "pattern_type": "success",
                    "hypothesis": "h1",
                    "factor_code": "c1",
                    "factor_family": "momentum",
                    "metrics": {"ic": 0.05},
                    "failure_reasons": [],
                    "score": 0.9,
                },
                {
                    "pattern_id": "s2",
                    "pattern_type": "success",
                    "hypothesis": "h2",
                    "factor_code": "c2",
                    "factor_family": "momentum",
                    "metrics": {"ic": 0.06},
                    "failure_reasons": [],
                    "score": 0.8,
                },
            ],
            # Second call for failures
            [
                {
                    "pattern_id": "f1",
                    "pattern_type": "failure",
                    "hypothesis": "h3",
                    "factor_code": "c3",
                    "factor_family": "momentum",
                    "metrics": {"ic": 0.01},
                    "failure_reasons": ["low_ic", "low_ir"],
                    "score": 0.7,
                },
            ],
        ]

        stats = pattern_memory.get_family_statistics("momentum")

        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1
        assert stats["success_rate"] == pytest.approx(2 / 3)
        assert stats["best_ic"] == 0.06
        assert "low_ic" in stats["common_failures"]

    def test_delete_pattern(self, pattern_memory, mock_vector_store):
        """Test deleting a pattern."""
        result = pattern_memory.delete_pattern("pattern-to-delete")

        assert result is True
        mock_vector_store.delete_pattern.assert_called_once_with(
            pattern_id="pattern-to-delete",
            collection_name="test_patterns",
        )

    def test_get_stats(self, pattern_memory, mock_vector_store):
        """Test getting overall stats."""
        mock_vector_store.get_pattern_stats.return_value = {"points_count": 100}

        stats = pattern_memory.get_stats()

        assert stats == {"points_count": 100}
        mock_vector_store.get_pattern_stats.assert_called_once_with("test_patterns")


class TestPatternMemoryWithDB:
    """Tests for PatternMemory with database integration."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.rollback = MagicMock()
        session.close = MagicMock()
        session.query.return_value.filter.return_value.delete = MagicMock()
        return session

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock = MagicMock()
        mock.add_pattern.return_value = "pattern-id"
        mock.delete_pattern.return_value = True
        return mock

    @pytest.fixture
    def pattern_memory_with_db(self, mock_vector_store, mock_session):
        """Create PatternMemory with mock DB session."""
        return PatternMemory(
            vector_store=mock_vector_store,
            session_factory=lambda: mock_session,
            collection_name="test_patterns",
        )

    def test_record_success_persists_to_db(
        self, pattern_memory_with_db, mock_vector_store, mock_session
    ):
        """Test that success patterns are persisted to DB."""
        # Patch at the import location in the module
        with patch("iqfmp.db.models.PatternRecordORM") as mock_orm:
            pattern_memory_with_db.record_success(
                hypothesis="Test hypothesis",
                factor_code="test_code",
                factor_family="momentum",
                metrics={"ic": 0.05},
            )

            # Should create ORM instance
            mock_orm.assert_called_once()

            # Should add to session
            mock_session.add.assert_called_once()

            # Should commit
            mock_session.commit.assert_called_once()

            # Should close session
            mock_session.close.assert_called_once()

        # Suppress unused parameter warning
        _ = mock_vector_store

    def test_record_failure_persists_to_db(
        self, pattern_memory_with_db, mock_vector_store, mock_session
    ):
        """Test that failure patterns are persisted to DB."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="Failed",
            factor_code="code",
            ic=0.01,
            ir=0.5,
            sharpe=0.3,
            max_drawdown=0.25,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC],
            trial_id="t1",
        )

        with patch("iqfmp.db.models.PatternRecordORM") as mock_orm:
            pattern_memory_with_db.record_failure(
                hypothesis="Failed",
                factor_code="code",
                factor_family="value",
                feedback=feedback,
            )

            mock_orm.assert_called_once()
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

        _ = mock_vector_store

    def test_db_error_rollback(
        self, pattern_memory_with_db, mock_vector_store, mock_session
    ):
        """Test that DB errors trigger rollback."""
        mock_session.commit.side_effect = Exception("DB error")

        with patch("iqfmp.db.models.PatternRecordORM"):
            with pytest.raises(Exception, match="DB error"):
                pattern_memory_with_db.record_success(
                    hypothesis="Test",
                    factor_code="code",
                    factor_family="momentum",
                    metrics={"ic": 0.05},
                )

            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()

        _ = mock_vector_store

    def test_delete_pattern_with_db(
        self, pattern_memory_with_db, mock_vector_store, mock_session
    ):
        """Test delete_pattern calls both vector store and DB."""
        with patch("iqfmp.db.models.PatternRecordORM"):
            result = pattern_memory_with_db.delete_pattern("pattern-to-delete")

            # Vector store should be called
            mock_vector_store.delete_pattern.assert_called_once_with(
                pattern_id="pattern-to-delete",
                collection_name="test_patterns",
            )

            # DB delete should be called
            mock_session.query.return_value.filter.return_value.delete.assert_called()

            # Session should be committed and closed
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

            assert result is True

    def test_delete_pattern_db_error_logged(
        self, pattern_memory_with_db, mock_vector_store, mock_session
    ):
        """Test delete_pattern logs DB errors but returns vector store result."""
        mock_session.commit.side_effect = Exception("DB delete error")

        with patch("iqfmp.db.models.PatternRecordORM"):
            # Should not raise, just log and return vector store result
            result = pattern_memory_with_db.delete_pattern("pattern-to-delete")

            # Should still return vector store result
            assert result is True

            # Rollback should be called on error
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()

        _ = mock_vector_store


class TestPatternRecordValidation:
    """Tests for PatternRecord validation."""

    def test_empty_factor_code_raises(self):
        """Test that empty factor_code raises ValueError."""
        with pytest.raises(ValueError, match="factor_code cannot be empty"):
            PatternRecord(
                pattern_id="test-123",
                pattern_type="success",
                hypothesis="Test hypothesis",
                factor_code="",  # Empty
                factor_family="momentum",
                metrics={"ic": 0.05},
            )

    def test_whitespace_only_factor_code_raises(self):
        """Test that whitespace-only factor_code raises ValueError."""
        with pytest.raises(ValueError, match="factor_code cannot be empty"):
            PatternRecord(
                pattern_id="test-123",
                pattern_type="success",
                hypothesis="Test hypothesis",
                factor_code="   ",  # Whitespace only
                factor_family="momentum",
                metrics={"ic": 0.05},
            )

    def test_invalid_pattern_type_raises(self):
        """Test that invalid pattern_type raises ValueError."""
        with pytest.raises(ValueError, match="pattern_type must be"):
            PatternRecord(
                pattern_id="test-123",
                pattern_type="invalid",  # Not "success" or "failure"
                hypothesis="Test hypothesis",
                factor_code="Ref($close, -1)",
                factor_family="momentum",
                metrics={"ic": 0.05},
            )
