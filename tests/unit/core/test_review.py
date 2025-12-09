"""Tests for Human Review Gate.

Six-dimensional test coverage:
1. Functional: Core review gate functionality
2. Boundary: Edge cases and limits
3. Exception: Error handling
4. Performance: Queue and timeout handling
5. Security: Review decision integrity
6. Compatibility: Different notifier implementations
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from iqfmp.core.review import (
    HumanReviewGate,
    ReviewRequest,
    ReviewDecision,
    ReviewStatus,
    ReviewConfig,
    NotifierBase,
    ReviewQueue,
)


class TestReviewRequestModel:
    """Test ReviewRequest data model."""

    def test_request_creation(self) -> None:
        """Test creating a review request."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="test_factor",
        )
        assert request.code == "result = 1 + 2"
        assert request.code_summary == "Simple arithmetic"
        assert request.factor_name == "test_factor"
        assert request.request_id is not None
        assert request.created_at is not None

    def test_request_with_metadata(self) -> None:
        """Test creating request with metadata."""
        request = ReviewRequest(
            code="result = x * y",
            code_summary="Multiplication",
            factor_name="multiply_factor",
            metadata={"author": "LLM", "version": "1.0"},
        )
        assert request.metadata["author"] == "LLM"
        assert request.metadata["version"] == "1.0"

    def test_request_id_uniqueness(self) -> None:
        """Test that request IDs are unique."""
        request1 = ReviewRequest(
            code="code1",
            code_summary="summary1",
            factor_name="factor1",
        )
        request2 = ReviewRequest(
            code="code2",
            code_summary="summary2",
            factor_name="factor2",
        )
        assert request1.request_id != request2.request_id


class TestReviewDecisionModel:
    """Test ReviewDecision data model."""

    def test_decision_approved(self) -> None:
        """Test approved decision."""
        decision = ReviewDecision(
            request_id="req-123",
            status=ReviewStatus.APPROVED,
            reviewer="admin",
        )
        assert decision.status == ReviewStatus.APPROVED
        assert decision.reviewer == "admin"
        assert decision.is_approved()

    def test_decision_rejected(self) -> None:
        """Test rejected decision."""
        decision = ReviewDecision(
            request_id="req-123",
            status=ReviewStatus.REJECTED,
            reviewer="admin",
            reason="Code logic incorrect",
        )
        assert decision.status == ReviewStatus.REJECTED
        assert decision.reason == "Code logic incorrect"
        assert not decision.is_approved()

    def test_decision_timeout(self) -> None:
        """Test timeout decision."""
        decision = ReviewDecision(
            request_id="req-123",
            status=ReviewStatus.TIMEOUT,
        )
        assert decision.status == ReviewStatus.TIMEOUT
        assert not decision.is_approved()

    def test_review_status_values(self) -> None:
        """Test all review status values exist."""
        assert ReviewStatus.PENDING is not None
        assert ReviewStatus.APPROVED is not None
        assert ReviewStatus.REJECTED is not None
        assert ReviewStatus.TIMEOUT is not None


class TestReviewConfigModel:
    """Test ReviewConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReviewConfig()
        assert config.timeout_seconds > 0
        assert config.max_queue_size > 0
        assert config.auto_reject_on_timeout is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ReviewConfig(
            timeout_seconds=300,
            max_queue_size=50,
            auto_reject_on_timeout=False,
        )
        assert config.timeout_seconds == 300
        assert config.max_queue_size == 50
        assert config.auto_reject_on_timeout is False


class MockNotifier(NotifierBase):
    """Mock notifier for testing."""

    def __init__(self) -> None:
        self.notifications: list[dict] = []
        self.should_fail = False

    async def send_review_request(self, request: ReviewRequest) -> bool:
        """Mock send review request."""
        if self.should_fail:
            return False
        self.notifications.append({
            "type": "request",
            "request_id": request.request_id,
            "factor_name": request.factor_name,
        })
        return True

    async def send_decision(self, decision: ReviewDecision) -> bool:
        """Mock send decision notification."""
        if self.should_fail:
            return False
        self.notifications.append({
            "type": "decision",
            "request_id": decision.request_id,
            "status": decision.status.value,
        })
        return True


class TestHumanReviewGateFunctional:
    """Functional tests for HumanReviewGate."""

    @pytest.fixture
    def notifier(self) -> MockNotifier:
        """Create a mock notifier."""
        return MockNotifier()

    @pytest.fixture
    def gate(self, notifier: MockNotifier) -> HumanReviewGate:
        """Create a review gate instance."""
        return HumanReviewGate(notifier=notifier)

    # === Basic Operations ===

    @pytest.mark.asyncio
    async def test_submit_review_request(
        self, gate: HumanReviewGate, notifier: MockNotifier
    ) -> None:
        """Test submitting a review request."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="add_factor",
        )
        request_id = await gate.submit(request)
        assert request_id == request.request_id
        assert len(notifier.notifications) == 1
        assert notifier.notifications[0]["type"] == "request"

    @pytest.mark.asyncio
    async def test_approve_request(
        self, gate: HumanReviewGate
    ) -> None:
        """Test approving a review request."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="add_factor",
        )
        await gate.submit(request)

        decision = await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="admin",
        )
        assert decision.status == ReviewStatus.APPROVED
        assert decision.is_approved()

    @pytest.mark.asyncio
    async def test_reject_request(
        self, gate: HumanReviewGate
    ) -> None:
        """Test rejecting a review request."""
        request = ReviewRequest(
            code="result = dangerous_operation()",
            code_summary="Dangerous operation",
            factor_name="bad_factor",
        )
        await gate.submit(request)

        decision = await gate.decide(
            request_id=request.request_id,
            approved=False,
            reviewer="admin",
            reason="Code is unsafe",
        )
        assert decision.status == ReviewStatus.REJECTED
        assert decision.reason == "Code is unsafe"
        assert not decision.is_approved()

    @pytest.mark.asyncio
    async def test_get_pending_requests(
        self, gate: HumanReviewGate
    ) -> None:
        """Test getting pending requests."""
        # Submit multiple requests
        for i in range(3):
            request = ReviewRequest(
                code=f"code_{i}",
                code_summary=f"summary_{i}",
                factor_name=f"factor_{i}",
            )
            await gate.submit(request)

        pending = await gate.get_pending_requests()
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_get_request_status(
        self, gate: HumanReviewGate
    ) -> None:
        """Test getting request status."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="add_factor",
        )
        await gate.submit(request)

        status = await gate.get_status(request.request_id)
        assert status == ReviewStatus.PENDING

        await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="admin",
        )

        status = await gate.get_status(request.request_id)
        assert status == ReviewStatus.APPROVED


class TestHumanReviewGateSecurity:
    """Security tests for review decision integrity."""

    @pytest.fixture
    def gate(self) -> HumanReviewGate:
        return HumanReviewGate(notifier=MockNotifier())

    @pytest.mark.asyncio
    async def test_cannot_decide_nonexistent_request(
        self, gate: HumanReviewGate
    ) -> None:
        """Test that deciding on nonexistent request fails."""
        with pytest.raises(ValueError, match="not found"):
            await gate.decide(
                request_id="nonexistent-id",
                approved=True,
                reviewer="admin",
            )

    @pytest.mark.asyncio
    async def test_cannot_decide_twice(
        self, gate: HumanReviewGate
    ) -> None:
        """Test that double decision is prevented."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="add_factor",
        )
        await gate.submit(request)

        # First decision
        await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="admin",
        )

        # Second decision should fail
        with pytest.raises(ValueError, match="already decided"):
            await gate.decide(
                request_id=request.request_id,
                approved=False,
                reviewer="admin2",
            )

    @pytest.mark.asyncio
    async def test_decision_records_reviewer(
        self, gate: HumanReviewGate
    ) -> None:
        """Test that reviewer is properly recorded."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="add_factor",
        )
        await gate.submit(request)

        decision = await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="security_admin",
        )
        assert decision.reviewer == "security_admin"
        assert decision.decided_at is not None

    @pytest.mark.asyncio
    async def test_decision_records_timestamp(
        self, gate: HumanReviewGate
    ) -> None:
        """Test that decision timestamp is recorded."""
        request = ReviewRequest(
            code="result = 1 + 2",
            code_summary="Simple arithmetic",
            factor_name="add_factor",
        )
        await gate.submit(request)

        before = datetime.now()
        decision = await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="admin",
        )
        after = datetime.now()

        assert before <= decision.decided_at <= after


class TestHumanReviewGateBoundary:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def gate(self) -> HumanReviewGate:
        return HumanReviewGate(
            notifier=MockNotifier(),
            config=ReviewConfig(max_queue_size=3),
        )

    @pytest.mark.asyncio
    async def test_empty_queue(self, gate: HumanReviewGate) -> None:
        """Test getting pending from empty queue."""
        pending = await gate.get_pending_requests()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_queue_at_capacity(
        self, gate: HumanReviewGate
    ) -> None:
        """Test queue at maximum capacity."""
        # Fill the queue
        for i in range(3):
            request = ReviewRequest(
                code=f"code_{i}",
                code_summary=f"summary_{i}",
                factor_name=f"factor_{i}",
            )
            await gate.submit(request)

        # Fourth request should fail
        request = ReviewRequest(
            code="overflow",
            code_summary="overflow",
            factor_name="overflow",
        )
        with pytest.raises(RuntimeError, match="queue.*full"):
            await gate.submit(request)

    @pytest.mark.asyncio
    async def test_long_code_summary(
        self, gate: HumanReviewGate
    ) -> None:
        """Test handling long code summary."""
        long_summary = "A" * 10000
        request = ReviewRequest(
            code="result = 1",
            code_summary=long_summary,
            factor_name="long_factor",
        )
        request_id = await gate.submit(request)
        assert request_id is not None

    @pytest.mark.asyncio
    async def test_empty_code(self, gate: HumanReviewGate) -> None:
        """Test submitting empty code."""
        request = ReviewRequest(
            code="",
            code_summary="Empty code",
            factor_name="empty_factor",
        )
        request_id = await gate.submit(request)
        assert request_id is not None


class TestHumanReviewGateException:
    """Exception handling tests."""

    @pytest.mark.asyncio
    async def test_notifier_failure_on_submit(self) -> None:
        """Test handling notifier failure on submit."""
        notifier = MockNotifier()
        notifier.should_fail = True
        gate = HumanReviewGate(notifier=notifier)

        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )

        # Submit should still succeed but log warning
        request_id = await gate.submit(request)
        assert request_id is not None

    @pytest.mark.asyncio
    async def test_get_status_nonexistent(self) -> None:
        """Test getting status of nonexistent request."""
        gate = HumanReviewGate(notifier=MockNotifier())
        status = await gate.get_status("nonexistent-id")
        assert status is None

    @pytest.mark.asyncio
    async def test_invalid_request_id_format(self) -> None:
        """Test handling invalid request ID."""
        gate = HumanReviewGate(notifier=MockNotifier())

        with pytest.raises(ValueError, match="not found"):
            await gate.decide(
                request_id="",
                approved=True,
                reviewer="admin",
            )


class TestHumanReviewGatePerformance:
    """Performance tests for timeout and queue handling."""

    @pytest.fixture
    def fast_timeout_gate(self) -> HumanReviewGate:
        """Create a gate with short timeout for testing."""
        return HumanReviewGate(
            notifier=MockNotifier(),
            config=ReviewConfig(
                timeout_seconds=1,
                auto_reject_on_timeout=True,
            ),
        )

    @pytest.mark.asyncio
    async def test_timeout_auto_reject(
        self, fast_timeout_gate: HumanReviewGate
    ) -> None:
        """Test that requests auto-reject on timeout."""
        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        await fast_timeout_gate.submit(request)

        # Wait for timeout
        await asyncio.sleep(1.5)

        # Check for timeout processing
        await fast_timeout_gate.process_timeouts()

        status = await fast_timeout_gate.get_status(request.request_id)
        assert status == ReviewStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_wait_for_decision_approved(self) -> None:
        """Test waiting for decision with approval."""
        gate = HumanReviewGate(
            notifier=MockNotifier(),
            config=ReviewConfig(timeout_seconds=10),
        )

        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        await gate.submit(request)

        # Simulate async decision
        async def approve_later():
            await asyncio.sleep(0.1)
            await gate.decide(
                request_id=request.request_id,
                approved=True,
                reviewer="admin",
            )

        asyncio.create_task(approve_later())

        # Wait for decision
        decision = await gate.wait_for_decision(
            request_id=request.request_id,
            timeout=5,
        )
        assert decision.status == ReviewStatus.APPROVED

    @pytest.mark.asyncio
    async def test_wait_for_decision_timeout(self) -> None:
        """Test waiting for decision with timeout."""
        gate = HumanReviewGate(
            notifier=MockNotifier(),
            config=ReviewConfig(timeout_seconds=10),
        )

        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        await gate.submit(request)

        # Wait for decision with short timeout
        decision = await gate.wait_for_decision(
            request_id=request.request_id,
            timeout=0.5,
        )
        assert decision.status == ReviewStatus.TIMEOUT


class TestHumanReviewGateCompatibility:
    """Compatibility tests for different notifier implementations."""

    @pytest.mark.asyncio
    async def test_no_notifier(self) -> None:
        """Test gate works without notifier."""
        gate = HumanReviewGate(notifier=None)

        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        request_id = await gate.submit(request)
        assert request_id is not None

        decision = await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="admin",
        )
        assert decision.is_approved()

    @pytest.mark.asyncio
    async def test_multiple_notifiers(self) -> None:
        """Test gate with multiple notifiers."""
        notifier1 = MockNotifier()
        notifier2 = MockNotifier()
        gate = HumanReviewGate(notifiers=[notifier1, notifier2])

        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        await gate.submit(request)

        # Both notifiers should receive the notification
        assert len(notifier1.notifications) == 1
        assert len(notifier2.notifications) == 1

    @pytest.mark.asyncio
    async def test_callback_notifier(self) -> None:
        """Test gate with callback-based decision."""
        received_requests: list[ReviewRequest] = []

        async def callback(request: ReviewRequest) -> None:
            received_requests.append(request)

        gate = HumanReviewGate(
            notifier=MockNotifier(),
            on_request_callback=callback,
        )

        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        await gate.submit(request)

        assert len(received_requests) == 1
        assert received_requests[0].request_id == request.request_id


class TestReviewQueueIntegration:
    """Integration tests for review queue."""

    @pytest.fixture
    def gate(self) -> HumanReviewGate:
        return HumanReviewGate(notifier=MockNotifier())

    @pytest.mark.asyncio
    async def test_queue_ordering(self, gate: HumanReviewGate) -> None:
        """Test that queue maintains FIFO order."""
        request_ids = []
        for i in range(3):
            request = ReviewRequest(
                code=f"code_{i}",
                code_summary=f"summary_{i}",
                factor_name=f"factor_{i}",
            )
            request_id = await gate.submit(request)
            request_ids.append(request_id)

        pending = await gate.get_pending_requests()
        pending_ids = [r.request_id for r in pending]

        assert pending_ids == request_ids

    @pytest.mark.asyncio
    async def test_completed_removed_from_pending(
        self, gate: HumanReviewGate
    ) -> None:
        """Test that completed requests are removed from pending."""
        request = ReviewRequest(
            code="result = 1",
            code_summary="Test",
            factor_name="test",
        )
        await gate.submit(request)

        assert len(await gate.get_pending_requests()) == 1

        await gate.decide(
            request_id=request.request_id,
            approved=True,
            reviewer="admin",
        )

        assert len(await gate.get_pending_requests()) == 0

    @pytest.mark.asyncio
    async def test_get_decision_history(
        self, gate: HumanReviewGate
    ) -> None:
        """Test retrieving decision history."""
        # Submit and decide multiple requests
        for i in range(3):
            request = ReviewRequest(
                code=f"code_{i}",
                code_summary=f"summary_{i}",
                factor_name=f"factor_{i}",
            )
            await gate.submit(request)
            await gate.decide(
                request_id=request.request_id,
                approved=(i % 2 == 0),
                reviewer="admin",
            )

        history = await gate.get_decision_history(limit=10)
        assert len(history) == 3
