"""Human Review Gate for code approval.

This module provides a human review gate for LLM-generated code.
It serves as the third layer of the three-layer security architecture.

Security Layers:
1. AST Security Checker - Static analysis (pre-execution)
2. Sandbox Executor - Runtime isolation
3. Human Review Gate (this module) - Manual approval

Critical State Persistence:
Per CLAUDE.md, review decisions are critical state that must be persisted.
Use RedisReviewQueue for production deployments.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """Status of a review request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ReviewRequest:
    """A request for human review of generated code."""

    code: str
    code_summary: str
    factor_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "request_id": self.request_id,
            "code": self.code,
            "code_summary": self.code_summary,
            "factor_name": self.factor_name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority,
        }


@dataclass
class ReviewDecision:
    """Decision made on a review request."""

    request_id: str
    status: ReviewStatus
    reviewer: Optional[str] = None
    reason: Optional[str] = None
    decided_at: datetime = field(default_factory=datetime.now)

    def is_approved(self) -> bool:
        """Check if the decision is approved."""
        return self.status == ReviewStatus.APPROVED

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass
class ReviewConfig:
    """Configuration for the review gate."""

    timeout_seconds: int = 3600  # 1 hour default
    max_queue_size: int = 100
    auto_reject_on_timeout: bool = True


class NotifierBase(ABC):
    """Base class for notification implementations."""

    @abstractmethod
    async def send_review_request(self, request: ReviewRequest) -> bool:
        """Send notification about new review request.

        Args:
            request: The review request to notify about.

        Returns:
            True if notification was sent successfully.
        """
        pass

    @abstractmethod
    async def send_decision(self, decision: ReviewDecision) -> bool:
        """Send notification about review decision.

        Args:
            decision: The review decision to notify about.

        Returns:
            True if notification was sent successfully.
        """
        pass


class ReviewQueue:
    """Queue for managing review requests."""

    def __init__(self, max_size: int = 100) -> None:
        """Initialize the review queue.

        Args:
            max_size: Maximum number of pending requests.
        """
        self.max_size = max_size
        self._pending: dict[str, ReviewRequest] = {}
        self._decisions: dict[str, ReviewDecision] = {}
        self._decision_events: dict[str, asyncio.Event] = {}

    def add(self, request: ReviewRequest) -> None:
        """Add a request to the queue.

        Args:
            request: The review request to add.

        Raises:
            RuntimeError: If queue is full.
        """
        if len(self._pending) >= self.max_size:
            raise RuntimeError(f"Review queue is full (max: {self.max_size})")

        self._pending[request.request_id] = request
        self._decision_events[request.request_id] = asyncio.Event()

    def get_pending(self) -> list[ReviewRequest]:
        """Get all pending requests in FIFO order.

        Returns:
            List of pending review requests.
        """
        return sorted(
            self._pending.values(),
            key=lambda r: r.created_at,
        )

    def get_request(self, request_id: str) -> Optional[ReviewRequest]:
        """Get a specific request by ID.

        Args:
            request_id: The request ID to find.

        Returns:
            The review request or None if not found.
        """
        return self._pending.get(request_id)

    def get_status(self, request_id: str) -> Optional[ReviewStatus]:
        """Get status of a request.

        Args:
            request_id: The request ID to check.

        Returns:
            The review status or None if not found.
        """
        if request_id in self._pending:
            return ReviewStatus.PENDING
        if request_id in self._decisions:
            return self._decisions[request_id].status
        return None

    def record_decision(self, decision: ReviewDecision) -> None:
        """Record a decision and remove from pending.

        Args:
            decision: The decision to record.
        """
        if decision.request_id in self._pending:
            del self._pending[decision.request_id]

        self._decisions[decision.request_id] = decision

        # Signal any waiters
        if decision.request_id in self._decision_events:
            self._decision_events[decision.request_id].set()

    def get_decision(self, request_id: str) -> Optional[ReviewDecision]:
        """Get decision for a request.

        Args:
            request_id: The request ID to find.

        Returns:
            The review decision or None if not decided.
        """
        return self._decisions.get(request_id)

    def get_decision_event(self, request_id: str) -> Optional[asyncio.Event]:
        """Get event for waiting on decision.

        Args:
            request_id: The request ID to wait for.

        Returns:
            The asyncio Event or None if not found.
        """
        return self._decision_events.get(request_id)

    def get_decision_history(self, limit: int = 100) -> list[ReviewDecision]:
        """Get recent decision history.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            List of recent decisions.
        """
        decisions = sorted(
            self._decisions.values(),
            key=lambda d: d.decided_at,
            reverse=True,
        )
        return decisions[:limit]

    def get_timed_out_requests(self, timeout_seconds: int) -> list[ReviewRequest]:
        """Get requests that have timed out.

        Args:
            timeout_seconds: Timeout threshold in seconds.

        Returns:
            List of timed out requests.
        """
        now = datetime.now()
        timed_out = []
        for request in self._pending.values():
            age = (now - request.created_at).total_seconds()
            if age > timeout_seconds:
                timed_out.append(request)
        return timed_out


class ReviewQueueError(Exception):
    """Raised when review queue operations fail."""

    pass


class RedisReviewQueue(ReviewQueue):
    """Redis-backed queue for managing review requests.

    Critical state per CLAUDE.md: Review decisions must be persisted
    to survive service restarts. This class stores pending requests
    and decisions in Redis.

    Note: Decision events (asyncio.Event) are still in-memory for the
    current process, but the actual state is recovered from Redis on init.
    """

    REDIS_PENDING_KEY = "iqfmp:review:pending"
    REDIS_DECISIONS_KEY = "iqfmp:review:decisions"
    REDIS_PENDING_TTL = 86400 * 7  # 7 days for pending requests
    REDIS_DECISION_TTL = 86400 * 30  # 30 days for decisions

    def __init__(self, max_size: int = 100, redis_client: Any = None) -> None:
        """Initialize the Redis-backed review queue.

        Args:
            max_size: Maximum number of pending requests.
            redis_client: Optional Redis client for dependency injection.

        Raises:
            ReviewQueueError: If Redis is unavailable and no client injected.
        """
        super().__init__(max_size)
        self._redis = redis_client if redis_client is not None else self._get_redis_client()
        # Load existing state from Redis
        self._load_from_redis()

    def _get_redis_client(self) -> Any:
        """Get Redis client. Raises if unavailable."""
        try:
            from iqfmp.db import get_redis_client
            client = get_redis_client()
            if client is None:
                raise ReviewQueueError(
                    "Redis unavailable. Review queue requires persistent storage "
                    "per CLAUDE.md critical state rules."
                )
            return client
        except ReviewQueueError:
            raise
        except Exception as e:
            raise ReviewQueueError(f"Failed to connect to Redis: {e}") from e

    def _serialize_request(self, request: ReviewRequest) -> str:
        """Serialize ReviewRequest to JSON."""
        return json.dumps(request.to_dict())

    def _deserialize_request(self, data: str) -> ReviewRequest:
        """Deserialize JSON to ReviewRequest."""
        obj = json.loads(data)
        return ReviewRequest(
            code=obj["code"],
            code_summary=obj["code_summary"],
            factor_name=obj["factor_name"],
            metadata=obj.get("metadata", {}),
            request_id=obj["request_id"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            priority=obj.get("priority", 0),
        )

    def _serialize_decision(self, decision: ReviewDecision) -> str:
        """Serialize ReviewDecision to JSON."""
        return json.dumps(decision.to_dict())

    def _deserialize_decision(self, data: str) -> ReviewDecision:
        """Deserialize JSON to ReviewDecision."""
        obj = json.loads(data)
        return ReviewDecision(
            request_id=obj["request_id"],
            status=ReviewStatus(obj["status"]),
            reviewer=obj.get("reviewer"),
            reason=obj.get("reason"),
            decided_at=datetime.fromisoformat(obj["decided_at"]),
        )

    def _load_from_redis(self) -> None:
        """Load existing state from Redis on initialization."""
        try:
            # Load pending requests
            pending_data = self._redis.hgetall(self.REDIS_PENDING_KEY)
            for request_id, data in pending_data.items():
                try:
                    request_id_str = request_id.decode() if isinstance(request_id, bytes) else request_id
                    data_str = data.decode() if isinstance(data, bytes) else data
                    request = self._deserialize_request(data_str)
                    self._pending[request_id_str] = request
                    self._decision_events[request_id_str] = asyncio.Event()
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to deserialize pending request {request_id}: {e}")

            # Load decisions
            decisions_data = self._redis.hgetall(self.REDIS_DECISIONS_KEY)
            for request_id, data in decisions_data.items():
                try:
                    request_id_str = request_id.decode() if isinstance(request_id, bytes) else request_id
                    data_str = data.decode() if isinstance(data, bytes) else data
                    decision = self._deserialize_decision(data_str)
                    self._decisions[request_id_str] = decision
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to deserialize decision {request_id}: {e}")

            logger.info(
                f"Loaded review queue from Redis: {len(self._pending)} pending, "
                f"{len(self._decisions)} decisions"
            )
        except Exception as e:
            logger.error(f"Failed to load review queue from Redis: {e}")
            raise ReviewQueueError(f"Failed to load review queue from Redis: {e}") from e

    def add(self, request: ReviewRequest) -> None:
        """Add a request to the queue and persist to Redis."""
        if len(self._pending) >= self.max_size:
            raise RuntimeError(f"Review queue is full (max: {self.max_size})")

        try:
            # Persist to Redis first
            self._redis.hset(
                self.REDIS_PENDING_KEY,
                request.request_id,
                self._serialize_request(request),
            )
            # Then update in-memory
            self._pending[request.request_id] = request
            self._decision_events[request.request_id] = asyncio.Event()
            logger.debug(f"Added review request to Redis: {request.request_id}")
        except Exception as e:
            raise ReviewQueueError(
                f"Failed to persist review request to Redis: {e}. "
                "Critical state persistence required."
            ) from e

    def record_decision(self, decision: ReviewDecision) -> None:
        """Record a decision and persist to Redis."""
        try:
            # Persist decision to Redis
            self._redis.hset(
                self.REDIS_DECISIONS_KEY,
                decision.request_id,
                self._serialize_decision(decision),
            )
            # Remove from pending in Redis
            self._redis.hdel(self.REDIS_PENDING_KEY, decision.request_id)

            # Update in-memory state
            if decision.request_id in self._pending:
                del self._pending[decision.request_id]
            self._decisions[decision.request_id] = decision

            # Signal any waiters
            if decision.request_id in self._decision_events:
                self._decision_events[decision.request_id].set()

            logger.debug(f"Recorded decision to Redis: {decision.request_id} -> {decision.status.value}")
        except Exception as e:
            raise ReviewQueueError(
                f"Failed to persist decision to Redis: {e}. "
                "Critical state persistence required."
            ) from e


def get_review_queue(
    max_size: int = 100,
    use_redis: bool = True,
    redis_client: Any = None,
) -> ReviewQueue:
    """Get review queue with appropriate storage backend.

    Args:
        max_size: Maximum queue size.
        use_redis: If True (default), use Redis-backed queue for persistence.
        redis_client: Optional Redis client for dependency injection.

    Returns:
        ReviewQueue instance (Redis-backed in production, memory for testing).
    """
    if use_redis:
        try:
            return RedisReviewQueue(max_size=max_size, redis_client=redis_client)
        except ReviewQueueError as e:
            logger.warning(
                f"Redis review queue unavailable: {e}. "
                "Falling back to in-memory queue. "
                "WARNING: Review decisions will NOT persist across restarts!"
            )
    return ReviewQueue(max_size=max_size)


class HumanReviewGate:
    """Gate for human review of LLM-generated code.

    This class manages the review process including:
    - Submitting code for review
    - Notifying reviewers
    - Recording decisions
    - Handling timeouts

    Critical State: Uses Redis-backed queue by default for persistence.
    """

    def __init__(
        self,
        notifier: Optional[NotifierBase] = None,
        notifiers: Optional[list[NotifierBase]] = None,
        config: Optional[ReviewConfig] = None,
        on_request_callback: Optional[Callable] = None,
        queue: Optional[ReviewQueue] = None,
        use_redis: bool = True,
    ) -> None:
        """Initialize the review gate.

        Args:
            notifier: Single notifier instance.
            notifiers: List of notifier instances (for multiple channels).
            config: Review configuration.
            on_request_callback: Callback for new requests.
            queue: Optional custom queue (for testing).
            use_redis: If True (default), use Redis-backed queue for persistence.
        """
        self.config = config or ReviewConfig()
        self._queue = queue or get_review_queue(
            max_size=self.config.max_queue_size,
            use_redis=use_redis,
        )

        # Handle notifiers
        self._notifiers: list[NotifierBase] = []
        if notifier:
            self._notifiers.append(notifier)
        if notifiers:
            self._notifiers.extend(notifiers)

        self._on_request_callback = on_request_callback

    async def submit(self, request: ReviewRequest) -> str:
        """Submit code for human review.

        Args:
            request: The review request to submit.

        Returns:
            The request ID.

        Raises:
            RuntimeError: If queue is full.
        """
        # Add to queue
        self._queue.add(request)

        # Notify reviewers (don't fail if notification fails)
        for notifier in self._notifiers:
            try:
                await notifier.send_review_request(request)
            except Exception as e:
                logger.warning(
                    f"Failed to send review request notification via {type(notifier).__name__}: {e}"
                )

        # Call callback if set
        if self._on_request_callback:
            try:
                await self._on_request_callback(request)
            except Exception as e:
                logger.warning(f"Review request callback failed: {e}")

        return request.request_id

    async def decide(
        self,
        request_id: str,
        approved: bool,
        reviewer: str,
        reason: Optional[str] = None,
    ) -> ReviewDecision:
        """Record a review decision.

        Args:
            request_id: The request ID being decided.
            approved: Whether the code is approved.
            reviewer: Who made the decision.
            reason: Reason for rejection (if rejected).

        Returns:
            The recorded decision.

        Raises:
            ValueError: If request not found or already decided.
        """
        # Check if request exists
        request = self._queue.get_request(request_id)
        if request is None:
            # Check if already decided
            existing = self._queue.get_decision(request_id)
            if existing:
                raise ValueError(f"Request {request_id} already decided")
            raise ValueError(f"Request {request_id} not found")

        # Create decision
        status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        decision = ReviewDecision(
            request_id=request_id,
            status=status,
            reviewer=reviewer,
            reason=reason,
        )

        # Record decision
        self._queue.record_decision(decision)

        # Notify about decision
        for notifier in self._notifiers:
            try:
                await notifier.send_decision(decision)
            except Exception as e:
                logger.warning(
                    f"Failed to send decision notification via {type(notifier).__name__}: {e}"
                )

        return decision

    async def get_pending_requests(self) -> list[ReviewRequest]:
        """Get all pending review requests.

        Returns:
            List of pending requests in FIFO order.
        """
        return self._queue.get_pending()

    async def get_status(self, request_id: str) -> Optional[ReviewStatus]:
        """Get status of a review request.

        Args:
            request_id: The request ID to check.

        Returns:
            The review status or None if not found.
        """
        return self._queue.get_status(request_id)

    async def process_timeouts(self) -> list[ReviewDecision]:
        """Process timed out requests.

        Returns:
            List of timeout decisions made.
        """
        if not self.config.auto_reject_on_timeout:
            return []

        timed_out = self._queue.get_timed_out_requests(
            self.config.timeout_seconds
        )
        decisions = []

        for request in timed_out:
            decision = ReviewDecision(
                request_id=request.request_id,
                status=ReviewStatus.TIMEOUT,
            )
            self._queue.record_decision(decision)
            decisions.append(decision)

        return decisions

    async def wait_for_decision(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> ReviewDecision:
        """Wait for a decision on a request.

        Args:
            request_id: The request ID to wait for.
            timeout: Optional timeout in seconds.

        Returns:
            The decision when made.
        """
        # Check if already decided
        existing = self._queue.get_decision(request_id)
        if existing:
            return existing

        # Get event to wait on
        event = self._queue.get_decision_event(request_id)
        if event is None:
            # Request not found, return timeout
            return ReviewDecision(
                request_id=request_id,
                status=ReviewStatus.TIMEOUT,
            )

        # Wait for decision
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            decision = self._queue.get_decision(request_id)
            if decision:
                return decision
        except asyncio.TimeoutError:
            pass

        # Timeout - create timeout decision
        decision = ReviewDecision(
            request_id=request_id,
            status=ReviewStatus.TIMEOUT,
        )
        self._queue.record_decision(decision)
        return decision

    async def get_decision_history(
        self, limit: int = 100
    ) -> list[ReviewDecision]:
        """Get recent decision history.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            List of recent decisions.
        """
        return self._queue.get_decision_history(limit=limit)
