"""Intelligent Retry with Error Classification and Exponential Backoff.

This module provides a robust retry mechanism for LLM API calls with:
1. Error classification (retryable vs non-retryable)
2. Exponential backoff with jitter
3. Per-error-type retry limits
4. Circuit breaker pattern (optional)

Implements RD-Agent pattern for reliable LLM operations.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error classification for retry decisions."""

    # Retryable errors
    RATE_LIMIT = "rate_limit"  # 429 Too Many Requests
    SERVER_ERROR = "server_error"  # 5xx errors
    TIMEOUT = "timeout"  # Request timeout
    NETWORK = "network"  # Connection errors
    TRANSIENT = "transient"  # Temporary failures

    # Non-retryable errors
    AUTH = "auth"  # 401/403 Authentication/Authorization
    INVALID_REQUEST = "invalid_request"  # 400 Bad Request
    NOT_FOUND = "not_found"  # 404 Not Found
    CONTENT_FILTER = "content_filter"  # Content blocked by safety filter
    QUOTA_EXCEEDED = "quota_exceeded"  # Billing/quota issues
    UNKNOWN = "unknown"  # Unclassified errors


# Retryable categories
RETRYABLE_CATEGORIES = {
    ErrorCategory.RATE_LIMIT,
    ErrorCategory.SERVER_ERROR,
    ErrorCategory.TIMEOUT,
    ErrorCategory.NETWORK,
    ErrorCategory.TRANSIENT,
}


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # General settings
    max_retries: int = 3
    max_total_time: float = 120.0  # Maximum total retry time in seconds

    # Backoff settings
    initial_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay between retries
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: float = 0.1  # Jitter factor (0.1 = ±10%)

    # Per-category settings
    rate_limit_delay: float = 30.0  # Longer delay for rate limits
    server_error_delay: float = 5.0  # Moderate delay for server errors

    # Circuit breaker (optional)
    enable_circuit_breaker: bool = False
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Time before trying again


@dataclass
class RetryState:
    """State tracking for retry operations."""

    attempt: int = 0
    total_time: float = 0.0
    last_error: Optional[Exception] = None
    last_category: Optional[ErrorCategory] = None
    delays: list[float] = field(default_factory=list)

    def record_attempt(self, delay: float, error: Exception, category: ErrorCategory) -> None:
        """Record a retry attempt."""
        self.attempt += 1
        self.total_time += delay
        self.last_error = error
        self.last_category = category
        self.delays.append(delay)


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    value: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0
    error_category: Optional[ErrorCategory] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_time": self.total_time,
            "error_category": self.error_category.value if self.error_category else None,
            "error": str(self.error) if self.error else None,
        }


class ErrorClassifier:
    """Classify errors for retry decisions."""

    # Error message patterns for classification
    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "too many requests",
        "429",
        "quota exceeded",
        "requests per minute",
        "RPM limit",
    ]

    SERVER_ERROR_PATTERNS = [
        "500",
        "502",
        "503",
        "504",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    ]

    TIMEOUT_PATTERNS = [
        "timeout",
        "timed out",
        "request timeout",
        "read timeout",
        "connect timeout",
    ]

    NETWORK_PATTERNS = [
        "connection",
        "network",
        "dns",
        "socket",
        "refused",
        "reset",
        "unreachable",
    ]

    AUTH_PATTERNS = [
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "authentication",
        "invalid api key",
        "api key invalid",
    ]

    CONTENT_FILTER_PATTERNS = [
        "content filter",
        "content_filter",
        "safety",
        "blocked",
        "policy violation",
        "harmful content",
    ]

    @classmethod
    def classify(cls, error: Exception) -> ErrorCategory:
        """Classify an error into a category.

        Args:
            error: Exception to classify

        Returns:
            ErrorCategory for the error
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check status code if available
        status_code = getattr(error, "status_code", None)
        if status_code:
            if status_code == 429:
                return ErrorCategory.RATE_LIMIT
            if status_code in (500, 502, 503, 504):
                return ErrorCategory.SERVER_ERROR
            if status_code == 401:
                return ErrorCategory.AUTH
            if status_code == 403:
                return ErrorCategory.AUTH
            if status_code == 400:
                return ErrorCategory.INVALID_REQUEST
            if status_code == 404:
                return ErrorCategory.NOT_FOUND

        # Pattern matching
        if any(p in error_str for p in cls.RATE_LIMIT_PATTERNS):
            return ErrorCategory.RATE_LIMIT

        if any(p in error_str for p in cls.SERVER_ERROR_PATTERNS):
            return ErrorCategory.SERVER_ERROR

        if any(p in error_str for p in cls.TIMEOUT_PATTERNS):
            return ErrorCategory.TIMEOUT

        if any(p in error_str for p in cls.NETWORK_PATTERNS):
            return ErrorCategory.NETWORK

        if any(p in error_str for p in cls.AUTH_PATTERNS):
            return ErrorCategory.AUTH

        if any(p in error_str for p in cls.CONTENT_FILTER_PATTERNS):
            return ErrorCategory.CONTENT_FILTER

        # Type-based classification
        if "timeout" in error_type:
            return ErrorCategory.TIMEOUT
        if "connection" in error_type or "network" in error_type:
            return ErrorCategory.NETWORK

        return ErrorCategory.UNKNOWN

    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        """Check if an error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if error is retryable
        """
        category = cls.classify(error)
        return category in RETRYABLE_CATEGORIES


class BackoffCalculator:
    """Calculate backoff delays with jitter."""

    def __init__(self, config: RetryConfig):
        """Initialize backoff calculator.

        Args:
            config: Retry configuration
        """
        self.config = config

    def calculate_delay(
        self,
        attempt: int,
        category: ErrorCategory,
    ) -> float:
        """Calculate delay for next retry.

        Args:
            attempt: Current attempt number (0-based)
            category: Error category

        Returns:
            Delay in seconds
        """
        # Base exponential backoff
        delay = self.config.initial_delay * (self.config.exponential_base ** attempt)

        # Apply category-specific adjustments
        if category == ErrorCategory.RATE_LIMIT:
            delay = max(delay, self.config.rate_limit_delay)
        elif category == ErrorCategory.SERVER_ERROR:
            delay = max(delay, self.config.server_error_delay)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        jitter_range = delay * self.config.jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay + jitter)

        return delay


class CircuitBreaker:
    """Simple circuit breaker for failure protection."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Time to wait before recovery attempt
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._state == "open":
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = "half-open"
                    return False
            return True
        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                f"Circuit breaker opened after {self._failures} failures"
            )

    def reset(self) -> None:
        """Reset circuit breaker state."""
        self._failures = 0
        self._last_failure_time = None
        self._state = "closed"


class RetryHandler:
    """Handler for retry operations with error classification.

    Provides intelligent retry with:
    - Error classification
    - Exponential backoff with jitter
    - Optional circuit breaker
    - Detailed logging

    Example:
        handler = RetryHandler()

        # Retry async function
        result = await handler.execute_async(
            some_async_function,
            arg1, arg2,
            kwarg1=value1
        )

        if result.success:
            print(f"Success after {result.attempts} attempts")
        else:
            print(f"Failed: {result.error}")
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
    ):
        """Initialize retry handler.

        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()
        self.classifier = ErrorClassifier()
        self.backoff = BackoffCalculator(self.config)
        self.circuit_breaker: Optional[CircuitBreaker] = None

        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
            )

    async def execute_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute async function with retry.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with outcome details
        """
        state = RetryState()
        start_time = time.time()

        while state.attempt <= self.config.max_retries:
            # Check circuit breaker
            if self.circuit_breaker and self.circuit_breaker.is_open:
                logger.warning("Circuit breaker is open, failing fast")
                return RetryResult(
                    success=False,
                    error=Exception("Circuit breaker open"),
                    attempts=state.attempt,
                    total_time=time.time() - start_time,
                    error_category=ErrorCategory.TRANSIENT,
                )

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()

                return RetryResult(
                    success=True,
                    value=result,
                    attempts=state.attempt + 1,
                    total_time=time.time() - start_time,
                )

            except Exception as e:
                # Classify error
                category = self.classifier.classify(e)
                is_retryable = category in RETRYABLE_CATEGORIES

                logger.warning(
                    f"Attempt {state.attempt + 1} failed: {e} "
                    f"(category={category.value}, retryable={is_retryable})"
                )

                # Record failure
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()

                # Check if we should retry
                if not is_retryable:
                    logger.error(f"Non-retryable error: {e}")
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=state.attempt + 1,
                        total_time=time.time() - start_time,
                        error_category=category,
                    )

                # Check if we've exceeded max retries
                if state.attempt >= self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=state.attempt + 1,
                        total_time=time.time() - start_time,
                        error_category=category,
                    )

                # Calculate and apply delay
                delay = self.backoff.calculate_delay(state.attempt, category)

                # Check total time limit
                if state.total_time + delay > self.config.max_total_time:
                    logger.error(
                        f"Total retry time ({state.total_time + delay}s) "
                        f"would exceed limit ({self.config.max_total_time}s)"
                    )
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=state.attempt + 1,
                        total_time=time.time() - start_time,
                        error_category=category,
                    )

                logger.info(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)

                state.record_attempt(delay, e, category)

        # Should not reach here
        return RetryResult(
            success=False,
            error=state.last_error,
            attempts=state.attempt,
            total_time=time.time() - start_time,
            error_category=state.last_category,
        )

    def execute_sync(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute sync function with retry.

        Args:
            func: Sync function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with outcome details
        """
        state = RetryState()
        start_time = time.time()

        while state.attempt <= self.config.max_retries:
            # Check circuit breaker
            if self.circuit_breaker and self.circuit_breaker.is_open:
                return RetryResult(
                    success=False,
                    error=Exception("Circuit breaker open"),
                    attempts=state.attempt,
                    total_time=time.time() - start_time,
                    error_category=ErrorCategory.TRANSIENT,
                )

            try:
                result = func(*args, **kwargs)

                if self.circuit_breaker:
                    self.circuit_breaker.record_success()

                return RetryResult(
                    success=True,
                    value=result,
                    attempts=state.attempt + 1,
                    total_time=time.time() - start_time,
                )

            except Exception as e:
                category = self.classifier.classify(e)
                is_retryable = category in RETRYABLE_CATEGORIES

                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()

                if not is_retryable or state.attempt >= self.config.max_retries:
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=state.attempt + 1,
                        total_time=time.time() - start_time,
                        error_category=category,
                    )

                delay = self.backoff.calculate_delay(state.attempt, category)

                if state.total_time + delay > self.config.max_total_time:
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=state.attempt + 1,
                        total_time=time.time() - start_time,
                        error_category=category,
                    )

                time.sleep(delay)
                state.record_attempt(delay, e, category)

        return RetryResult(
            success=False,
            error=state.last_error,
            attempts=state.attempt,
            total_time=time.time() - start_time,
            error_category=state.last_category,
        )


# Decorator for easy use
T = TypeVar('T')


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    enable_circuit_breaker: bool = False,
) -> Callable:
    """Decorator for adding retry with backoff to async functions.

    Args:
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        enable_circuit_breaker: Enable circuit breaker

    Returns:
        Decorated function
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        enable_circuit_breaker=enable_circuit_breaker,
    )
    handler = RetryHandler(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            result = await handler.execute_async(func, *args, **kwargs)
            if result.success:
                return result.value
            raise result.error or Exception("Retry failed without error")
        return wrapper

    return decorator


# Convenience function for direct use
async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    **kwargs: Any,
) -> RetryResult:
    """Retry an async function with default settings.

    Args:
        func: Async function to retry
        *args: Positional arguments
        max_retries: Maximum retries
        initial_delay: Initial delay
        **kwargs: Keyword arguments

    Returns:
        RetryResult
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
    )
    handler = RetryHandler(config)
    return await handler.execute_async(func, *args, **kwargs)
