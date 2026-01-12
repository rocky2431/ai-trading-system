"""LLM Provider for OpenRouter API integration.

This module provides a unified interface for accessing multiple LLM models
through OpenRouter, with support for:
- Multi-model switching (DeepSeek, Claude, GPT)
- Fallback chains for reliability
- Request caching for efficiency (in-memory + persistent Redis/PostgreSQL)
- Rate limiting for cost control
- Auto-continue for truncated responses
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

from iqfmp.llm.cache import PromptCache, PromptCacheStats, get_prompt_cache
from iqfmp.llm.retry import (
    ErrorCategory,
    ErrorClassifier,
    RetryConfig,
    RetryHandler,
    RetryResult,
)
from iqfmp.llm.validation import (
    JSONSchemaValidator,
    OutputType,
    SchemaValidationResult,
)


# === Error Classes ===

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class ModelNotAvailableError(LLMError):
    """Raised when requested model is not available."""
    pass


# === Enums ===

class ModelType(str, Enum):
    """Supported LLM model types."""
    DEEPSEEK_V3 = "deepseek-v3"
    DEEPSEEK_V3_SPECIAL = "deepseek-v3-special"
    CLAUDE_35_SONNET = "claude-3.5-sonnet"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_HAIKU = "claude-3-haiku"


# Model ID mapping for OpenRouter
MODEL_ID_MAP: dict[ModelType, str] = {
    ModelType.DEEPSEEK_V3: "deepseek/deepseek-chat",
    ModelType.DEEPSEEK_V3_SPECIAL: "deepseek/deepseek-v3.2-speciale",
    ModelType.CLAUDE_35_SONNET: "anthropic/claude-3.5-sonnet",
    ModelType.GPT_4O: "openai/gpt-4o",
    ModelType.GPT_4O_MINI: "openai/gpt-4o-mini",
    ModelType.CLAUDE_3_HAIKU: "anthropic/claude-3-haiku-20240307",
}


# === Data Classes ===

@dataclass
class LLMResponse:
    """Response from LLM API call.

    Extended with metadata for auto-continue and debugging support.
    P2 Fix: Added prompt_id and prompt_version for tracking.
    """
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: Optional[float] = None
    cached: bool = False
    # New fields for RD-Agent parity
    finish_reason: Optional[str] = None  # "stop" | "length" | "content_filter"
    raw_response: Optional[dict] = None  # Original API response (for debugging)
    model_id: Optional[str] = None  # Actual OpenRouter model ID used
    cost_estimate: Optional[float] = None  # Estimated cost in USD
    request_id: Optional[str] = None  # Request tracking ID
    # P2 Fix: Prompt version tracking
    prompt_id: Optional[str] = None  # Prompt template identifier
    prompt_version: Optional[str] = None  # Prompt template version

    @property
    def is_truncated(self) -> bool:
        """Check if response was truncated due to length."""
        return self.finish_reason == "length"

    @property
    def needs_continuation(self) -> bool:
        """Check if response needs auto-continuation."""
        return self.finish_reason == "length"


@dataclass
class FallbackChain:
    """Configuration for model fallback chain."""
    models: list[ModelType]
    max_retries: int = 3

    def get_models(self) -> list[ModelType]:
        """Get ordered list of fallback models."""
        return self.models.copy()


@dataclass
class LLMConfig:
    """Configuration for LLM Provider."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: ModelType = ModelType.DEEPSEEK_V3
    timeout: int = 60
    fallback_chain: Optional[FallbackChain] = None
    cache_enabled: bool = False
    cache_ttl: int = 300  # seconds

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API key is required")

        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid base URL")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables.

        Supports loading from .env file via pydantic-settings.
        """
        from dotenv import load_dotenv
        load_dotenv()  # Load .env file

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        default_model_str = os.getenv("OPENROUTER_DEFAULT_MODEL", "deepseek-v3-special")

        # Map string to ModelType
        model_map = {m.value: m for m in ModelType}
        default_model = model_map.get(default_model_str, ModelType.DEEPSEEK_V3_SPECIAL)

        return cls(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
        )

    def to_log_safe_dict(self) -> dict[str, Any]:
        """Return config dict with masked sensitive values."""
        return {
            "api_key": "****",
            "base_url": self.base_url,
            "default_model": self.default_model.value,
            "timeout": self.timeout,
            "cache_enabled": self.cache_enabled,
        }


# === Rate Limiter ===

def _deduplicate_candidates(candidates: list[str]) -> list[str]:
    """Remove duplicate candidates based on normalized content.

    Args:
        candidates: List of candidate strings

    Returns:
        Deduplicated list preserving original order
    """
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        normalized = c.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(c)
    return unique


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._request_timestamps: list[float] = []
        self._tokens_used: int = 0
        self._token_reset_time: float = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()

            # Clean up old timestamps (older than 1 minute)
            self._request_timestamps = [
                ts for ts in self._request_timestamps if now - ts < 60
            ]

            # Check if we're at the limit
            if len(self._request_timestamps) >= self.requests_per_minute:
                # Wait until oldest request expires
                wait_time = 60 - (now - self._request_timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self._request_timestamps.append(time.time())
            return True

    def record_usage(self, tokens: int = 0) -> None:
        """Record token usage."""
        now = time.time()

        # Reset token counter if minute has passed
        if now - self._token_reset_time >= 60:
            self._tokens_used = 0
            self._token_reset_time = now

        self._tokens_used += tokens

    def get_tokens_used(self) -> int:
        """Get tokens used in current window."""
        return self._tokens_used

    def get_remaining_requests(self) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        active = len([ts for ts in self._request_timestamps if now - ts < 60])
        return max(0, self.requests_per_minute - active)


# === LLM Provider ===

class LLMProvider:
    """Unified LLM provider for OpenRouter API."""

    def __init__(
        self,
        config: LLMConfig,
        use_persistent_cache: bool = True,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://iqfmp.io",
                "X-Title": "IQFMP",
            },
        )
        # In-memory cache for fast lookups within session
        self._cache: dict[str, tuple[LLMResponse, float]] = {}
        # Persistent two-tier cache for cross-session deduplication
        # L1: Redis (hot cache, ~1ms latency)
        # L2: PostgreSQL (persistent, ~10ms latency)
        self._use_persistent_cache = use_persistent_cache
        self._persistent_cache: Optional[PromptCache] = None
        if use_persistent_cache:
            try:
                self._persistent_cache = get_prompt_cache()
            except Exception as e:
                # Fallback to in-memory only if cache init fails
                logger.warning(f"Persistent cache init failed, using in-memory only: {e}")
                self._persistent_cache = None
        self._rate_limiter = RateLimiter()
        # Retry handler with error classification and exponential backoff
        self._retry_config = retry_config or RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=60.0,
            rate_limit_delay=30.0,
            server_error_delay=5.0,
        )
        self._retry_handler = RetryHandler(self._retry_config)
        self._error_classifier = ErrorClassifier()
        self._usage_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
            "retried_requests": 0,
            "failed_requests": 0,
        }

    def __repr__(self) -> str:
        """Return string representation without sensitive data."""
        return f"LLMProvider(model={self._config.default_model.value})"

    def __str__(self) -> str:
        """Return string representation without sensitive data."""
        return f"LLMProvider(model={self._config.default_model.value})"

    async def complete(
        self,
        prompt: str,
        model: Optional[ModelType | str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        auto_continue: bool = True,
        max_continue_rounds: int = 5,
        n_candidates: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        P2 Fix: Added n_candidates and seed for multi-candidate generation.

        Args:
            prompt: The input prompt.
            model: Optional model to use - can be ModelType enum or OpenRouter model ID string
                   (e.g., "deepseek/deepseek-coder-v3"). Defaults to config default.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt to guide the model.
            auto_continue: If True, automatically continue truncated responses (default True).
            max_continue_rounds: Maximum continuation rounds for auto_continue (default 5).
            n_candidates: Number of candidates to generate (default 1, max 10).
            seed: Random seed for reproducibility (optional).

        Returns:
            LLMResponse with generated content (first candidate if n_candidates > 1).
            Use generate_candidates() for accessing all candidates.

        Raises:
            ValueError: If prompt is empty.
            LLMError: On API errors.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Optional tracing / metadata (best-effort)
        execution_id = kwargs.pop("execution_id", None)
        conversation_id = kwargs.pop("conversation_id", None)
        agent = kwargs.pop("agent", None)
        prompt_id = kwargs.pop("prompt_id", None)
        prompt_version = kwargs.pop("prompt_version", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # P2 Fix: Add n_candidates and seed to kwargs
        if n_candidates > 1:
            kwargs["n"] = min(n_candidates, 10)  # Cap at 10
        if seed is not None:
            kwargs["seed"] = seed

        if auto_continue:
            response = await self._execute_with_auto_continue(
                messages=messages,
                model=model,
                max_continue_rounds=max_continue_rounds,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            response = await self._execute_with_fallback(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        # Attach prompt version info for replay/debugging.
        response.prompt_id = str(prompt_id) if prompt_id is not None else None
        response.prompt_version = str(prompt_version) if prompt_version is not None else None

        # Best-effort trace persistence (execution_id is the primary key).
        if execution_id is not None:
            try:
                from iqfmp.llm.trace import get_llm_trace_store, LLMTraceRecord

                exec_id = str(execution_id)
                conv_id = str(conversation_id) if conversation_id is not None else exec_id
                record = LLMTraceRecord.now(
                    execution_id=exec_id,
                    conversation_id=conv_id,
                    agent=str(agent) if agent is not None else None,
                    model=response.model,
                    prompt_id=response.prompt_id,
                    prompt_version=response.prompt_version,
                    messages=[{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages],
                    response=response.content,
                    usage={k: int(v) for k, v in (response.usage or {}).items()},
                    cached=bool(response.cached),
                    cost_estimate=response.cost_estimate,
                    request_id=response.request_id,
                )
                await get_llm_trace_store().record(record)
            except Exception as e:
                # Log but don't fail the main operation - trace is for audit, not critical path
                logger.warning(f"Failed to record LLM trace for cost tracking: {e}")

        return response

    async def generate_candidates(
        self,
        prompt: str,
        n_candidates: int = 3,
        model: Optional[ModelType | str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        deduplicate: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Generate multiple candidate responses for the same prompt.

        P2 Fix: New method for multi-candidate generation with deduplication.

        Args:
            prompt: The input prompt.
            n_candidates: Number of candidates to generate (default 3, max 10).
            model: Optional model to use.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature (higher = more diverse).
            system_prompt: Optional system prompt.
            seed: Random seed for reproducibility.
            deduplicate: If True, remove duplicate responses (default True).

        Returns:
            List of generated response strings.
        """
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")

        n_candidates = min(n_candidates, 10)  # Cap at 10

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build kwargs with n and seed
        api_kwargs = {**kwargs, "n": n_candidates}
        if seed is not None:
            api_kwargs["seed"] = seed

        # Call API with n_candidates
        try:
            response = await self._call_api_multi(
                messages=messages,
                model=model or self._config.default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                **api_kwargs,
            )

            candidates = response if isinstance(response, list) else [response.content]

            if deduplicate:
                candidates = _deduplicate_candidates(candidates)

            return candidates

        except Exception as e:
            # Fallback: generate sequentially
            logger.warning(f"Multi-candidate API failed: {e}. Falling back to sequential generation.")
            candidates = []
            failed_count = 0
            for i in range(n_candidates):
                try:
                    current_seed = seed + i if seed is not None else None
                    if current_seed is not None:
                        api_kwargs["seed"] = current_seed

                    resp = await self._execute_with_fallback(
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **{k: v for k, v in api_kwargs.items() if k != "n"},
                    )
                    candidates.append(resp.content)
                except Exception as candidate_error:
                    failed_count += 1
                    logger.warning(
                        f"Candidate {i + 1}/{n_candidates} generation failed: {candidate_error}"
                    )

            # Log summary of failures
            if failed_count > 0:
                logger.warning(
                    f"Candidate generation: {failed_count}/{n_candidates} failed, "
                    f"{len(candidates)} candidates generated"
                )

            # If all candidates failed, raise an error instead of returning empty list
            if not candidates and n_candidates > 0:
                raise LLMError(
                    f"All {n_candidates} candidate generations failed. "
                    "Strategy selection may be suboptimal."
                )

            if deduplicate:
                candidates = _deduplicate_candidates(candidates)

            return candidates

    async def _call_api_multi(
        self,
        messages: list[dict[str, str]],
        model: ModelType | str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[str]:
        """Make API call with n > 1 to get multiple candidates.

        P2 Fix: Internal method for multi-candidate API calls.
        """
        await self._rate_limiter.acquire()

        model_id = self.get_model_id(model)
        n = kwargs.pop("n", 1)
        seed = kwargs.pop("seed", None)

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "n": n,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed

        start_time = time.time()

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )

            if response.status_code != 200:
                raise LLMError(f"API Error: {response.status_code} - {response.text}")

            data = response.json()
            choices = data.get("choices", [])

            # Extract all candidate responses
            candidates = [choice["message"]["content"] for choice in choices]

            # Track usage
            usage = data.get("usage", {})
            self.record_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Multi-candidate API call: n={n}, got {len(candidates)} candidates in {latency_ms:.0f}ms")

            return candidates

        except Exception as e:
            raise LLMError(f"Multi-candidate API call failed: {e}")

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[ModelType | str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        auto_continue: bool = True,
        max_continue_rounds: int = 5,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            messages: List of chat messages.
            model: Optional model to use - can be ModelType enum or OpenRouter model ID string.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            auto_continue: If True, automatically continue truncated responses (default True).
            max_continue_rounds: Maximum continuation rounds for auto_continue (default 5).

        Returns:
            LLMResponse with generated content.
        """
        if auto_continue:
            return await self._execute_with_auto_continue(
                messages=messages,
                model=model,
                max_continue_rounds=max_continue_rounds,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            return await self._execute_with_fallback(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

    async def _execute_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: Optional[ModelType | str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute request with fallback chain if configured.

        Args:
            messages: Chat messages.
            model: Model to use - can be ModelType enum or OpenRouter model ID string.
        """
        # Determine target model (direct OpenRouter model ID string OR ModelType enum)
        target_model: ModelType | str = model or self._config.default_model
        model_str = self.get_model_id(target_model)
        temperature = kwargs.get("temperature", 0.7)

        # Check in-memory cache first
        cache_key = ""
        if self._config.cache_enabled:
            cache_key = self._generate_cache_key(
                str(messages),
                model_str,
                kwargs.get("max_tokens"),
            )
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached

        # Check persistent cache (cross-session)
        if self._use_persistent_cache:
            cached_content = self._get_from_persistent_cache(
                messages=messages,
                model=model_str,
                temperature=temperature,
            )
            if cached_content:
                # Return cached response with cached=True flag
                cached_response = LLMResponse(
                    content=cached_content,
                    model=model_str,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    cached=True,
                    finish_reason="stop",
                )
                # Also add to in-memory cache for faster subsequent lookups
                if self._config.cache_enabled and cache_key:
                    self._save_to_cache(cache_key, cached_response)
                return cached_response

        # Build fallback chain:
        # - ModelType enums: use fallback_chain if configured
        # - direct OpenRouter model ID strings: no fallback chain
        if isinstance(target_model, str) and not isinstance(target_model, ModelType):
            models_to_try: list[ModelType | str] = [target_model]
        elif self._config.fallback_chain:
            models_to_try = list(self._config.fallback_chain.get_models())
            if target_model not in models_to_try:
                models_to_try.insert(0, target_model)
        else:
            models_to_try = [target_model]

        last_error: Optional[Exception] = None
        last_category: Optional[ErrorCategory] = None

        for attempt_model in models_to_try:
            # Use retry handler for intelligent retry with backoff
            async def _make_api_call() -> LLMResponse:
                return await self._call_api(
                    messages=messages,
                    model=attempt_model,
                    **kwargs,
                )

            retry_result = await self._retry_handler.execute_async(_make_api_call)

            if retry_result.success:
                response = retry_result.value

                # Track retry statistics
                if retry_result.attempts > 1:
                    self._usage_stats["retried_requests"] += 1

                # Cache successful response (in-memory)
                if self._config.cache_enabled and cache_key:
                    self._save_to_cache(cache_key, response)

                # Save to persistent cache (cross-session)
                if self._use_persistent_cache:
                    tokens_saved = response.usage.get("total_tokens", 0)
                    self._save_to_persistent_cache(
                        messages=messages,
                        model=model_str,
                        response=response.content,
                        tokens_saved=tokens_saved,
                        temperature=temperature,
                    )

                return response

            # Retry failed - check if we should try next model in fallback chain
            last_error = retry_result.error
            last_category = retry_result.error_category
            error_category = retry_result.error_category

            # Log failure details
            import logging
            logging.getLogger(__name__).warning(
                f"Model {attempt_model} failed after {retry_result.attempts} attempts: "
                f"{last_error} (category={error_category.value if error_category else 'unknown'})"
            )

            # Track failed requests
            self._usage_stats["failed_requests"] += 1

            # For prompt- or credential-related failures, switching models won't help.
            # For model-unavailable and other errors, try the next model in the fallback chain.
            if error_category in {
                ErrorCategory.AUTH,
                ErrorCategory.INVALID_REQUEST,
                ErrorCategory.QUOTA_EXCEEDED,
            }:
                break

        # All models failed
        if self._config.fallback_chain and isinstance(target_model, ModelType):
            raise LLMError(f"All fallback models failed: {last_error}")

        if last_error is None:
            raise LLMError("Unknown error")

        # Preserve typed LLM errors (e.g., RateLimitError, ModelNotAvailableError)
        if isinstance(last_error, LLMError):
            raise last_error

        # Wrap retryable infrastructure errors into LLMError for a stable public API.
        if last_category == ErrorCategory.TIMEOUT or isinstance(last_error, asyncio.TimeoutError):
            raise LLMError("Timeout") from last_error
        if last_category == ErrorCategory.NETWORK:
            raise LLMError(f"Network error: {last_error}") from last_error
        if last_category == ErrorCategory.SERVER_ERROR:
            raise LLMError(f"Server error: {last_error}") from last_error
        if last_category == ErrorCategory.TRANSIENT:
            raise LLMError(f"Transient error: {last_error}") from last_error

        raise LLMError(str(last_error) or "Unknown error") from last_error

    async def _execute_with_auto_continue(
        self,
        messages: list[dict[str, str]],
        model: Optional[ModelType | str] = None,
        max_continue_rounds: int = 5,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute request with automatic continuation for truncated responses.

        When finish_reason is "length", automatically append "continue" and
        concatenate the responses (RD-Agent pattern).

        Args:
            messages: Chat messages.
            model: Model to use.
            max_continue_rounds: Maximum continuation rounds (default 5).
            **kwargs: Additional arguments for API call.

        Returns:
            LLMResponse with concatenated content from all rounds.
        """
        import logging
        logger = logging.getLogger(__name__)

        all_content = ""
        all_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        current_messages = [msg.copy() for msg in messages]  # Deep copy
        final_response: Optional[LLMResponse] = None
        total_cost = 0.0

        for round_idx in range(max_continue_rounds):
            # Call with fallback
            response = await self._execute_with_fallback(
                messages=current_messages,
                model=model,
                **kwargs,
            )

            all_content += response.content
            final_response = response

            # Accumulate usage
            for key in all_usage:
                all_usage[key] += response.usage.get(key, 0)

            # Accumulate cost
            if response.cost_estimate:
                total_cost += response.cost_estimate

            # Check if we need to continue
            if response.finish_reason != "length":
                logger.debug(f"Auto-continue complete after {round_idx + 1} round(s)")
                break

            # Response was truncated, prepare continuation
            logger.info(f"Response truncated, continuing (round {round_idx + 2}/{max_continue_rounds})")
            current_messages.append({"role": "assistant", "content": response.content})
            current_messages.append({"role": "user", "content": "continue"})
        else:
            # Reached max rounds
            logger.warning(f"Reached max continuation rounds ({max_continue_rounds})")

        # Build final response with combined content
        if final_response is None:
            raise LLMError("No response received")

        return LLMResponse(
            content=all_content,
            model=final_response.model,
            usage=all_usage,
            latency_ms=final_response.latency_ms,
            finish_reason=final_response.finish_reason,
            raw_response=final_response.raw_response,
            model_id=final_response.model_id,
            cost_estimate=total_cost if total_cost > 0 else None,
            request_id=final_response.request_id,
        )

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: ModelType | str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make actual API call to OpenRouter.

        Args:
            messages: Chat messages.
            model: Model to use - can be ModelType enum or direct OpenRouter model ID string.
        """
        await self._rate_limiter.acquire()

        # Get model ID - either from enum mapping or use string directly
        model_id = self.get_model_id(model)

        # Log the actual model being used for debugging
        import logging
        logging.getLogger(__name__).info(f"LLM API call using model: {model_id}")

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        start_time = time.time()

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")

            if response.status_code == 404:
                raise ModelNotAvailableError(f"Model {model_id} not available")

            if response.status_code != 200:
                raise LLMError(f"API Error: {response.status_code} - {response.text}")

            data = response.json()

            # Extract response with extended metadata
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Extract request ID from headers if available
            request_id = response.headers.get("x-request-id")

            # Estimate cost (OpenRouter provides this in some responses)
            # Fallback to rough estimate based on tokens
            cost_estimate = None
            if "cost" in data:
                cost_estimate = data["cost"]
            elif usage:
                # Rough estimate: $0.002 per 1K tokens (varies by model)
                total_tokens = usage.get("total_tokens", 0)
                cost_estimate = total_tokens * 0.000002

            # Track usage
            self.record_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )

            # Log if response was truncated
            import logging
            logger = logging.getLogger(__name__)
            if finish_reason == "length":
                logger.warning(f"Response truncated (finish_reason=length), may need continuation")

            return LLMResponse(
                content=content,
                model=model_id,
                usage=usage,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                raw_response=data,
                model_id=model_id,
                cost_estimate=cost_estimate,
                request_id=request_id,
            )

        except httpx.TimeoutException:
            raise LLMError("Timeout: Request timed out")
        except httpx.ConnectError:
            raise LLMError("Network: Connection failed")
        except asyncio.TimeoutError:
            raise LLMError("Timeout: Request timed out")
        except ConnectionError:
            raise LLMError("Network: Connection failed")

    def get_model_id(self, model: ModelType | str) -> str:
        """Get OpenRouter model ID for model type.

        Args:
            model: Either a ModelType enum or a direct OpenRouter model ID string.
                   String model IDs are returned as-is (for frontend-configured models).

        Returns:
            OpenRouter model ID string.
        """
        # IMPORTANT: ModelType is a `str` subclass; check it first.
        if isinstance(model, ModelType):
            return MODEL_ID_MAP.get(model, MODEL_ID_MAP[ModelType.DEEPSEEK_V3])
        return model

    def supported_models(self) -> list[ModelType]:
        """Get list of supported models."""
        return list(ModelType)

    def _generate_cache_key(
        self,
        content: str,
        model: ModelType | str,
        max_tokens: Optional[int],
    ) -> str:
        """Generate cache key for request."""
        model_str = model.value if isinstance(model, ModelType) else model
        key_data = f"{content}:{model_str}:{max_tokens}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[LLMResponse]:
        """Get response from cache if valid."""
        if key not in self._cache:
            return None

        response, timestamp = self._cache[key]
        if time.time() - timestamp > self._config.cache_ttl:
            del self._cache[key]
            return None

        response.cached = True
        return response

    def _get_from_persistent_cache(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> Optional[str]:
        """Get response from persistent cache (Redis L1 + PostgreSQL L2).

        Args:
            messages: Chat messages
            model: Model identifier
            temperature: Sampling temperature

        Returns:
            Cached response content or None
        """
        if self._persistent_cache is None:
            return None
        try:
            # Use sync wrapper for compatibility with both sync and async contexts
            return self._persistent_cache.get_sync(
                messages=messages,
                model=model,
                temperature=temperature,
            )
        except Exception as e:
            logger.debug(f"Persistent cache read failed (continuing without cache): {e}")
            return None

    def _save_to_persistent_cache(
        self,
        messages: list[dict[str, str]],
        model: str,
        response: str,
        tokens_saved: int,
        temperature: float,
    ) -> None:
        """Save response to persistent cache (Redis L1 + PostgreSQL L2).

        Args:
            messages: Chat messages
            model: Model identifier
            response: Response content
            tokens_saved: Estimated tokens saved
            temperature: Sampling temperature
        """
        if self._persistent_cache is None:
            return
        try:
            # Use sync wrapper for compatibility with both sync and async contexts
            self._persistent_cache.set_sync(
                messages=messages,
                model=model,
                response=response,
                tokens_saved=tokens_saved,
                temperature=temperature,
            )
        except Exception as e:
            # Cache writes are non-critical - log but don't fail
            logger.debug(f"Persistent cache write failed: {e}")

    def _save_to_cache(self, key: str, response: LLMResponse) -> None:
        """Save response to cache."""
        self._cache[key] = (response, time.time())

    def record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage statistics."""
        self._usage_stats["total_prompt_tokens"] += prompt_tokens
        self._usage_stats["total_completion_tokens"] += completion_tokens
        self._usage_stats["total_requests"] += 1

    def get_usage_stats(self) -> dict[str, int]:
        """Get usage statistics."""
        return self._usage_stats.copy()

    def get_cache_stats(self) -> Optional[PromptCacheStats]:
        """Get persistent cache statistics.

        Returns:
            PromptCacheStats with hit rate, entries, tokens saved, L1/L2 hits, etc.
            Returns None if persistent cache is not enabled.
        """
        if self._persistent_cache is None:
            return None
        try:
            # Use sync wrapper for compatibility
            return self._persistent_cache.get_stats_sync()
        except Exception as e:
            logger.debug(f"Failed to get cache stats: {e}")
            return None

    async def complete_structured(
        self,
        prompt: str,
        output_type: OutputType,
        model: Optional[ModelType | str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        auto_repair: bool = True,
        **kwargs: Any,
    ) -> tuple[LLMResponse, SchemaValidationResult]:
        """Generate structured output with JSON schema validation.

        Combines LLM completion with automatic schema validation and retry.
        Implements RD-Agent pattern for reliable structured outputs.

        Args:
            prompt: Input prompt (should request JSON output)
            output_type: Expected output type for schema validation
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            max_retries: Maximum retries on validation failure (default: 3)
            auto_repair: Attempt to repair common JSON issues (default: True)
            **kwargs: Additional arguments

        Returns:
            Tuple of (LLMResponse, SchemaValidationResult)

        Raises:
            LLMError: If all retries fail
        """
        import logging
        logger = logging.getLogger(__name__)

        # Initialize schema validator
        validator = JSONSchemaValidator()

        # Build system prompt with JSON formatting instruction
        json_instruction = (
            "\n\nIMPORTANT: Your response MUST be valid JSON that matches the expected schema. "
            "Do not include any text before or after the JSON object. "
            "Use double quotes for strings. Use null instead of None."
        )

        effective_system_prompt = (system_prompt or "") + json_instruction

        # Add example to prompt if available
        example = validator._get_example_output(output_type)
        if example:
            prompt_with_example = (
                f"{prompt}\n\n"
                f"Expected JSON format:\n{example}\n\n"
                "Return ONLY the JSON object, no additional text."
            )
        else:
            prompt_with_example = prompt

        last_error = None
        last_response = None
        last_validation = None

        for attempt in range(max_retries):
            try:
                # Call LLM
                response = await self.complete(
                    prompt=prompt_with_example if attempt == 0 else self._build_retry_prompt(
                        prompt_with_example, last_validation
                    ),
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=effective_system_prompt,
                    auto_continue=False,  # Structured output should be complete
                    **kwargs,
                )
                last_response = response

                # Validate response
                validation = validator.validate(
                    response.content,
                    output_type,
                    auto_repair=auto_repair,
                )
                last_validation = validation

                if validation.is_valid:
                    logger.info(
                        f"Structured output validation passed on attempt {attempt + 1}"
                    )
                    return response, validation

                # Validation failed, log and retry
                logger.warning(
                    f"Structured output validation failed on attempt {attempt + 1}: "
                    f"{validation.error_message}"
                )

            except Exception as e:
                last_error = e
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")

        # All retries exhausted
        if last_response and last_validation:
            logger.error(
                f"Structured output validation failed after {max_retries} attempts. "
                f"Last error: {last_validation.error_message}"
            )
            return last_response, last_validation

        # No successful response at all
        raise LLMError(
            f"Failed to get valid structured output after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _build_retry_prompt(
        self,
        original_prompt: str,
        last_validation: Optional[SchemaValidationResult],
    ) -> str:
        """Build retry prompt with validation feedback.

        Args:
            original_prompt: Original prompt
            last_validation: Last validation result

        Returns:
            Retry prompt with feedback
        """
        if not last_validation:
            return original_prompt

        error_feedback = (
            f"Your previous response was invalid JSON.\n"
            f"Error: {last_validation.error_message}\n"
        )

        if last_validation.repair_hint:
            error_feedback += f"Fix hint: {last_validation.repair_hint}\n"

        return f"{error_feedback}\n{original_prompt}"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "LLMProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
