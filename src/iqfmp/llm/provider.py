"""LLM Provider for OpenRouter API integration.

This module provides a unified interface for accessing multiple LLM models
through OpenRouter, with support for:
- Multi-model switching (DeepSeek, Claude, GPT)
- Fallback chains for reliability
- Request caching for efficiency
- Rate limiting for cost control
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import httpx


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
    """Response from LLM API call."""
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: Optional[float] = None
    cached: bool = False


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

    def __init__(self, config: LLMConfig) -> None:
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
        self._cache: dict[str, tuple[LLMResponse, float]] = {}
        self._rate_limiter = RateLimiter()
        self._usage_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
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
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt.
            model: Optional model to use - can be ModelType enum or OpenRouter model ID string
                   (e.g., "deepseek/deepseek-coder-v3"). Defaults to config default.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt to guide the model.

        Returns:
            LLMResponse with generated content.

        Raises:
            ValueError: If prompt is empty.
            LLMError: On API errors.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self._execute_with_fallback(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[ModelType | str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            messages: List of chat messages.
            model: Optional model to use - can be ModelType enum or OpenRouter model ID string.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with generated content.
        """
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
        # Determine target model (string ID or ModelType)
        target_model: ModelType | str = model or self._config.default_model

        # Check cache first
        if self._config.cache_enabled:
            # For cache key, convert to string representation
            model_str = target_model.value if isinstance(target_model, ModelType) else target_model
            cache_key = self._generate_cache_key(
                str(messages),
                model_str,
                kwargs.get("max_tokens"),
            )
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached

        # Build fallback chain - only for ModelType models
        # String model IDs (from frontend config) don't use fallback
        if isinstance(target_model, str):
            # Direct OpenRouter model ID - no fallback chain
            models_to_try: list[ModelType | str] = [target_model]
        elif self._config.fallback_chain:
            models_to_try = list(self._config.fallback_chain.get_models())
            if target_model not in models_to_try:
                models_to_try.insert(0, target_model)
        else:
            models_to_try = [target_model]

        last_error: Optional[Exception] = None

        for attempt_model in models_to_try:
            try:
                response = await self._call_api(
                    messages=messages,
                    model=attempt_model,
                    **kwargs,
                )

                # Cache successful response
                if self._config.cache_enabled:
                    self._save_to_cache(cache_key, response)

                return response

            except ModelNotAvailableError as e:
                last_error = e
                continue
            except RateLimitError as e:
                last_error = e
                continue
            except asyncio.TimeoutError as e:
                last_error = LLMError("Timeout: Request timed out")
                break
            except ConnectionError as e:
                last_error = LLMError(f"Network: {e}")
                break
            except Exception as e:
                last_error = e
                break

        # All models failed
        if self._config.fallback_chain and isinstance(target_model, ModelType):
            raise LLMError(f"All fallback models failed: {last_error}")
        raise last_error or LLMError("Unknown error")

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

            # Extract response
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Track usage
            self.record_usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )

            return LLMResponse(
                content=content,
                model=model_id,
                usage=usage,
                latency_ms=latency_ms,
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
        # If string, return as-is (direct OpenRouter model ID from frontend config)
        if isinstance(model, str):
            return model
        # If ModelType enum, look up in mapping
        return MODEL_ID_MAP.get(model, MODEL_ID_MAP[ModelType.DEEPSEEK_V3])

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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "LLMProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
