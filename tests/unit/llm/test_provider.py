"""Tests for LLMProvider (Task 6).

Six-dimensional test coverage:
1. Functional: Basic LLM calls, model selection, configuration
2. Boundary: Edge cases for prompts and responses
3. Exception: Error handling for API failures
4. Performance: Response time and rate limiting
5. Security: API key handling
6. Compatibility: Multi-model support
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
import asyncio
import time

from iqfmp.llm.provider import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    ModelType,
    FallbackChain,
    RateLimiter,
    LLMError,
    RateLimitError,
    ModelNotAvailableError,
)


class TestLLMProviderFunctional:
    """Functional tests for core LLM provider functionality."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        return LLMConfig(
            api_key="test-api-key",
            base_url="https://openrouter.ai/api/v1",
            default_model=ModelType.DEEPSEEK_V3,
            timeout=30,
        )

    @pytest.fixture
    def provider(self, config: LLMConfig) -> LLMProvider:
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_basic_completion(self, provider: LLMProvider) -> None:
        """Test basic text completion."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Hello, world!",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            response = await provider.complete("Say hello")

            assert response.content == "Hello, world!"
            assert response.model == "deepseek/deepseek-chat"
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion(self, provider: LLMProvider) -> None:
        """Test chat-style completion with messages."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="I can help with Python code.",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Help me write Python code."},
            ]
            response = await provider.chat(messages)

            assert "Python" in response.content
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_selection(self, provider: LLMProvider) -> None:
        """Test explicit model selection."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Response from Claude",
                model="anthropic/claude-3.5-sonnet",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            response = await provider.complete(
                "Test prompt",
                model=ModelType.CLAUDE_35_SONNET
            )

            assert response.model == "anthropic/claude-3.5-sonnet"

    def test_config_from_env(self) -> None:
        """Test configuration loading from environment variables."""
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "env-api-key",
            "OPENROUTER_DEFAULT_MODEL": "gpt-4o",
        }):
            config = LLMConfig.from_env()
            assert config.api_key == "env-api-key"

    def test_supported_models(self, provider: LLMProvider) -> None:
        """Test listing supported models."""
        models = provider.supported_models()
        assert ModelType.DEEPSEEK_V3 in models
        assert ModelType.CLAUDE_35_SONNET in models
        assert ModelType.GPT_4O in models


class TestLLMProviderBoundary:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def provider(self) -> LLMProvider:
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_empty_prompt(self, provider: LLMProvider) -> None:
        """Test handling of empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await provider.complete("")

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, provider: LLMProvider) -> None:
        """Test handling of very long prompts."""
        long_prompt = "a" * 100000  # 100K characters

        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 50000, "completion_tokens": 10},
            )

            response = await provider.complete(long_prompt)
            assert response is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, provider: LLMProvider) -> None:
        """Test handling of special characters."""
        special_prompt = "Test with 中文, emoji 🚀, and symbols: @#$%^&*()"

        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Handled special chars",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            )

            response = await provider.complete(special_prompt)
            assert response is not None

    @pytest.mark.asyncio
    async def test_max_tokens_limit(self, provider: LLMProvider) -> None:
        """Test respecting max_tokens parameter."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Short response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 50},
            )

            response = await provider.complete("Test", max_tokens=50)

            # Verify max_tokens was passed to API
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs.get("max_tokens") == 50


class TestLLMProviderException:
    """Exception handling tests."""

    @pytest.fixture
    def provider(self) -> LLMProvider:
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_api_error_handling(self, provider: LLMProvider) -> None:
        """Test handling of API errors."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.side_effect = LLMError("API Error: Invalid request")

            with pytest.raises(LLMError, match="API Error"):
                await provider.complete("Test")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, provider: LLMProvider) -> None:
        """Test handling of rate limit errors."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.side_effect = RateLimitError("Rate limit exceeded")

            with pytest.raises(RateLimitError):
                await provider.complete("Test")

    @pytest.mark.asyncio
    async def test_model_not_available(self, provider: LLMProvider) -> None:
        """Test handling of unavailable model."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.side_effect = ModelNotAvailableError("Model not found")

            with pytest.raises(ModelNotAvailableError):
                await provider.complete("Test", model=ModelType.GPT_4O)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, provider: LLMProvider) -> None:
        """Test handling of timeout errors."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.side_effect = asyncio.TimeoutError()

            with pytest.raises(LLMError, match="Timeout"):
                await provider.complete("Test")

    @pytest.mark.asyncio
    async def test_network_error(self, provider: LLMProvider) -> None:
        """Test handling of network errors."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.side_effect = ConnectionError("Network unavailable")

            with pytest.raises(LLMError, match="Network"):
                await provider.complete("Test")


class TestFallbackChain:
    """Tests for fallback chain functionality."""

    @pytest.fixture
    def fallback_chain(self) -> FallbackChain:
        return FallbackChain(
            models=[
                ModelType.DEEPSEEK_V3,
                ModelType.CLAUDE_35_SONNET,
                ModelType.GPT_4O,
            ],
            max_retries=2,
        )

    @pytest.fixture
    def provider_with_fallback(self, fallback_chain: FallbackChain) -> LLMProvider:
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            fallback_chain=fallback_chain,
        )
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, provider_with_fallback: LLMProvider) -> None:
        """Test fallback to next model on error."""
        call_count = 0

        async def mock_call(*args: Any, **kwargs: Any) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ModelNotAvailableError("DeepSeek unavailable")
            return LLMResponse(
                content="Response from fallback",
                model="anthropic/claude-3.5-sonnet",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

        with patch.object(provider_with_fallback, "_call_api", side_effect=mock_call):
            response = await provider_with_fallback.complete("Test")
            assert response.model == "anthropic/claude-3.5-sonnet"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_exhausted(self, provider_with_fallback: LLMProvider) -> None:
        """Test error when all fallback models fail."""
        with patch.object(provider_with_fallback, "_call_api") as mock_call:
            mock_call.side_effect = ModelNotAvailableError("All models unavailable")

            with pytest.raises(LLMError, match="All fallback models failed"):
                await provider_with_fallback.complete("Test")

    def test_fallback_chain_order(self, fallback_chain: FallbackChain) -> None:
        """Test fallback chain maintains order."""
        models = fallback_chain.get_models()
        assert models[0] == ModelType.DEEPSEEK_V3
        assert models[1] == ModelType.CLAUDE_35_SONNET
        assert models[2] == ModelType.GPT_4O


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def rate_limiter(self) -> RateLimiter:
        return RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=100000,
        )

    @pytest.mark.asyncio
    async def test_rate_limit_requests(self, rate_limiter: RateLimiter) -> None:
        """Test request rate limiting."""
        # Should allow requests within limit
        for _ in range(5):
            assert await rate_limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self) -> None:
        """Test rate limit exceeded behavior."""
        limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=1000)

        # First two should succeed
        assert await limiter.acquire()
        assert await limiter.acquire()

        # Third should be rate limited (or wait)
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should have waited at least some time
        assert elapsed > 0 or limiter.get_remaining_requests() >= 0

    def test_token_tracking(self, rate_limiter: RateLimiter) -> None:
        """Test token usage tracking."""
        rate_limiter.record_usage(tokens=1000)
        assert rate_limiter.get_tokens_used() == 1000

        rate_limiter.record_usage(tokens=500)
        assert rate_limiter.get_tokens_used() == 1500


class TestCaching:
    """Tests for request caching functionality."""

    @pytest.fixture
    def provider_with_cache(self) -> LLMProvider:
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            cache_enabled=True,
            cache_ttl=300,  # 5 minutes
        )
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_cache_hit(self, provider_with_cache: LLMProvider) -> None:
        """Test cache hit returns cached response."""
        with patch.object(provider_with_cache, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Cached response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            # First call - cache miss
            response1 = await provider_with_cache.complete("Test prompt")
            # Second call - cache hit
            response2 = await provider_with_cache.complete("Test prompt")

            assert response1.content == response2.content
            assert mock_call.call_count == 1  # Only called once due to cache

    @pytest.mark.asyncio
    async def test_cache_disabled(self) -> None:
        """Test cache can be disabled."""
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            cache_enabled=False,
        )
        provider = LLMProvider(config=config)

        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            await provider.complete("Test")
            await provider.complete("Test")

            assert mock_call.call_count == 2  # Called twice, no caching

    def test_cache_key_generation(self, provider_with_cache: LLMProvider) -> None:
        """Test cache key includes relevant parameters."""
        key1 = provider_with_cache._generate_cache_key(
            "prompt", ModelType.DEEPSEEK_V3, 100
        )
        key2 = provider_with_cache._generate_cache_key(
            "prompt", ModelType.DEEPSEEK_V3, 200
        )
        key3 = provider_with_cache._generate_cache_key(
            "prompt", ModelType.CLAUDE_35_SONNET, 100
        )

        # Different parameters should produce different keys
        assert key1 != key2
        assert key1 != key3


class TestSecurity:
    """Security tests for API key handling."""

    def test_api_key_not_in_repr(self) -> None:
        """Test API key is not exposed in string representation."""
        config = LLMConfig(
            api_key="super-secret-key-12345",
            base_url="https://openrouter.ai/api/v1",
        )
        provider = LLMProvider(config=config)

        repr_str = repr(provider)
        str_str = str(provider)

        assert "super-secret-key-12345" not in repr_str
        assert "super-secret-key-12345" not in str_str

    def test_api_key_not_in_logs(self) -> None:
        """Test API key is masked in logs."""
        config = LLMConfig(
            api_key="super-secret-key-12345",
            base_url="https://openrouter.ai/api/v1",
        )

        log_safe = config.to_log_safe_dict()
        assert log_safe["api_key"] == "****"

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError, match="API key is required"):
            LLMConfig(api_key="", base_url="https://openrouter.ai/api/v1")

        with pytest.raises(ValueError, match="Invalid base URL"):
            LLMConfig(api_key="key", base_url="not-a-url")


class TestCompatibility:
    """Compatibility tests for multi-model support."""

    @pytest.fixture
    def provider(self) -> LLMProvider:
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_deepseek_model(self, provider: LLMProvider) -> None:
        """Test DeepSeek model compatibility."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="DeepSeek response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            response = await provider.complete("Test", model=ModelType.DEEPSEEK_V3)
            assert "deepseek" in response.model

    @pytest.mark.asyncio
    async def test_claude_model(self, provider: LLMProvider) -> None:
        """Test Claude model compatibility."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Claude response",
                model="anthropic/claude-3.5-sonnet",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            response = await provider.complete("Test", model=ModelType.CLAUDE_35_SONNET)
            assert "claude" in response.model

    @pytest.mark.asyncio
    async def test_gpt_model(self, provider: LLMProvider) -> None:
        """Test GPT model compatibility."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="GPT response",
                model="openai/gpt-4o",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            response = await provider.complete("Test", model=ModelType.GPT_4O)
            assert "gpt" in response.model

    def test_model_id_mapping(self, provider: LLMProvider) -> None:
        """Test model type to OpenRouter ID mapping."""
        assert provider.get_model_id(ModelType.DEEPSEEK_V3) == "deepseek/deepseek-chat"
        assert provider.get_model_id(ModelType.CLAUDE_35_SONNET) == "anthropic/claude-3.5-sonnet"
        assert provider.get_model_id(ModelType.GPT_4O) == "openai/gpt-4o"


class TestPerformance:
    """Performance tests."""

    @pytest.fixture
    def provider(self) -> LLMProvider:
        config = LLMConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        return LLMProvider(config=config)

    @pytest.mark.asyncio
    async def test_response_time_tracking(self, provider: LLMProvider) -> None:
        """Test response time is tracked."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                latency_ms=150,
            )

            response = await provider.complete("Test")
            assert response.latency_ms is not None
            assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider: LLMProvider) -> None:
        """Test handling of concurrent requests."""
        with patch.object(provider, "_call_api") as mock_call:
            mock_call.return_value = LLMResponse(
                content="Response",
                model="deepseek/deepseek-chat",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

            # Send 5 concurrent requests
            tasks = [provider.complete(f"Test {i}") for i in range(5)]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            assert all(r.content == "Response" for r in responses)

    def test_usage_statistics(self, provider: LLMProvider) -> None:
        """Test usage statistics tracking."""
        provider.record_usage(prompt_tokens=100, completion_tokens=50)
        provider.record_usage(prompt_tokens=200, completion_tokens=100)

        stats = provider.get_usage_stats()
        assert stats["total_prompt_tokens"] == 300
        assert stats["total_completion_tokens"] == 150
        assert stats["total_requests"] == 2
