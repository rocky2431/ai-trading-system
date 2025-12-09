"""Tests for FactorGenerationAgent (Task 8).

Six-dimensional test coverage:
1. Functional: Factor generation, prompt rendering, code extraction
2. Boundary: Edge cases for inputs
3. Exception: Error handling for LLM and security failures
4. Performance: Generation time
5. Security: AST checker integration
6. Compatibility: Different factor families
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
import asyncio
import time

from iqfmp.agents.factor_generation import (
    FactorGenerationAgent,
    FactorGenerationConfig,
    FactorFamily,
    FactorPromptTemplate,
    GeneratedFactor,
    FactorGenerationError,
    SecurityViolationError,
    InvalidFactorError,
)


class TestFactorPromptTemplate:
    """Tests for prompt template rendering."""

    def test_render_basic_prompt(self) -> None:
        """Test basic prompt template rendering."""
        template = FactorPromptTemplate()
        prompt = template.render(
            user_request="Create a momentum factor based on 20-day returns",
        )
        assert "momentum" in prompt.lower() or "20-day" in prompt.lower()
        assert len(prompt) > 100

    def test_render_with_factor_family(self) -> None:
        """Test prompt rendering with factor family constraint."""
        template = FactorPromptTemplate()
        prompt = template.render(
            user_request="Create a volatility factor",
            factor_family=FactorFamily.VOLATILITY,
        )
        assert "volatility" in prompt.lower()

    def test_render_with_examples(self) -> None:
        """Test prompt rendering includes examples."""
        template = FactorPromptTemplate()
        prompt = template.render(
            user_request="Create a factor",
            include_examples=True,
        )
        assert "example" in prompt.lower() or "def " in prompt

    def test_system_prompt_content(self) -> None:
        """Test system prompt contains key instructions."""
        template = FactorPromptTemplate()
        system = template.get_system_prompt()
        assert "qlib" in system.lower() or "factor" in system.lower()
        assert "python" in system.lower()


class TestFactorFamily:
    """Tests for factor family definitions."""

    def test_all_families_defined(self) -> None:
        """Test all expected factor families exist."""
        expected = ["MOMENTUM", "VALUE", "VOLATILITY", "QUALITY", "SENTIMENT", "LIQUIDITY"]
        for family_name in expected:
            assert hasattr(FactorFamily, family_name)

    def test_family_has_allowed_fields(self) -> None:
        """Test each family has allowed fields defined."""
        for family in FactorFamily:
            fields = family.get_allowed_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_momentum_family_fields(self) -> None:
        """Test momentum family has expected fields."""
        fields = FactorFamily.MOMENTUM.get_allowed_fields()
        # Momentum should include price-related fields
        assert any("close" in f.lower() or "return" in f.lower() for f in fields)

    def test_volatility_family_fields(self) -> None:
        """Test volatility family has expected fields."""
        fields = FactorFamily.VOLATILITY.get_allowed_fields()
        # Volatility should include high/low or std-related fields
        assert len(fields) > 0


class TestGeneratedFactor:
    """Tests for generated factor data structure."""

    def test_create_factor(self) -> None:
        """Test creating a generated factor."""
        factor = GeneratedFactor(
            name="momentum_20d",
            description="20-day momentum factor",
            code="def factor(df): return df['close'].pct_change(20)",
            family=FactorFamily.MOMENTUM,
        )
        assert factor.name == "momentum_20d"
        assert factor.family == FactorFamily.MOMENTUM

    def test_factor_validation(self) -> None:
        """Test factor validation."""
        factor = GeneratedFactor(
            name="test_factor",
            description="Test",
            code="def factor(df): return df['close']",
            family=FactorFamily.MOMENTUM,
        )
        assert factor.is_valid()

    def test_factor_to_dict(self) -> None:
        """Test factor serialization."""
        factor = GeneratedFactor(
            name="test",
            description="Test factor",
            code="def factor(df): return df['close']",
            family=FactorFamily.MOMENTUM,
        )
        d = factor.to_dict()
        assert "name" in d
        assert "code" in d
        assert "family" in d


class TestFactorGenerationAgentFunctional:
    """Functional tests for FactorGenerationAgent."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def momentum_factor(df):
    """20-day momentum factor."""
    return df['close'].pct_change(20)
```''',
        ))
        return provider

    @pytest.fixture
    def agent(self, mock_llm_provider: MagicMock) -> FactorGenerationAgent:
        """Create agent with mock LLM."""
        config = FactorGenerationConfig(
            name="test_agent",
            security_check_enabled=True,
        )
        return FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_generate_factor(self, agent: FactorGenerationAgent) -> None:
        """Test basic factor generation."""
        result = await agent.generate(
            user_request="Create a 20-day momentum factor",
        )
        assert result is not None
        assert isinstance(result, GeneratedFactor)
        assert "momentum" in result.name.lower() or "factor" in result.name.lower()

    @pytest.mark.asyncio
    async def test_generate_with_family(self, agent: FactorGenerationAgent) -> None:
        """Test generation with specific factor family."""
        result = await agent.generate(
            user_request="Create a volatility factor",
            factor_family=FactorFamily.VOLATILITY,
        )
        assert result is not None
        assert result.family == FactorFamily.VOLATILITY

    @pytest.mark.asyncio
    async def test_code_extraction(self, agent: FactorGenerationAgent) -> None:
        """Test code is properly extracted from LLM response."""
        result = await agent.generate(
            user_request="Create a factor",
        )
        assert result.code is not None
        assert "def " in result.code
        assert "return " in result.code

    @pytest.mark.asyncio
    async def test_multiple_generations(self, agent: FactorGenerationAgent) -> None:
        """Test multiple sequential generations."""
        results = []
        for i in range(3):
            result = await agent.generate(
                user_request=f"Create factor {i}",
            )
            results.append(result)
        assert len(results) == 3
        assert all(r is not None for r in results)


class TestFactorGenerationAgentBoundary:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=MagicMock(
            content="def factor(df): return df['close']",
        ))
        return provider

    @pytest.fixture
    def agent(self, mock_llm_provider: MagicMock) -> FactorGenerationAgent:
        """Create agent with mock LLM."""
        config = FactorGenerationConfig(name="test")
        return FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_empty_request(self, agent: FactorGenerationAgent) -> None:
        """Test handling of empty request."""
        with pytest.raises(ValueError, match="Request cannot be empty"):
            await agent.generate(user_request="")

    @pytest.mark.asyncio
    async def test_very_long_request(
        self, agent: FactorGenerationAgent, mock_llm_provider: MagicMock
    ) -> None:
        """Test handling of very long request."""
        long_request = "Create a momentum factor " * 1000
        result = await agent.generate(user_request=long_request)
        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters(
        self, agent: FactorGenerationAgent, mock_llm_provider: MagicMock
    ) -> None:
        """Test handling of special characters in request."""
        result = await agent.generate(
            user_request="Create factor with 中文 and emoji 🚀 @#$%"
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_minimal_valid_code(
        self, agent: FactorGenerationAgent
    ) -> None:
        """Test handling of minimal valid code."""
        result = await agent.generate(user_request="Simple factor")
        assert result is not None
        assert len(result.code) > 0


class TestFactorGenerationAgentException:
    """Exception handling tests."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, mock_llm_provider: MagicMock) -> None:
        """Test handling of LLM errors."""
        mock_llm_provider.complete = AsyncMock(
            side_effect=Exception("LLM API Error")
        )
        config = FactorGenerationConfig(name="test")
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(FactorGenerationError, match="LLM"):
            await agent.generate(user_request="Create factor")

    @pytest.mark.asyncio
    async def test_invalid_code_handling(self, mock_llm_provider: MagicMock) -> None:
        """Test handling of invalid code from LLM."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content="This is not valid Python code at all @#$%^&",
        ))
        config = FactorGenerationConfig(name="test")
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(InvalidFactorError):
            await agent.generate(user_request="Create factor")

    @pytest.mark.asyncio
    async def test_no_code_in_response(self, mock_llm_provider: MagicMock) -> None:
        """Test handling when LLM returns no code."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content="I cannot generate code for this request.",
        ))
        config = FactorGenerationConfig(name="test")
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(InvalidFactorError, match="No code"):
            await agent.generate(user_request="Create factor")


class TestFactorGenerationSecurity:
    """Security tests for AST checker integration."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_dangerous_code_rejected(self, mock_llm_provider: MagicMock) -> None:
        """Test that dangerous code is rejected."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def malicious_factor(df):
    import os
    os.system("rm -rf /")
    return df['close']
```''',
        ))
        config = FactorGenerationConfig(
            name="test",
            security_check_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(SecurityViolationError):
            await agent.generate(user_request="Create factor")

    @pytest.mark.asyncio
    async def test_eval_rejected(self, mock_llm_provider: MagicMock) -> None:
        """Test that eval is rejected."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def factor(df):
    return eval("df['close']")
```''',
        ))
        config = FactorGenerationConfig(
            name="test",
            security_check_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(SecurityViolationError):
            await agent.generate(user_request="Create factor")

    @pytest.mark.asyncio
    async def test_exec_rejected(self, mock_llm_provider: MagicMock) -> None:
        """Test that exec is rejected."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def factor(df):
    exec("x = 1")
    return df['close']
```''',
        ))
        config = FactorGenerationConfig(
            name="test",
            security_check_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(SecurityViolationError):
            await agent.generate(user_request="Create factor")

    @pytest.mark.asyncio
    async def test_safe_code_accepted(self, mock_llm_provider: MagicMock) -> None:
        """Test that safe code is accepted."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def safe_factor(df):
    """Safe momentum factor."""
    import numpy as np
    returns = df['close'].pct_change(20)
    return returns.fillna(0)
```''',
        ))
        config = FactorGenerationConfig(
            name="test",
            security_check_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        result = await agent.generate(user_request="Create momentum factor")
        assert result is not None

    @pytest.mark.asyncio
    async def test_security_bypass_disabled(self, mock_llm_provider: MagicMock) -> None:
        """Test that security check can be disabled (for testing only)."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content="def factor(df): return eval('1')",
        ))
        config = FactorGenerationConfig(
            name="test",
            security_check_enabled=False,  # Dangerous - testing only
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        # Should not raise when security check is disabled
        result = await agent.generate(user_request="Create factor")
        assert result is not None


class TestFactorGenerationPerformance:
    """Performance tests."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider with delay."""
        provider = MagicMock()

        async def delayed_complete(*args: Any, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(0.05)  # 50ms delay
            return MagicMock(
                content="def factor(df): return df['close'].pct_change(20)",
            )

        provider.complete = delayed_complete
        return provider

    @pytest.mark.asyncio
    async def test_generation_time(self, mock_llm_provider: MagicMock) -> None:
        """Test generation completes in reasonable time."""
        config = FactorGenerationConfig(name="test")
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        start = time.time()
        result = await agent.generate(user_request="Create factor")
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 5.0  # Should complete within 5 seconds


class TestFactorGenerationCompatibility:
    """Compatibility tests for different configurations."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=MagicMock(
            content="def factor(df): return df['close']",
        ))
        return provider

    @pytest.mark.asyncio
    async def test_all_factor_families(self, mock_llm_provider: MagicMock) -> None:
        """Test generation works with all factor families."""
        config = FactorGenerationConfig(name="test")
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        for family in FactorFamily:
            result = await agent.generate(
                user_request=f"Create {family.value} factor",
                factor_family=family,
            )
            assert result is not None
            assert result.family == family

    @pytest.mark.asyncio
    async def test_different_code_formats(self, mock_llm_provider: MagicMock) -> None:
        """Test extraction of different code formats."""
        config = FactorGenerationConfig(name="test")
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        # Test markdown code block format
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='```python\ndef factor(df): return df["close"]\n```',
        ))
        result1 = await agent.generate(user_request="Create factor")
        assert result1 is not None

        # Test plain code format
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='def factor(df): return df["close"]',
        ))
        result2 = await agent.generate(user_request="Create factor")
        assert result2 is not None

    def test_config_defaults(self) -> None:
        """Test configuration defaults."""
        config = FactorGenerationConfig(name="test")
        assert config.security_check_enabled is True
        assert config.max_retries >= 1
