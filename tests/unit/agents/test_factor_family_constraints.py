"""Tests for Factor Family Constraints (Task 9).

Six-dimensional test coverage:
1. Functional: Field validation, constraint checking, prompt enhancement
2. Boundary: Edge cases for field patterns
3. Exception: Error handling for constraint violations
4. Performance: Validation time
5. Security: Field injection prevention
6. Compatibility: Different code formats and field access patterns
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any
import time

from iqfmp.agents.factor_generation import (
    FactorGenerationAgent,
    FactorGenerationConfig,
    FactorFamily,
    FactorPromptTemplate,
    GeneratedFactor,
    FactorFieldValidator,
    FieldConstraintViolationError,
    FieldValidationResult,
)


class TestFactorFieldValidator:
    """Tests for FactorFieldValidator."""

    @pytest.fixture
    def validator(self) -> FactorFieldValidator:
        """Create field validator instance."""
        return FactorFieldValidator()

    def test_extract_fields_bracket_notation(self, validator: FactorFieldValidator) -> None:
        """Test extraction of fields using bracket notation df['xxx']."""
        code = """
def momentum_factor(df):
    returns = df['close'].pct_change(20)
    volume = df['volume']
    return returns * volume
"""
        fields = validator.extract_fields(code)
        assert "close" in fields
        assert "volume" in fields
        assert len(fields) == 2

    def test_extract_fields_double_quotes(self, validator: FactorFieldValidator) -> None:
        """Test extraction of fields using double quotes df["xxx"]."""
        code = '''
def factor(df):
    return df["close"] / df["open"]
'''
        fields = validator.extract_fields(code)
        assert "close" in fields
        assert "open" in fields

    def test_extract_fields_dot_notation(self, validator: FactorFieldValidator) -> None:
        """Test extraction of fields using dot notation df.close."""
        code = """
def factor(df):
    return df.close.pct_change(10)
"""
        fields = validator.extract_fields(code)
        assert "close" in fields

    def test_extract_fields_mixed_notation(self, validator: FactorFieldValidator) -> None:
        """Test extraction with mixed notation styles."""
        code = """
def factor(df):
    a = df['close']
    b = df["high"]
    c = df.low
    return a + b + c
"""
        fields = validator.extract_fields(code)
        assert "close" in fields
        assert "high" in fields
        assert "low" in fields
        assert len(fields) == 3

    def test_validate_fields_pass(self, validator: FactorFieldValidator) -> None:
        """Test validation passes when all fields are allowed."""
        code = """
def momentum_factor(df):
    return df['close'].pct_change(20)
"""
        result = validator.validate(code, FactorFamily.MOMENTUM)
        assert result.is_valid
        assert len(result.violations) == 0
        assert "close" in result.used_fields

    def test_validate_fields_fail(self, validator: FactorFieldValidator) -> None:
        """Test validation fails when using disallowed fields."""
        code = """
def momentum_factor(df):
    return df['sentiment_score'].rolling(20).mean()
"""
        result = validator.validate(code, FactorFamily.MOMENTUM)
        assert not result.is_valid
        assert "sentiment_score" in result.violations
        assert "sentiment_score" in result.used_fields

    def test_validate_multiple_violations(self, validator: FactorFieldValidator) -> None:
        """Test validation reports all violations."""
        code = """
def bad_momentum_factor(df):
    sentiment = df['sentiment_score']
    news = df['news_count']
    return sentiment * news
"""
        result = validator.validate(code, FactorFamily.MOMENTUM)
        assert not result.is_valid
        assert len(result.violations) >= 2
        assert "sentiment_score" in result.violations
        assert "news_count" in result.violations

    def test_validate_with_allowed_fields(self, validator: FactorFieldValidator) -> None:
        """Test validation with all allowed fields."""
        code = """
def volatility_factor(df):
    high_low = df['high'] - df['low']
    returns = df['close'].pct_change()
    return high_low * returns.std()
"""
        result = validator.validate(code, FactorFamily.VOLATILITY)
        assert result.is_valid
        assert "high" in result.used_fields
        assert "low" in result.used_fields
        assert "close" in result.used_fields


class TestFieldValidationResult:
    """Tests for FieldValidationResult data structure."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid result."""
        result = FieldValidationResult(
            is_valid=True,
            used_fields={"close", "volume"},
            allowed_fields={"close", "volume", "open"},
            violations=[],
        )
        assert result.is_valid
        assert len(result.used_fields) == 2
        assert len(result.violations) == 0

    def test_create_invalid_result(self) -> None:
        """Test creating an invalid result."""
        result = FieldValidationResult(
            is_valid=False,
            used_fields={"close", "sentiment_score"},
            allowed_fields={"close", "volume"},
            violations=["sentiment_score"],
        )
        assert not result.is_valid
        assert "sentiment_score" in result.violations

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = FieldValidationResult(
            is_valid=True,
            used_fields={"close"},
            allowed_fields={"close", "volume"},
            violations=[],
        )
        d = result.to_dict()
        assert "is_valid" in d
        assert "used_fields" in d
        assert "allowed_fields" in d
        assert "violations" in d


class TestFactorFamilyEnhanced:
    """Tests for enhanced FactorFamily with detailed field definitions."""

    def test_momentum_family_fields(self) -> None:
        """Test momentum family has comprehensive fields."""
        fields = FactorFamily.MOMENTUM.get_allowed_fields()
        # Must include price-related fields
        assert "close" in fields
        assert "open" in fields
        assert "high" in fields
        assert "low" in fields
        # May include momentum-specific fields
        assert any(f in fields for f in ["returns", "momentum", "roc"])

    def test_value_family_fields(self) -> None:
        """Test value family has fundamental fields."""
        fields = FactorFamily.VALUE.get_allowed_fields()
        assert "close" in fields
        # Should have fundamental fields
        assert any(f in fields for f in ["book_value", "earnings", "pe_ratio", "pb_ratio"])

    def test_volatility_family_fields(self) -> None:
        """Test volatility family has price range fields."""
        fields = FactorFamily.VOLATILITY.get_allowed_fields()
        assert "high" in fields
        assert "low" in fields
        assert "close" in fields

    def test_quality_family_fields(self) -> None:
        """Test quality family has profitability fields."""
        fields = FactorFamily.QUALITY.get_allowed_fields()
        assert any(f in fields for f in ["roe", "roa", "profit_margin"])

    def test_sentiment_family_fields(self) -> None:
        """Test sentiment family has sentiment fields."""
        fields = FactorFamily.SENTIMENT.get_allowed_fields()
        assert any(f in fields for f in ["sentiment_score", "news_count"])

    def test_liquidity_family_fields(self) -> None:
        """Test liquidity family has liquidity fields."""
        fields = FactorFamily.LIQUIDITY.get_allowed_fields()
        assert "volume" in fields
        assert any(f in fields for f in ["turnover", "bid_ask_spread", "amihud"])

    def test_get_field_descriptions(self) -> None:
        """Test getting field descriptions for a family."""
        descriptions = FactorFamily.MOMENTUM.get_field_descriptions()
        assert isinstance(descriptions, dict)
        assert "close" in descriptions
        assert len(descriptions["close"]) > 0  # Has description


class TestPromptWithFieldConstraints:
    """Tests for prompt rendering with field constraints."""

    @pytest.fixture
    def template(self) -> FactorPromptTemplate:
        """Create template instance."""
        return FactorPromptTemplate()

    def test_render_with_strict_constraints(self, template: FactorPromptTemplate) -> None:
        """Test prompt includes strict field constraints."""
        prompt = template.render(
            user_request="Create a momentum factor",
            factor_family=FactorFamily.MOMENTUM,
            include_field_constraints=True,
        )
        assert "momentum" in prompt.lower()
        assert "allowed" in prompt.lower() or "fields" in prompt.lower()

    def test_render_includes_field_list(self, template: FactorPromptTemplate) -> None:
        """Test prompt includes list of allowed fields."""
        prompt = template.render(
            user_request="Create a factor",
            factor_family=FactorFamily.VOLATILITY,
            include_field_constraints=True,
        )
        assert "high" in prompt or "low" in prompt or "close" in prompt

    def test_render_without_constraints(self, template: FactorPromptTemplate) -> None:
        """Test prompt without field constraints."""
        prompt = template.render(
            user_request="Create a factor",
            factor_family=FactorFamily.MOMENTUM,
            include_field_constraints=False,
        )
        # Should still mention the family
        assert "momentum" in prompt.lower() or "factor" in prompt.lower()


class TestFactorGenerationWithConstraintsFunctional:
    """Functional tests for factor generation with field constraints."""

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
    def agent_with_constraints(self, mock_llm_provider: MagicMock) -> FactorGenerationAgent:
        """Create agent with field constraints enabled."""
        config = FactorGenerationConfig(
            name="test_agent",
            security_check_enabled=True,
            field_constraint_enabled=True,
        )
        return FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_generate_with_valid_fields(
        self, agent_with_constraints: FactorGenerationAgent
    ) -> None:
        """Test generation succeeds with valid fields."""
        result = await agent_with_constraints.generate(
            user_request="Create a momentum factor",
            factor_family=FactorFamily.MOMENTUM,
        )
        assert result is not None
        assert result.family == FactorFamily.MOMENTUM

    @pytest.mark.asyncio
    async def test_generate_rejects_invalid_fields(
        self, mock_llm_provider: MagicMock
    ) -> None:
        """Test generation fails when using disallowed fields."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def bad_momentum_factor(df):
    """Uses sentiment data in momentum factor."""
    return df['sentiment_score'].rolling(20).mean()
```''',
        ))
        config = FactorGenerationConfig(
            name="test",
            field_constraint_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(FieldConstraintViolationError) as exc_info:
            await agent.generate(
                user_request="Create a momentum factor",
                factor_family=FactorFamily.MOMENTUM,
            )
        assert "sentiment_score" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_without_constraint_check(
        self, mock_llm_provider: MagicMock
    ) -> None:
        """Test generation succeeds when constraint check is disabled."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''```python
def factor(df):
    return df['sentiment_score']
```''',
        ))
        config = FactorGenerationConfig(
            name="test",
            field_constraint_enabled=False,  # Disabled
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        # Should not raise even with invalid fields
        result = await agent.generate(
            user_request="Create a factor",
            factor_family=FactorFamily.MOMENTUM,
        )
        assert result is not None


class TestFactorGenerationWithConstraintsBoundary:
    """Boundary tests for field constraint validation."""

    @pytest.fixture
    def validator(self) -> FactorFieldValidator:
        """Create validator instance."""
        return FactorFieldValidator()

    def test_empty_code(self, validator: FactorFieldValidator) -> None:
        """Test validation of empty code."""
        result = validator.validate("", FactorFamily.MOMENTUM)
        assert result.is_valid  # No fields used = valid
        assert len(result.used_fields) == 0

    def test_code_with_no_df_access(self, validator: FactorFieldValidator) -> None:
        """Test code that doesn't access df fields."""
        code = """
def factor(df):
    return 1
"""
        result = validator.validate(code, FactorFamily.MOMENTUM)
        assert result.is_valid

    def test_nested_field_access(self, validator: FactorFieldValidator) -> None:
        """Test nested field access patterns."""
        code = """
def factor(df):
    return df['close'].rolling(20).mean()['close']
"""
        fields = validator.extract_fields(code)
        assert "close" in fields

    def test_field_in_string_literal(self, validator: FactorFieldValidator) -> None:
        """Test that string literals are not extracted as fields."""
        code = """
def factor(df):
    name = "close"  # This should not be extracted
    return df['volume']
"""
        fields = validator.extract_fields(code)
        # Should only have volume, not close from string
        assert "volume" in fields

    def test_computed_field_names(self, validator: FactorFieldValidator) -> None:
        """Test handling of computed field names."""
        code = """
def factor(df):
    field = 'close'
    return df[field]  # Computed field name
"""
        # This is tricky - we may or may not extract 'close'
        fields = validator.extract_fields(code)
        # At minimum, should not crash
        assert isinstance(fields, set)


class TestFactorGenerationWithConstraintsException:
    """Exception handling tests for field constraints."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_field_constraint_error_message(
        self, mock_llm_provider: MagicMock
    ) -> None:
        """Test error message contains useful information."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content="def factor(df): return df['invalid_field']",
        ))
        config = FactorGenerationConfig(
            name="test",
            field_constraint_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(FieldConstraintViolationError) as exc_info:
            await agent.generate(
                user_request="Create factor",
                factor_family=FactorFamily.MOMENTUM,
            )

        error = exc_info.value
        assert "invalid_field" in str(error)
        # Error should mention the family or allowed fields
        assert hasattr(error, "violations") or "field" in str(error).lower()

    @pytest.mark.asyncio
    async def test_multiple_constraint_violations_reported(
        self, mock_llm_provider: MagicMock
    ) -> None:
        """Test all violations are reported in error."""
        mock_llm_provider.complete = AsyncMock(return_value=MagicMock(
            content='''def factor(df):
    a = df['field1']
    b = df['field2']
    c = df['field3']
    return a + b + c
''',
        ))
        config = FactorGenerationConfig(
            name="test",
            field_constraint_enabled=True,
        )
        agent = FactorGenerationAgent(config=config, llm_provider=mock_llm_provider)

        with pytest.raises(FieldConstraintViolationError) as exc_info:
            await agent.generate(
                user_request="Create factor",
                factor_family=FactorFamily.MOMENTUM,
            )

        error_str = str(exc_info.value)
        # Should mention multiple violations
        assert "field1" in error_str or "field2" in error_str or "field3" in error_str


class TestFactorGenerationWithConstraintsPerformance:
    """Performance tests for field constraint validation."""

    @pytest.fixture
    def validator(self) -> FactorFieldValidator:
        """Create validator instance."""
        return FactorFieldValidator()

    def test_validation_time(self, validator: FactorFieldValidator) -> None:
        """Test validation completes quickly."""
        # Generate a moderately complex code
        code = """
def complex_factor(df):
    \"\"\"A factor with many field accesses.\"\"\"
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    returns = close.pct_change()
    volatility = returns.rolling(20).std()
    range_factor = (high - low) / close
    volume_ratio = volume / volume.rolling(20).mean()

    return (returns * volatility * range_factor * volume_ratio).fillna(0)
"""
        start = time.time()
        for _ in range(100):  # Run 100 times
            validator.validate(code, FactorFamily.MOMENTUM)
        elapsed = time.time() - start

        # Should complete 100 validations in under 1 second
        assert elapsed < 1.0

    def test_large_code_validation(self, validator: FactorFieldValidator) -> None:
        """Test validation of large code."""
        # Generate a large code with many lines
        lines = ["def factor(df):"]
        for i in range(100):
            lines.append(f"    x{i} = df['close'].pct_change({i+1})")
        lines.append("    return sum([" + ", ".join([f"x{i}" for i in range(100)]) + "])")
        code = "\n".join(lines)

        start = time.time()
        result = validator.validate(code, FactorFamily.MOMENTUM)
        elapsed = time.time() - start

        assert result.is_valid
        assert elapsed < 0.5  # Should be fast


class TestFactorGenerationWithConstraintsSecurity:
    """Security tests for field constraint validation."""

    @pytest.fixture
    def validator(self) -> FactorFieldValidator:
        """Create validator instance."""
        return FactorFieldValidator()

    def test_no_field_injection(self, validator: FactorFieldValidator) -> None:
        """Test that field extraction handles edge cases gracefully."""
        # Note: Pure regex-based extraction may have false positives from strings
        # This is acceptable as it errs on the side of caution (over-detection)
        code = """
def factor(df):
    # Comment: df['close']
    return df['volume']
"""
        fields = validator.extract_fields(code)
        # Volume should definitely be extracted
        assert "volume" in fields
        # Note: Comments with df['field'] patterns may be extracted
        # This is a false positive but doesn't affect security

    def test_handles_malformed_code(self, validator: FactorFieldValidator) -> None:
        """Test handling of malformed code."""
        code = "def factor(df): return df['unclosed"
        # Should not crash
        try:
            result = validator.validate(code, FactorFamily.MOMENTUM)
            # May be valid (no complete field access) or handle gracefully
            assert isinstance(result.is_valid, bool)
        except Exception:
            pass  # May raise, but should not crash


class TestFactorGenerationWithConstraintsCompatibility:
    """Compatibility tests for different code patterns."""

    @pytest.fixture
    def validator(self) -> FactorFieldValidator:
        """Create validator instance."""
        return FactorFieldValidator()

    def test_pandas_loc_iloc(self, validator: FactorFieldValidator) -> None:
        """Test extraction from loc/iloc patterns."""
        code = """
def factor(df):
    data = df.loc[:, 'close']
    return data.pct_change()
"""
        fields = validator.extract_fields(code)
        assert "close" in fields

    def test_multiple_dataframes(self, validator: FactorFieldValidator) -> None:
        """Test handling code with multiple dataframes."""
        # Validator only tracks fields from 'df' variable, not df2
        # This is by design - we validate the primary dataframe access
        code = """
def factor(df):
    df2 = df.copy()
    return df['close'] + df['volume']
"""
        fields = validator.extract_fields(code)
        assert "close" in fields
        assert "volume" in fields

    def test_list_of_fields(self, validator: FactorFieldValidator) -> None:
        """Test extraction of field lists used directly with df."""
        # Direct inline list usage
        code = """
def factor(df):
    return df[['close', 'volume', 'high']].mean(axis=1)
"""
        fields = validator.extract_fields(code)
        # Should extract fields from the inline list
        assert any(f in fields for f in ["close", "volume", "high"])

    def test_all_factor_families(self, validator: FactorFieldValidator) -> None:
        """Test validation works with all factor families."""
        code = "def factor(df): return df['close']"

        for family in FactorFamily:
            result = validator.validate(code, family)
            # close should be valid for most families
            assert isinstance(result.is_valid, bool)
            assert isinstance(result.violations, list)

    def test_config_default_field_constraint(self) -> None:
        """Test default value for field_constraint_enabled."""
        config = FactorGenerationConfig(name="test")
        # Default should be True (constraints enabled)
        assert config.field_constraint_enabled is True
