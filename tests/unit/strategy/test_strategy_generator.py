"""Tests for Strategy Generator (Task 15)."""

import pytest
from typing import Any

from iqfmp.strategy.generator import (
    StrategyGenerator,
    StrategyTemplate,
    StrategyConfig,
    WeightingScheme,
    GeneratedStrategy,
    StrategyValidator,
    InvalidStrategyError,
)


@pytest.fixture
def generator() -> StrategyGenerator:
    """Create strategy generator."""
    return StrategyGenerator()


@pytest.fixture
def sample_factors() -> list[dict[str, Any]]:
    """Sample factor configuration."""
    return [
        {"name": "momentum_20d", "weight": 0.4},
        {"name": "value_pb", "weight": 0.3},
        {"name": "volatility_30d", "weight": 0.3},
    ]


class TestStrategyTemplate:
    """Tests for StrategyTemplate."""

    def test_create_template(self) -> None:
        """Test creating strategy template."""
        template = StrategyTemplate(
            name="multi_factor",
            factors=["momentum", "value"],
        )
        assert template.name == "multi_factor"

    def test_template_render(self) -> None:
        """Test template rendering."""
        template = StrategyTemplate(
            name="test_strategy",
            factors=["momentum"],
        )
        code = template.render()
        assert "class" in code
        assert "Strategy" in code

    def test_template_with_weights(self) -> None:
        """Test template with factor weights."""
        template = StrategyTemplate(
            name="weighted_strategy",
            factors=["momentum", "value"],
            weights=[0.6, 0.4],
        )
        code = template.render()
        assert "0.6" in code or "0.4" in code


class TestWeightingScheme:
    """Tests for WeightingScheme."""

    def test_equal_weights(self) -> None:
        """Test equal weighting."""
        scheme = WeightingScheme.EQUAL
        weights = scheme.calculate(n_factors=4)
        assert len(weights) == 4
        assert all(w == 0.25 for w in weights)

    def test_custom_weights(self) -> None:
        """Test custom weighting."""
        scheme = WeightingScheme.CUSTOM
        weights = [0.5, 0.3, 0.2]
        assert sum(weights) == pytest.approx(1.0)

    def test_ic_weighted(self) -> None:
        """Test IC-based weighting."""
        scheme = WeightingScheme.IC_WEIGHTED
        ic_values = [0.05, 0.03, 0.02]
        weights = scheme.calculate_from_ic(ic_values)
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] > weights[1] > weights[2]


class TestStrategyGenerator:
    """Tests for StrategyGenerator."""

    def test_generate_basic(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test basic strategy generation."""
        result = generator.generate(
            name="test_strategy",
            factors=sample_factors,
        )
        assert isinstance(result, GeneratedStrategy)
        assert result.code is not None

    def test_generate_qlib_compatible(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test Qlib-compatible code generation."""
        result = generator.generate(
            name="qlib_strategy",
            factors=sample_factors,
            qlib_compatible=True,
        )
        assert "BaseStrategy" in result.code or "Strategy" in result.code

    def test_generate_with_config(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test generation with config."""
        config = StrategyConfig(
            rebalance_frequency="daily",
            max_position_size=0.1,
            use_stop_loss=True,
        )
        result = generator.generate(
            name="configured_strategy",
            factors=sample_factors,
            config=config,
        )
        assert result.config == config

    def test_generate_includes_imports(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test that generated code includes imports."""
        result = generator.generate(
            name="test_strategy",
            factors=sample_factors,
        )
        assert "import" in result.code

    def test_generate_includes_docstring(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test that generated code includes docstring."""
        result = generator.generate(
            name="documented_strategy",
            factors=sample_factors,
        )
        assert '"""' in result.code or "'''" in result.code


class TestStrategyValidator:
    """Tests for StrategyValidator."""

    def test_validate_valid_code(self) -> None:
        """Test validating valid code."""
        validator = StrategyValidator()
        code = '''
class TestStrategy:
    def __init__(self):
        self.factors = []

    def generate_signal(self, data):
        return 0
'''
        result = validator.validate(code)
        assert result.is_valid

    def test_validate_syntax_error(self) -> None:
        """Test detecting syntax errors."""
        validator = StrategyValidator()
        code = "def broken( :"
        result = validator.validate(code)
        assert not result.is_valid
        assert "syntax" in result.errors[0].lower()

    def test_validate_no_dangerous_imports(self) -> None:
        """Test detecting dangerous imports."""
        validator = StrategyValidator()
        code = "import os\nos.system('rm -rf /')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_validate_no_eval(self) -> None:
        """Test detecting eval usage."""
        validator = StrategyValidator()
        code = "eval('print(1)')"
        result = validator.validate(code)
        assert not result.is_valid


class TestGeneratedStrategy:
    """Tests for GeneratedStrategy."""

    def test_strategy_to_dict(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test strategy serialization."""
        result = generator.generate(
            name="test_strategy",
            factors=sample_factors,
        )
        d = result.to_dict()
        assert "name" in d
        assert "code" in d
        assert "factors" in d

    def test_strategy_save_to_file(
        self, generator: StrategyGenerator, sample_factors: list[dict], tmp_path
    ) -> None:
        """Test saving strategy to file."""
        result = generator.generate(
            name="test_strategy",
            factors=sample_factors,
        )
        filepath = tmp_path / "strategy.py"
        result.save(filepath)
        assert filepath.exists()

    def test_strategy_get_factor_names(
        self, generator: StrategyGenerator, sample_factors: list[dict]
    ) -> None:
        """Test getting factor names."""
        result = generator.generate(
            name="test_strategy",
            factors=sample_factors,
        )
        names = result.get_factor_names()
        assert "momentum_20d" in names


class TestStrategyBoundary:
    """Boundary tests."""

    def test_single_factor(self, generator: StrategyGenerator) -> None:
        """Test with single factor."""
        result = generator.generate(
            name="single_factor",
            factors=[{"name": "momentum", "weight": 1.0}],
        )
        assert result.code is not None

    def test_many_factors(self, generator: StrategyGenerator) -> None:
        """Test with many factors."""
        factors = [
            {"name": f"factor_{i}", "weight": 1/20}
            for i in range(20)
        ]
        result = generator.generate(
            name="many_factors",
            factors=factors,
        )
        assert result.code is not None

    def test_empty_factors(self, generator: StrategyGenerator) -> None:
        """Test with empty factors."""
        with pytest.raises(InvalidStrategyError):
            generator.generate(
                name="empty_strategy",
                factors=[],
            )


class TestStrategyException:
    """Exception handling tests."""

    def test_invalid_weight_sum(self, generator: StrategyGenerator) -> None:
        """Test invalid weight sum."""
        factors = [
            {"name": "factor1", "weight": 0.5},
            {"name": "factor2", "weight": 0.3},
            # Sum = 0.8, not 1.0
        ]
        # Should normalize or warn
        result = generator.generate(
            name="test",
            factors=factors,
            normalize_weights=True,
        )
        assert result is not None

    def test_negative_weight(self, generator: StrategyGenerator) -> None:
        """Test negative weight handling."""
        factors = [
            {"name": "factor1", "weight": -0.5},  # Short factor
            {"name": "factor2", "weight": 1.5},
        ]
        result = generator.generate(
            name="long_short",
            factors=factors,
            allow_negative_weights=True,
        )
        assert result is not None


class TestStrategyPerformance:
    """Performance tests."""

    def test_generation_speed(self, generator: StrategyGenerator) -> None:
        """Test generation speed."""
        import time

        factors = [{"name": f"f_{i}", "weight": 0.1} for i in range(10)]

        start = time.time()
        for _ in range(100):
            generator.generate(name="perf_test", factors=factors)
        elapsed = time.time() - start

        assert elapsed < 2.0
