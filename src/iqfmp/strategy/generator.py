"""Strategy Generator for multi-factor strategies.

This module provides:
- StrategyTemplate: Template for strategy code generation
- StrategyGenerator: Generates strategy code from factors
- StrategyValidator: Validates generated strategy code
- WeightingScheme: Factor weighting strategies
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class InvalidStrategyError(Exception):
    """Raised when strategy configuration is invalid."""

    pass


class WeightingScheme(Enum):
    """Factor weighting schemes."""

    EQUAL = "equal"
    CUSTOM = "custom"
    IC_WEIGHTED = "ic_weighted"

    def calculate(self, n_factors: int) -> list[float]:
        """Calculate equal weights for n factors.

        Args:
            n_factors: Number of factors

        Returns:
            List of equal weights summing to 1.0
        """
        if n_factors <= 0:
            return []
        weight = 1.0 / n_factors
        return [weight] * n_factors

    def calculate_from_ic(self, ic_values: list[float]) -> list[float]:
        """Calculate weights based on IC values.

        Args:
            ic_values: List of Information Coefficient values

        Returns:
            List of weights proportional to IC, summing to 1.0
        """
        if not ic_values:
            return []

        # Use absolute IC values for weighting
        abs_ic = [abs(ic) for ic in ic_values]
        total_ic = sum(abs_ic)

        if total_ic == 0:
            # Fall back to equal weights
            return self.calculate(len(ic_values))

        return [ic / total_ic for ic in abs_ic]


@dataclass
class StrategyConfig:
    """Configuration for strategy generation."""

    rebalance_frequency: str = "daily"
    max_position_size: float = 0.1
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    universe: str = "all"


@dataclass
class StrategyTemplate:
    """Template for strategy code generation."""

    name: str
    factors: list[str]
    weights: Optional[list[float]] = None
    description: str = ""

    def render(self) -> str:
        """Render the strategy template to Python code.

        Returns:
            Generated Python code as string
        """
        weights = self.weights or [1.0 / len(self.factors)] * len(self.factors)

        # Build factor weight assignments
        factor_weights = []
        for i, (factor, weight) in enumerate(zip(self.factors, weights)):
            factor_weights.append(f'            "{factor}": {weight},')

        factor_weights_str = "\n".join(factor_weights)

        code = f'''"""
{self.name} Strategy

Auto-generated multi-factor strategy.
Factors: {", ".join(self.factors)}
"""

import numpy as np
import pandas as pd


class {self._class_name()}Strategy:
    """Multi-factor strategy combining {len(self.factors)} factors."""

    def __init__(self):
        """Initialize strategy with factor weights."""
        self.factor_weights = {{
{factor_weights_str}
        }}
        self.factors = list(self.factor_weights.keys())

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from factor data.

        Args:
            data: DataFrame with factor columns

        Returns:
            Series of combined signals
        """
        signal = pd.Series(0.0, index=data.index)

        for factor, weight in self.factor_weights.items():
            if factor in data.columns:
                signal += data[factor] * weight

        return signal

    def get_positions(self, signal: pd.Series, top_n: int = 10) -> pd.Series:
        """Convert signals to positions.

        Args:
            signal: Combined factor signal
            top_n: Number of top positions to take

        Returns:
            Series of position weights
        """
        ranked = signal.rank(ascending=False)
        positions = (ranked <= top_n).astype(float)
        return positions / positions.sum()
'''
        return code

    def _class_name(self) -> str:
        """Convert strategy name to class name format."""
        # Convert snake_case or kebab-case to PascalCase
        words = re.split(r"[_\-\s]+", self.name)
        return "".join(word.capitalize() for word in words)


@dataclass
class ValidationResult:
    """Result of strategy code validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class StrategyValidator:
    """Validates generated strategy code for safety and correctness."""

    # Dangerous modules that should not be imported
    DANGEROUS_MODULES = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "pickle",
        "marshal",
        "shelve",
    }

    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
    }

    def validate(self, code: str) -> ValidationResult:
        """Validate strategy code.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult with is_valid flag and any errors
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e.msg}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for dangerous imports
        dangerous_imports = self._find_dangerous_imports(code)
        if dangerous_imports:
            errors.append(
                f"Dangerous imports detected: {', '.join(dangerous_imports)}"
            )

        # Check for dangerous built-in usage
        dangerous_builtins = self._find_dangerous_builtins(code)
        if dangerous_builtins:
            errors.append(
                f"Dangerous built-in usage: {', '.join(dangerous_builtins)}"
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def _find_dangerous_imports(self, code: str) -> list[str]:
        """Find dangerous module imports in code."""
        found = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in self.DANGEROUS_MODULES:
                            found.append(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split(".")[0]
                        if module in self.DANGEROUS_MODULES:
                            found.append(module)
        except SyntaxError:
            pass
        return found

    def _find_dangerous_builtins(self, code: str) -> list[str]:
        """Find dangerous built-in function calls in code."""
        found = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.DANGEROUS_BUILTINS:
                            found.append(node.func.id)
        except SyntaxError:
            pass
        return found


@dataclass
class GeneratedStrategy:
    """A generated strategy with code and metadata."""

    name: str
    code: str
    factors: list[dict[str, Any]]
    config: Optional[StrategyConfig] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert strategy to dictionary.

        Returns:
            Dictionary representation of strategy
        """
        return {
            "name": self.name,
            "code": self.code,
            "factors": self.factors,
            "config": {
                "rebalance_frequency": self.config.rebalance_frequency,
                "max_position_size": self.config.max_position_size,
                "use_stop_loss": self.config.use_stop_loss,
            }
            if self.config
            else None,
            "metadata": self.metadata,
        }

    def save(self, filepath: Path | str) -> None:
        """Save strategy code to file.

        Args:
            filepath: Path to save the strategy file
        """
        filepath = Path(filepath)
        filepath.write_text(self.code)

    def get_factor_names(self) -> list[str]:
        """Get list of factor names used in strategy.

        Returns:
            List of factor names
        """
        return [f["name"] for f in self.factors]


class StrategyGenerator:
    """Generates multi-factor strategy code from configuration."""

    def __init__(self) -> None:
        """Initialize the strategy generator."""
        self.validator = StrategyValidator()

    def generate(
        self,
        name: str,
        factors: list[dict[str, Any]],
        config: Optional[StrategyConfig] = None,
        qlib_compatible: bool = False,
        normalize_weights: bool = False,
        allow_negative_weights: bool = False,
    ) -> GeneratedStrategy:
        """Generate a multi-factor strategy.

        Args:
            name: Strategy name
            factors: List of factor dicts with 'name' and 'weight' keys
            config: Optional strategy configuration
            qlib_compatible: Whether to generate Qlib-compatible code
            normalize_weights: Whether to normalize weights to sum to 1
            allow_negative_weights: Whether to allow negative (short) weights

        Returns:
            GeneratedStrategy with code and metadata

        Raises:
            InvalidStrategyError: If configuration is invalid
        """
        # Validate factors
        if not factors:
            raise InvalidStrategyError("At least one factor is required")

        # Extract factor names and weights
        factor_names = [f["name"] for f in factors]
        weights = [f.get("weight", 1.0 / len(factors)) for f in factors]

        # Validate weights
        if not allow_negative_weights:
            for i, w in enumerate(weights):
                if w < 0:
                    raise InvalidStrategyError(
                        f"Negative weight for factor {factor_names[i]}"
                    )

        # Normalize weights if requested
        if normalize_weights:
            total = sum(abs(w) for w in weights)
            if total > 0:
                weights = [w / total for w in weights]

        # Create template and render
        template = StrategyTemplate(
            name=name,
            factors=factor_names,
            weights=weights,
        )

        if qlib_compatible:
            code = self._generate_qlib_code(name, factor_names, weights, config)
        else:
            code = template.render()

        # Validate generated code
        validation = self.validator.validate(code)
        if not validation.is_valid:
            raise InvalidStrategyError(
                f"Generated code validation failed: {validation.errors}"
            )

        return GeneratedStrategy(
            name=name,
            code=code,
            factors=factors,
            config=config,
        )

    def _generate_qlib_code(
        self,
        name: str,
        factors: list[str],
        weights: list[float],
        config: Optional[StrategyConfig] = None,
    ) -> str:
        """Generate Qlib-compatible strategy code.

        Args:
            name: Strategy name
            factors: List of factor names
            weights: List of factor weights
            config: Optional strategy configuration

        Returns:
            Qlib-compatible Python code
        """
        # Build factor weight assignments
        factor_weights = []
        for factor, weight in zip(factors, weights):
            factor_weights.append(f'            "{factor}": {weight},')

        factor_weights_str = "\n".join(factor_weights)

        rebalance = config.rebalance_frequency if config else "daily"

        code = f'''"""
{name} Strategy (Qlib Compatible)

Auto-generated multi-factor strategy for Qlib.
Factors: {", ".join(factors)}
"""

import numpy as np
import pandas as pd
from qlib.contrib.strategy import BaseStrategy


class {self._to_class_name(name)}Strategy(BaseStrategy):
    """Qlib-compatible multi-factor strategy."""

    def __init__(self, **kwargs):
        """Initialize strategy."""
        super().__init__(**kwargs)
        self.factor_weights = {{
{factor_weights_str}
        }}
        self.rebalance_frequency = "{rebalance}"

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals.

        Args:
            data: Factor data

        Returns:
            Combined signal series
        """
        signal = pd.Series(0.0, index=data.index)

        for factor, weight in self.factor_weights.items():
            if factor in data.columns:
                signal += data[factor] * weight

        return signal
'''
        return code

    def _to_class_name(self, name: str) -> str:
        """Convert name to PascalCase class name."""
        words = re.split(r"[_\-\s]+", name)
        return "".join(word.capitalize() for word in words)
