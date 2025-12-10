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
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from scipy import optimize


class InvalidStrategyError(Exception):
    """Raised when strategy configuration is invalid."""

    pass


class InsufficientFactorsError(Exception):
    """Raised when not enough factors are provided for strategy generation."""

    pass


class CombinationMethod(str, Enum):
    """Factor combination methods for multi-factor strategies."""

    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHT = "ic_weight"
    OPTIMIZATION = "optimization"
    RANK_IC_WEIGHT = "rank_ic_weight"


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

    def generate_from_factors(
        self,
        factors: list[dict[str, Any]],
        combination_method: str | CombinationMethod = CombinationMethod.EQUAL_WEIGHT,
        min_diversity_ratio: float = 0.5,
        target_volatility: float = 0.15,
    ) -> "Strategy":
        """从验证通过的因子生成策略 (Task 8.1).

        Args:
            factors: Factor dictionaries with id, name, family, code, cluster_id, metrics
            combination_method: 组合方法 (equal_weight, ic_weight, optimization)
            min_diversity_ratio: 最小多样性比率 (不同 cluster 占比)
            target_volatility: 目标波动率 (用于优化方法)

        Returns:
            Strategy object with weights and Qlib config

        Raises:
            InsufficientFactorsError: If fewer than 2 factors provided
        """
        if len(factors) < 2:
            raise InsufficientFactorsError(
                f"At least 2 factors required for multi-factor strategy, got {len(factors)}"
            )

        # 1. 检查因子多样性 (不同 cluster)
        diversity_info = self._check_factor_diversity(factors, min_diversity_ratio)
        if not diversity_info["is_diverse"]:
            warnings.warn(
                f"因子多样性不足 (cluster 覆盖率 {diversity_info['cluster_ratio']:.2%})，"
                f"建议选择不同聚类的因子以降低相关性"
            )

        # 2. 根据组合方法计算权重
        if isinstance(combination_method, str):
            combination_method = CombinationMethod(combination_method)

        if combination_method == CombinationMethod.EQUAL_WEIGHT:
            weights = self._equal_weights(factors)
        elif combination_method == CombinationMethod.IC_WEIGHT:
            weights = self._ic_weighted(factors)
        elif combination_method == CombinationMethod.RANK_IC_WEIGHT:
            weights = self._rank_ic_weighted(factors)
        elif combination_method == CombinationMethod.OPTIMIZATION:
            weights = self._optimize_weights(factors, target_volatility)
        else:
            weights = self._equal_weights(factors)

        # 3. 生成 Qlib 策略配置
        qlib_config = self._generate_qlib_strategy_config(factors, weights)

        # 4. 生成策略代码
        factor_dicts = [
            {"name": f.get("name", f"factor_{i}"), "weight": weights[f.get("id", str(i))]}
            for i, f in enumerate(factors)
        ]
        generated = self.generate(
            name=f"combined_{len(factors)}factors",
            factors=factor_dicts,
            qlib_compatible=True,
            normalize_weights=True,
        )

        # 5. 创建策略记录
        return Strategy(
            id=str(uuid4()),
            name=f"combined_{len(factors)}factors",
            description=f"Multi-factor strategy combining {len(factors)} factors using {combination_method.value}",
            factor_weights=weights,
            qlib_config=qlib_config,
            code=generated.code,
            factors=[f.get("id", str(i)) for i, f in enumerate(factors)],
            combination_method=combination_method.value,
            diversity_info=diversity_info,
            created_at=datetime.now(),
        )

    def _check_factor_diversity(
        self, factors: list[dict[str, Any]], min_ratio: float = 0.5
    ) -> dict[str, Any]:
        """检查因子多样性 (不同 cluster 占比).

        Args:
            factors: Factor dictionaries
            min_ratio: Minimum ratio of unique clusters

        Returns:
            Diversity info dict with is_diverse, cluster_ratio, unique_clusters
        """
        cluster_ids = [f.get("cluster_id") for f in factors if f.get("cluster_id")]
        unique_clusters = set(cluster_ids)

        # 如果没有 cluster 信息，假设多样性足够
        if not cluster_ids:
            return {
                "is_diverse": True,
                "cluster_ratio": 1.0,
                "unique_clusters": 0,
                "total_with_cluster": 0,
                "message": "No cluster information available",
            }

        cluster_ratio = len(unique_clusters) / len(cluster_ids)
        return {
            "is_diverse": cluster_ratio >= min_ratio,
            "cluster_ratio": cluster_ratio,
            "unique_clusters": len(unique_clusters),
            "total_with_cluster": len(cluster_ids),
            "message": (
                f"Good diversity: {len(unique_clusters)} unique clusters"
                if cluster_ratio >= min_ratio
                else f"Low diversity: only {len(unique_clusters)} unique clusters"
            ),
        }

    def _equal_weights(self, factors: list[dict[str, Any]]) -> dict[str, float]:
        """计算等权重."""
        weight = 1.0 / len(factors)
        return {f.get("id", str(i)): weight for i, f in enumerate(factors)}

    def _ic_weighted(self, factors: list[dict[str, Any]]) -> dict[str, float]:
        """基于 IC 值的权重计算.

        权重与 |IC| 成正比，确保高 IC 因子获得更高权重。
        """
        ic_values = {}
        for i, f in enumerate(factors):
            factor_id = f.get("id", str(i))
            metrics = f.get("metrics") or {}
            # 支持多种 IC 字段名
            ic = metrics.get("ic_mean") or metrics.get("ic") or metrics.get("mean_ic") or 0.0
            ic_values[factor_id] = abs(float(ic))

        total_ic = sum(ic_values.values())

        # 如果所有 IC 都是 0，回退到等权重
        if total_ic == 0:
            return self._equal_weights(factors)

        return {fid: ic / total_ic for fid, ic in ic_values.items()}

    def _rank_ic_weighted(self, factors: list[dict[str, Any]]) -> dict[str, float]:
        """基于 Rank IC 的权重计算.

        适用于截面策略，使用 Rank IC 而非 IC。
        """
        ic_values = {}
        for i, f in enumerate(factors):
            factor_id = f.get("id", str(i))
            metrics = f.get("metrics") or {}
            # 优先使用 rank_ic，否则回退到 ic
            ic = metrics.get("rank_ic") or metrics.get("ic_mean") or 0.0
            ic_values[factor_id] = abs(float(ic))

        total_ic = sum(ic_values.values())

        if total_ic == 0:
            return self._equal_weights(factors)

        return {fid: ic / total_ic for fid, ic in ic_values.items()}

    def _optimize_weights(
        self,
        factors: list[dict[str, Any]],
        target_volatility: float = 0.15,
    ) -> dict[str, float]:
        """使用均值-方差优化计算权重.

        最大化夏普比率，目标波动率约束。

        Args:
            factors: Factor dictionaries with metrics
            target_volatility: Target portfolio volatility

        Returns:
            Optimized weights dict
        """
        n = len(factors)

        # 提取收益和波动率估计
        returns = []
        volatilities = []
        for f in factors:
            metrics = f.get("metrics") or {}
            # 使用 IC 作为预期收益的代理
            ic = metrics.get("ic_mean") or metrics.get("ic") or 0.02
            sharpe = metrics.get("sharpe") or 1.0
            vol = abs(ic / sharpe) if sharpe != 0 else 0.1
            returns.append(float(ic))
            volatilities.append(float(vol))

        returns = np.array(returns)
        volatilities = np.array(volatilities)

        # 简化假设：因子之间的相关性为 0.3 (保守估计)
        correlation = 0.3
        cov_matrix = np.outer(volatilities, volatilities) * (
            np.eye(n) + (1 - np.eye(n)) * correlation
        )

        def neg_sharpe(w):
            """Negative Sharpe ratio (for minimization)."""
            port_return = np.dot(w, returns)
            port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            return -port_return / port_vol if port_vol > 0 else 0

        # 约束条件
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 权重和为 1
        ]

        # 边界条件：0 <= w <= 0.5 (单因子最大 50%)
        bounds = [(0, 0.5) for _ in range(n)]

        # 初始等权重
        w0 = np.ones(n) / n

        try:
            result = optimize.minimize(
                neg_sharpe,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100},
            )

            if result.success:
                weights_arr = result.x
                # 确保权重非负并归一化
                weights_arr = np.maximum(weights_arr, 0)
                weights_arr = weights_arr / np.sum(weights_arr)
            else:
                # 优化失败，回退到 IC 加权
                return self._ic_weighted(factors)

        except Exception:
            # 任何错误都回退到 IC 加权
            return self._ic_weighted(factors)

        return {
            factors[i].get("id", str(i)): float(weights_arr[i])
            for i in range(n)
        }

    def _generate_qlib_strategy_config(
        self, factors: list[dict[str, Any]], weights: dict[str, float]
    ) -> dict[str, Any]:
        """生成 Qlib 策略配置.

        Args:
            factors: Factor dictionaries
            weights: Factor weights

        Returns:
            Qlib-compatible strategy configuration
        """
        factor_configs = []
        for i, f in enumerate(factors):
            factor_id = f.get("id", str(i))
            factor_configs.append({
                "name": f.get("name", factor_id),
                "expression": f.get("code", ""),
                "weight": weights.get(factor_id, 1.0 / len(factors)),
                "family": f.get("family", []),
            })

        return {
            "model": {
                "class": "LinearModel",
                "module_path": "qlib.contrib.model.linear",
                "kwargs": {},
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                    },
                    "segments": {
                        "train": ("2010-01-01", "2019-12-31"),
                        "valid": ("2020-01-01", "2020-12-31"),
                        "test": ("2021-01-01", "2022-12-31"),
                    },
                },
            },
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "topk": 30,
                    "n_drop": 5,
                },
            },
            "backtest": {
                "start_time": "2021-01-01",
                "end_time": "2022-12-31",
                "account": 100000000,
                "benchmark": "SH000300",
                "exchange_kwargs": {
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
            "factors": factor_configs,
        }


@dataclass
class Strategy:
    """Strategy entity for multi-factor combination (Task 8.1)."""

    id: str
    name: str
    factor_weights: dict[str, float]
    qlib_config: dict[str, Any]
    code: str
    factors: list[str]
    combination_method: str
    created_at: datetime
    description: str = ""
    diversity_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert strategy to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "factor_weights": self.factor_weights,
            "qlib_config": self.qlib_config,
            "code": self.code,
            "factors": self.factors,
            "combination_method": self.combination_method,
            "diversity_info": self.diversity_info,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def get_weight(self, factor_id: str) -> float:
        """Get weight for a specific factor."""
        return self.factor_weights.get(factor_id, 0.0)

    def get_top_factors(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N factors by weight."""
        sorted_weights = sorted(
            self.factor_weights.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_weights[:n]
