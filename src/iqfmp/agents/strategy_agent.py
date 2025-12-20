"""Strategy Assembly Agent for IQFMP.

Assembles multiple factors into trading strategies with weighting,
combining methods, and portfolio construction rules, enhanced with LLM.

LLM Integration:
- Uses frontend-configured model from ConfigService via model_config.py
- LLM provides intelligent strategy design recommendations
- Suggests optimal factor combinations and weighting approaches

Six-dimensional coverage:
1. Functional: Factor combination, weight optimization, signal generation
2. Boundary: Single factor, many factors, extreme weights
3. Exception: Invalid factors, missing data, weight constraints
4. Performance: Efficient matrix operations
5. Security: Weight constraint validation
6. Compatibility: Multiple combination methods
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol
import json
import logging
import re

import numpy as np
import pandas as pd

from iqfmp.agents.orchestrator import AgentState


# =============================================================================
# LLM Integration Protocol and System Prompts
# =============================================================================

class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider interface."""

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate completion from the LLM."""
        ...


STRATEGY_SYSTEM_PROMPT = """You are an expert portfolio manager specializing in multi-factor strategy design.

Your task is to analyze factor combinations and provide strategic recommendations:
1. Suggest optimal factor weighting approaches based on factor characteristics
2. Recommend portfolio construction methods suited to the factors
3. Identify potential risks from factor correlations
4. Propose risk management overlays

Consider:
- Factor IC and IR values for quality assessment
- Factor correlations to avoid redundancy
- Market regime sensitivity
- Transaction costs and turnover implications
- Risk-adjusted return optimization

Provide specific, actionable strategy recommendations."""


logger = logging.getLogger(__name__)


class StrategyAgentError(Exception):
    """Base error for strategy agent failures."""

    pass


class InsufficientFactorsError(StrategyAgentError):
    """Raised when not enough factors to build strategy."""

    pass


class WeightConstraintError(StrategyAgentError):
    """Raised when weight constraints are violated."""

    pass


class CombinationMethod(Enum):
    """Methods for combining factor signals."""

    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHTED = "ic_weighted"
    IR_WEIGHTED = "ir_weighted"
    RANK_WEIGHTED = "rank_weighted"
    CUSTOM = "custom"


class SignalTransform(Enum):
    """Transformations applied to factor signals."""

    RAW = "raw"
    ZSCORE = "zscore"
    RANK = "rank"
    PERCENTILE = "percentile"
    WINSORIZE = "winsorize"


class PortfolioConstruction(Enum):
    """Portfolio construction methods."""

    LONG_SHORT = "long_short"
    LONG_ONLY = "long_only"
    TOP_N = "top_n"
    THRESHOLD = "threshold"


@dataclass
class StrategyConfig:
    """Configuration for strategy assembly."""

    # Factor combination
    combination_method: CombinationMethod = CombinationMethod.IC_WEIGHTED
    signal_transform: SignalTransform = SignalTransform.ZSCORE

    # Weight constraints
    min_factor_weight: float = 0.05
    max_factor_weight: float = 0.5
    max_factors: int = 10

    # Portfolio construction
    construction_method: PortfolioConstruction = PortfolioConstruction.LONG_SHORT
    top_n: int = 10
    signal_threshold: float = 1.0  # Z-score threshold

    # Risk constraints
    max_position_size: float = 0.1
    max_sector_exposure: float = 0.3
    max_turnover: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "combination_method": self.combination_method.value,
            "signal_transform": self.signal_transform.value,
            "min_factor_weight": self.min_factor_weight,
            "max_factor_weight": self.max_factor_weight,
            "max_factors": self.max_factors,
            "construction_method": self.construction_method.value,
            "top_n": self.top_n,
            "signal_threshold": self.signal_threshold,
            "max_position_size": self.max_position_size,
            "max_sector_exposure": self.max_sector_exposure,
            "max_turnover": self.max_turnover,
        }


@dataclass
class FactorWeight:
    """Weight assignment for a factor."""

    factor_name: str
    weight: float
    ic: float = 0.0
    ir: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor_name": self.factor_name,
            "weight": self.weight,
            "ic": self.ic,
            "ir": self.ir,
        }


@dataclass
class StrategySignal:
    """Generated strategy signal."""

    date: str
    symbol: str
    combined_signal: float
    position: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "symbol": self.symbol,
            "combined_signal": self.combined_signal,
            "position": self.position,
            "rank": self.rank,
        }


@dataclass
class StrategyResult:
    """Result from strategy assembly."""

    strategy_name: str
    factor_weights: list[FactorWeight]
    n_factors: int
    combination_method: str
    construction_method: str
    signals: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "strategy_name": self.strategy_name,
            "factor_weights": [fw.to_dict() for fw in self.factor_weights],
            "n_factors": self.n_factors,
            "combination_method": self.combination_method,
            "construction_method": self.construction_method,
        }


class StrategyAssemblyAgent:
    """Agent for assembling multi-factor trading strategies with LLM support.

    This agent combines evaluated factors into trading strategies
    using various weighting and portfolio construction methods,
    enhanced with LLM-powered strategy design recommendations.

    Responsibilities:
    - Select best factors from evaluation results
    - Calculate optimal factor weights
    - Combine factor signals
    - Generate portfolio positions
    - Apply risk constraints
    - Generate LLM-powered strategy recommendations

    LLM Integration:
    - Uses frontend-configured model via get_agent_full_config("strategy_assembly")
    - Provides intelligent strategy design insights
    - Recommends optimal factor combinations

    Usage:
        agent = StrategyAssemblyAgent(config, llm_provider=llm)
        new_state = await agent.assemble(state)
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        llm_provider: Optional[LLMProviderProtocol] = None,
    ) -> None:
        """Initialize the strategy assembly agent with LLM support.

        Args:
            config: Strategy configuration
            llm_provider: LLM provider for AI-powered recommendations
        """
        self.config = config or StrategyConfig()
        self.llm_provider = llm_provider

    async def generate_llm_recommendations(
        self,
        factor_weights: list[FactorWeight],
        evaluation_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate LLM-powered strategy recommendations.

        Args:
            factor_weights: Calculated factor weights
            evaluation_results: Evaluation results for each factor

        Returns:
            Dict with strategy insights and recommendations
        """
        if self.llm_provider is None:
            return self._generate_template_recommendations(factor_weights)

        # Get model configuration from ConfigService
        from iqfmp.agents.model_config import get_agent_full_config
        model_id, temperature, custom_system_prompt = get_agent_full_config("strategy_assembly")

        # Build analysis prompt
        factors_summary = "\n".join([
            f"- {fw.factor_name}: weight={fw.weight:.2%}, IC={fw.ic:.4f}, IR={fw.ir:.2f}"
            for fw in factor_weights
        ])

        prompt = f"""Analyze the following multi-factor strategy configuration:

Factors in Strategy:
{factors_summary}

Combination Method: {self.config.combination_method.value}
Portfolio Construction: {self.config.construction_method.value}
Max Position Size: {self.config.max_position_size:.1%}
Max Factor Weight: {self.config.max_factor_weight:.1%}

Provide:
1. Assessment of the factor combination quality (2-3 sentences)
2. Potential risks from this factor mix
3. 2-3 specific recommendations for optimization

Format as JSON:
```json
{{
    "assessment": "Your assessment...",
    "risks": ["Risk 1", "Risk 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"]
}}
```"""

        try:
            system_prompt = custom_system_prompt or STRATEGY_SYSTEM_PROMPT

            # Prefer schema-validated structured output when using the native LLMProvider.
            from iqfmp.llm.provider import LLMProvider
            from iqfmp.llm.validation.json_schema import OutputType

            if isinstance(self.llm_provider, LLMProvider):
                _resp, validation = await self.llm_provider.complete_structured(
                    prompt=prompt,
                    output_type=OutputType.STRATEGY_RECOMMENDATIONS,
                    model=model_id,
                    temperature=temperature,
                    max_tokens=1024,
                    system_prompt=system_prompt,
                )
                if validation.is_valid and isinstance(validation.data, dict):
                    return validation.data

            response = await self.llm_provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=1024,
            )

            return self._parse_llm_recommendations(response.content)

        except Exception as e:
            logger.error(f"LLM recommendations failed: {e}. Using template.")
            return self._generate_template_recommendations(factor_weights)

    def _parse_llm_recommendations(self, response_text: str) -> dict[str, Any]:
        """Parse LLM recommendations response."""
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "assessment": response_text[:300],
            "risks": [],
            "recommendations": [],
        }

    def _generate_template_recommendations(
        self,
        factor_weights: list[FactorWeight],
    ) -> dict[str, Any]:
        """Generate template-based recommendations when LLM is unavailable."""
        n_factors = len(factor_weights)
        avg_ic = sum(abs(fw.ic) for fw in factor_weights) / n_factors if n_factors > 0 else 0

        assessment = f"Strategy combines {n_factors} factors with average IC of {avg_ic:.4f}."
        risks = ["Factor correlation may reduce diversification benefit"]
        recommendations = [
            "Monitor factor performance stability over time",
            "Consider adding regime-based factor rotation",
        ]

        if n_factors < 3:
            risks.append("Low number of factors increases concentration risk")
            recommendations.append("Add more uncorrelated factors for diversification")

        return {
            "assessment": assessment,
            "risks": risks,
            "recommendations": recommendations,
        }

    async def assemble(self, state: AgentState) -> AgentState:
        """Assemble strategy from evaluated factors.

        This is the main entry point for StateGraph integration.

        Args:
            state: Current agent state containing:
                - context["factors_passed"]: List of factor names that passed evaluation
                - context["evaluation_results"]: Evaluation results with metrics
                - context["factor_data"]: DataFrame with factor values

        Returns:
            Updated state with strategy in context["strategy_result"]

        Raises:
            StrategyAgentError: On assembly failure
        """
        logger.info("StrategyAssemblyAgent: Starting assembly")

        context = state.context
        factors_passed = context.get("factors_passed", [])
        evaluation_results = context.get("evaluation_results", [])
        factor_data = context.get("factor_data")

        if not factors_passed:
            logger.warning("No passed factors for strategy assembly")
            return state.update(
                context={
                    **context,
                    "strategy_result": None,
                    "strategy_error": "No factors passed evaluation",
                }
            )

        if len(factors_passed) < 1:
            raise InsufficientFactorsError(
                "At least 1 factor required for strategy"
            )

        # Select top factors
        selected_factors = self._select_factors(
            factors_passed, evaluation_results
        )

        # Calculate weights
        factor_weights = self._calculate_weights(
            selected_factors, evaluation_results
        )

        # Validate weights
        self._validate_weights(factor_weights)

        # Generate combined signal if data available
        signals = None
        if factor_data is not None:
            signals = self._generate_signals(factor_weights, factor_data)

        # Build result
        strategy_result = StrategyResult(
            strategy_name=self._generate_strategy_name(selected_factors),
            factor_weights=factor_weights,
            n_factors=len(factor_weights),
            combination_method=self.config.combination_method.value,
            construction_method=self.config.construction_method.value,
            signals=signals,
        )

        # Update state
        new_context = {
            **context,
            "strategy_result": strategy_result.to_dict(),
            "factor_weights": [fw.to_dict() for fw in factor_weights],
            "strategy_signals": signals.to_dict("records") if signals is not None else None,
        }

        logger.info(
            f"StrategyAssemblyAgent: Completed. "
            f"Strategy: {strategy_result.strategy_name}, "
            f"Factors: {len(factor_weights)}"
        )

        return state.update(context=new_context)

    def _select_factors(
        self,
        factors_passed: list[str],
        evaluation_results: list[dict[str, Any]],
    ) -> list[str]:
        """Select top factors for the strategy.

        Args:
            factors_passed: List of factor names that passed
            evaluation_results: Full evaluation results

        Returns:
            Selected factor names
        """
        # Build metrics lookup
        metrics_map = {}
        for result in evaluation_results:
            name = result.get("factor_name")
            if name in factors_passed:
                metrics = result.get("metrics", {})
                metrics_map[name] = {
                    "ic": abs(metrics.get("ic", 0)),
                    "ir": metrics.get("ir", 0),
                }

        # Sort by IC or IR based on combination method
        if self.config.combination_method == CombinationMethod.IR_WEIGHTED:
            sorted_factors = sorted(
                metrics_map.items(),
                key=lambda x: x[1]["ir"],
                reverse=True,
            )
        else:
            sorted_factors = sorted(
                metrics_map.items(),
                key=lambda x: x[1]["ic"],
                reverse=True,
            )

        # Limit to max factors
        selected = [f[0] for f in sorted_factors[: self.config.max_factors]]

        logger.info(f"Selected {len(selected)} factors: {selected}")
        return selected

    def _calculate_weights(
        self,
        selected_factors: list[str],
        evaluation_results: list[dict[str, Any]],
    ) -> list[FactorWeight]:
        """Calculate factor weights.

        Args:
            selected_factors: Selected factor names
            evaluation_results: Full evaluation results

        Returns:
            List of factor weights
        """
        # Build metrics lookup
        metrics_map = {}
        for result in evaluation_results:
            name = result.get("factor_name")
            if name in selected_factors:
                metrics = result.get("metrics", {})
                metrics_map[name] = {
                    "ic": metrics.get("ic", 0),
                    "ir": metrics.get("ir", 0),
                }

        n = len(selected_factors)

        if self.config.combination_method == CombinationMethod.EQUAL_WEIGHT:
            weights = {f: 1.0 / n for f in selected_factors}

        elif self.config.combination_method == CombinationMethod.IC_WEIGHTED:
            ics = [abs(metrics_map[f]["ic"]) for f in selected_factors]
            total_ic = sum(ics) or 1.0
            weights = {
                f: abs(metrics_map[f]["ic"]) / total_ic
                for f in selected_factors
            }

        elif self.config.combination_method == CombinationMethod.IR_WEIGHTED:
            irs = [max(0, metrics_map[f]["ir"]) for f in selected_factors]
            total_ir = sum(irs) or 1.0
            weights = {
                f: max(0, metrics_map[f]["ir"]) / total_ir
                for f in selected_factors
            }

        else:  # RANK_WEIGHTED or CUSTOM
            # Use rank-based weights (higher IC = higher rank = higher weight)
            ics = [(f, abs(metrics_map[f]["ic"])) for f in selected_factors]
            ics.sort(key=lambda x: x[1], reverse=True)
            rank_weights = {f: n - i for i, (f, _) in enumerate(ics)}
            total_rank = sum(rank_weights.values())
            weights = {f: rank_weights[f] / total_rank for f in selected_factors}

        # Apply min/max constraints
        weights = self._apply_weight_constraints(weights)

        # Create FactorWeight objects
        factor_weights = []
        for factor_name, weight in weights.items():
            metrics = metrics_map.get(factor_name, {})
            factor_weights.append(
                FactorWeight(
                    factor_name=factor_name,
                    weight=weight,
                    ic=metrics.get("ic", 0),
                    ir=metrics.get("ir", 0),
                )
            )

        return factor_weights

    def _apply_weight_constraints(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Apply min/max weight constraints.

        Args:
            weights: Raw factor weights

        Returns:
            Constrained weights
        """
        constrained = {}

        # Apply bounds
        for factor, weight in weights.items():
            constrained[factor] = max(
                self.config.min_factor_weight,
                min(self.config.max_factor_weight, weight),
            )

        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {f: w / total for f, w in constrained.items()}

        return constrained

    def _validate_weights(self, factor_weights: list[FactorWeight]) -> None:
        """Validate factor weights.

        Args:
            factor_weights: List of factor weights

        Raises:
            WeightConstraintError: If constraints violated
        """
        total = sum(fw.weight for fw in factor_weights)

        if abs(total - 1.0) > 0.01:
            raise WeightConstraintError(
                f"Weights must sum to 1.0, got {total:.4f}"
            )

        for fw in factor_weights:
            if fw.weight < 0:
                raise WeightConstraintError(
                    f"Negative weight for {fw.factor_name}: {fw.weight}"
                )

    def _generate_signals(
        self,
        factor_weights: list[FactorWeight],
        factor_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate combined strategy signals.

        Args:
            factor_weights: Factor weight assignments
            factor_data: DataFrame with factor values

        Returns:
            DataFrame with combined signals
        """
        if factor_data is None or factor_data.empty:
            return pd.DataFrame()

        df = factor_data.copy()

        # Initialize combined signal
        df["combined_signal"] = 0.0

        # Combine factors
        for fw in factor_weights:
            factor_col = f"factor_{fw.factor_name}"
            if factor_col in df.columns:
                # Transform signal
                signal = self._transform_signal(df[factor_col])
                df["combined_signal"] += fw.weight * signal

        # Apply signal transform to combined
        df["combined_signal"] = self._transform_signal(df["combined_signal"])

        # Generate positions based on construction method
        df["position"] = self._generate_positions(df["combined_signal"])

        # Add ranking
        df["rank"] = df["combined_signal"].rank(ascending=False, method="first")

        return df[["combined_signal", "position", "rank"]]

    def _transform_signal(self, signal: pd.Series) -> pd.Series:
        """Apply signal transformation.

        Args:
            signal: Raw signal series

        Returns:
            Transformed signal
        """
        if self.config.signal_transform == SignalTransform.RAW:
            return signal

        elif self.config.signal_transform == SignalTransform.ZSCORE:
            mean = signal.mean()
            std = signal.std()
            if std > 0:
                return (signal - mean) / std
            return signal - mean

        elif self.config.signal_transform == SignalTransform.RANK:
            return signal.rank(pct=True) - 0.5

        elif self.config.signal_transform == SignalTransform.PERCENTILE:
            return signal.rank(pct=True)

        elif self.config.signal_transform == SignalTransform.WINSORIZE:
            lower = signal.quantile(0.01)
            upper = signal.quantile(0.99)
            return signal.clip(lower, upper)

        return signal

    def _generate_positions(self, signal: pd.Series) -> pd.Series:
        """Generate portfolio positions from signals.

        Args:
            signal: Combined signal series

        Returns:
            Position series
        """
        if self.config.construction_method == PortfolioConstruction.LONG_SHORT:
            # Long positive signals, short negative signals
            positions = np.sign(signal) * np.abs(signal)
            # Normalize to [-1, 1]
            max_abs = positions.abs().max()
            if max_abs > 0:
                positions = positions / max_abs
            return positions

        elif self.config.construction_method == PortfolioConstruction.LONG_ONLY:
            # Long top signals only
            positions = signal.clip(lower=0)
            total = positions.sum()
            if total > 0:
                positions = positions / total
            return positions

        elif self.config.construction_method == PortfolioConstruction.TOP_N:
            # Equal weight top N
            n = min(self.config.top_n, len(signal))
            positions = pd.Series(0.0, index=signal.index)
            top_idx = signal.nlargest(n).index
            positions.loc[top_idx] = 1.0 / n
            return positions

        elif self.config.construction_method == PortfolioConstruction.THRESHOLD:
            # Position only if signal exceeds threshold
            positions = pd.Series(0.0, index=signal.index)
            positions[signal > self.config.signal_threshold] = 1.0
            positions[signal < -self.config.signal_threshold] = -1.0
            # Normalize
            n_positions = (positions != 0).sum()
            if n_positions > 0:
                positions = positions / n_positions
            return positions

        return pd.Series(0.0, index=signal.index)

    def _generate_strategy_name(self, factors: list[str]) -> str:
        """Generate a descriptive strategy name.

        Args:
            factors: List of factor names

        Returns:
            Strategy name
        """
        n = len(factors)
        method = self.config.combination_method.value
        construction = self.config.construction_method.value

        if n == 1:
            return f"Single_Factor_{factors[0]}_{construction}"
        else:
            return f"Multi_Factor_{n}_{method}_{construction}"

    def assemble_from_factors(
        self,
        factor_names: list[str],
        factor_metrics: dict[str, dict[str, float]],
        factor_data: Optional[pd.DataFrame] = None,
    ) -> StrategyResult:
        """Assemble strategy from explicit factor inputs.

        Convenience method for strategy assembly outside StateGraph context.

        Args:
            factor_names: List of factor names
            factor_metrics: Dict of factor_name -> {ic, ir} metrics
            factor_data: Optional DataFrame with factor values

        Returns:
            StrategyResult with assembled strategy
        """
        # Build mock evaluation results
        evaluation_results = [
            {
                "factor_name": name,
                "metrics": factor_metrics.get(name, {"ic": 0, "ir": 0}),
            }
            for name in factor_names
        ]

        # Select and weight
        selected = self._select_factors(factor_names, evaluation_results)
        weights = self._calculate_weights(selected, evaluation_results)
        self._validate_weights(weights)

        # Generate signals
        signals = None
        if factor_data is not None:
            signals = self._generate_signals(weights, factor_data)

        return StrategyResult(
            strategy_name=self._generate_strategy_name(selected),
            factor_weights=weights,
            n_factors=len(weights),
            combination_method=self.config.combination_method.value,
            construction_method=self.config.construction_method.value,
            signals=signals,
        )


# Node function for StateGraph
async def assemble_strategy_node(state: AgentState) -> AgentState:
    """StateGraph node function for strategy assembly.

    Args:
        state: Current agent state

    Returns:
        Updated state with strategy result
    """
    agent = StrategyAssemblyAgent()
    return await agent.assemble(state)


# Factory function
def create_strategy_agent(
    config: Optional[StrategyConfig] = None,
) -> StrategyAssemblyAgent:
    """Factory function to create a StrategyAssemblyAgent.

    Args:
        config: Strategy configuration

    Returns:
        Configured StrategyAssemblyAgent instance
    """
    return StrategyAssemblyAgent(config=config)
