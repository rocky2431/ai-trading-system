"""Factor Evaluation Agent for IQFMP.

Wraps FactorEvaluator as a StateGraph-compatible agent node with LLM support.
Performs comprehensive factor evaluation with IC/IR/Sharpe metrics,
CV validation, and stability analysis.

LLM Integration:
- Uses frontend-configured model from ConfigService via model_config.py
- LLM generates intelligent evaluation insights and recommendations
- Falls back to template-based reports if LLM is unavailable

Six-dimensional coverage:
1. Functional: Factor evaluation, metrics calculation, report generation
2. Boundary: Empty data, minimal periods, edge cases
3. Exception: Invalid factors, missing columns, NaN handling
4. Performance: Batch evaluation optimization
5. Security: Input validation
6. Compatibility: Multiple factor families, data formats
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
import json
import logging
import re

import pandas as pd

# P0 SECURITY: Import ASTSecurityChecker for mandatory code validation
from iqfmp.core.security import ASTSecurityChecker

# Module-level security checker instance (reused for performance)
_security_checker = ASTSecurityChecker()

logger = logging.getLogger(__name__)


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


EVALUATION_SYSTEM_PROMPT = """You are an expert quantitative analyst specializing in factor evaluation and performance analysis.

Your task is to analyze factor performance metrics and provide actionable insights:
1. Explain why a factor succeeded or failed based on the metrics
2. Identify patterns and anomalies in the evaluation results
3. Suggest improvements to the factor formula or parameters
4. Provide recommendations for factor combination or refinement

Focus on:
- Information Coefficient (IC) - predictive power, should be > 0.03 for significance
- Information Ratio (IR) - IC consistency, should be > 1.0 for reliability
- Sharpe Ratio - risk-adjusted returns
- Max Drawdown - downside risk
- Stability across different market regimes

Provide specific, actionable recommendations."""

from iqfmp.agents.orchestrator import AgentState
from iqfmp.evaluation.factor_evaluator import (
    FactorEvaluator,
    EvaluationConfig,
    EvaluationResult,
    FactorReport,
    EvaluationPipeline,
    InvalidFactorError,
    EvaluationFailedError,
)
from iqfmp.evaluation.research_ledger import ResearchLedger, MemoryStorage


logger = logging.getLogger(__name__)


class EvaluationAgentError(Exception):
    """Base error for evaluation agent failures."""

    pass


class DataValidationError(EvaluationAgentError):
    """Raised when input data validation fails."""

    pass


class EvaluationTimeoutError(EvaluationAgentError):
    """Raised when evaluation times out."""

    pass


@dataclass
class EvaluationAgentConfig:
    """Configuration for Factor Evaluation Agent."""

    # Evaluation settings
    ic_threshold: float = 0.03
    ir_threshold: float = 1.0
    use_cv_splits: bool = True
    run_stability_analysis: bool = True

    # Processing settings
    min_data_points: int = 100
    max_factors_per_batch: int = 50
    timeout_per_factor: float = 30.0  # seconds

    # Output settings
    generate_reports: bool = True
    include_recommendations: bool = True

    # Security / execution mode
    allow_python_factors: bool = False  # Default: Qlib expression-only

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ic_threshold": self.ic_threshold,
            "ir_threshold": self.ir_threshold,
            "use_cv_splits": self.use_cv_splits,
            "run_stability_analysis": self.run_stability_analysis,
            "min_data_points": self.min_data_points,
            "max_factors_per_batch": self.max_factors_per_batch,
            "timeout_per_factor": self.timeout_per_factor,
            "generate_reports": self.generate_reports,
            "include_recommendations": self.include_recommendations,
            "allow_python_factors": self.allow_python_factors,
        }


@dataclass
class EvaluationAgentResult:
    """Result from factor evaluation agent."""

    factor_name: str
    success: bool
    result: Optional[EvaluationResult] = None
    report: Optional[FactorReport] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        data = {
            "factor_name": self.factor_name,
            "success": self.success,
        }
        if self.result:
            data["metrics"] = self.result.metrics.to_dict()
            data["passes_threshold"] = self.result.passes_threshold
            data["threshold_used"] = self.result.threshold_used
            data["trial_id"] = self.result.trial_id
        if self.report:
            data["grade"] = self.report.grade
            data["recommendations"] = self.report.recommendations
        if self.error:
            data["error"] = self.error
        return data


class FactorEvaluationAgent:
    """Agent for evaluating generated factors with LLM-powered insights.

    This agent wraps the FactorEvaluator to work within the
    StateGraph orchestration framework with LLM-powered analysis.

    Responsibilities:
    - Extract factor data from agent state
    - Run comprehensive evaluation (IC, IR, Sharpe, etc.)
    - Perform CV validation and stability analysis
    - Generate LLM-powered evaluation reports and insights
    - Update state with results

    LLM Integration:
    - Uses frontend-configured model via get_agent_full_config("factor_evaluation")
    - Generates intelligent insights about factor performance
    - Provides actionable recommendations for improvement

    Usage:
        agent = FactorEvaluationAgent(config, llm_provider=llm)
        new_state = await agent.evaluate(state)
    """

    def __init__(
        self,
        config: Optional[EvaluationAgentConfig] = None,
        ledger: Optional[ResearchLedger] = None,
        llm_provider: Optional[LLMProviderProtocol] = None,
    ) -> None:
        """Initialize the evaluation agent with LLM support.

        Args:
            config: Agent configuration
            ledger: Research ledger for trial tracking
            llm_provider: LLM provider for AI-powered analysis
        """
        self.config = config or EvaluationAgentConfig()
        self.ledger = ledger or ResearchLedger(storage=MemoryStorage())
        self.llm_provider = llm_provider
        # Qlib expression engine for factor_value computation (expression-only by default).
        from iqfmp.core.qlib_crypto import QlibExpressionEngine
        self._expression_engine = QlibExpressionEngine(require_qlib=True)

        # Create evaluator with config
        eval_config = EvaluationConfig(
            ic_threshold=self.config.ic_threshold,
            ir_threshold=self.config.ir_threshold,
            use_cv_splits=self.config.use_cv_splits,
            run_stability_analysis=self.config.run_stability_analysis,
            min_periods=20,
        )
        self.evaluator = FactorEvaluator(ledger=self.ledger, config=eval_config)
        self.pipeline = EvaluationPipeline(ledger=self.ledger, config=eval_config)

    async def generate_llm_insights(
        self,
        factor_name: str,
        factor_family: str,
        metrics: dict[str, Any],
        passes_threshold: bool,
    ) -> dict[str, Any]:
        """Generate LLM-powered insights for factor evaluation.

        Args:
            factor_name: Name of the factor
            factor_family: Factor family/category
            metrics: Evaluation metrics
            passes_threshold: Whether factor passed evaluation threshold

        Returns:
            Dict with insights and recommendations
        """
        if self.llm_provider is None:
            return self._generate_template_insights(factor_name, metrics, passes_threshold)

        # Get model configuration from ConfigService
        from iqfmp.agents.model_config import get_agent_full_config
        model_id, temperature, custom_system_prompt = get_agent_full_config("factor_evaluation")

        # Build analysis prompt
        status = "PASSED" if passes_threshold else "FAILED"
        prompt = f"""Analyze the following factor evaluation results:

Factor: {factor_name}
Family: {factor_family}
Status: {status}

Metrics:
- IC (Information Coefficient): {metrics.get('ic', 'N/A')}
- IR (Information Ratio): {metrics.get('ir', 'N/A')}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {metrics.get('max_drawdown', 'N/A')}
- Turnover: {metrics.get('turnover', 'N/A')}

Stability Analysis:
{json.dumps(metrics.get('stability', {}), indent=2)}

Provide:
1. A brief analysis of why this factor {"succeeded" if passes_threshold else "failed"} (2-3 sentences)
2. Key insights about the factor's predictive power and consistency
3. 2-3 specific recommendations for improvement

Format as JSON:
```json
{{
    "analysis": "Your analysis...",
    "insights": ["Insight 1", "Insight 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"]
}}
```"""

        try:
            system_prompt = custom_system_prompt or EVALUATION_SYSTEM_PROMPT

            # Prefer schema-validated structured output when using the native LLMProvider.
            from iqfmp.llm.provider import LLMProvider
            from iqfmp.llm.validation.json_schema import OutputType

            if isinstance(self.llm_provider, LLMProvider):
                _resp, validation = await self.llm_provider.complete_structured(
                    prompt=prompt,
                    output_type=OutputType.EVALUATION_INSIGHTS,
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

            # Parse LLM response (legacy fallback)
            return self._parse_llm_insights(response.content)

        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}. Using template.")
            return self._generate_template_insights(factor_name, metrics, passes_threshold)

    def _parse_llm_insights(self, response_text: str) -> dict[str, Any]:
        """Parse LLM insights response."""
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "analysis": response_text[:300],
            "insights": [],
            "recommendations": [],
        }

    def _generate_template_insights(
        self,
        factor_name: str,
        metrics: dict[str, Any],
        passes_threshold: bool,
    ) -> dict[str, Any]:
        """Generate template-based insights when LLM is unavailable."""
        ic = abs(metrics.get("ic", 0))
        ir = metrics.get("ir", 0)

        if passes_threshold:
            analysis = f"Factor {factor_name} passed evaluation with IC={ic:.4f} and IR={ir:.2f}, showing significant predictive power."
            insights = [
                f"IC of {ic:.4f} exceeds threshold, indicating meaningful signal",
                f"IR of {ir:.2f} suggests consistent performance over time",
            ]
            recommendations = [
                "Consider combining with complementary factors",
                "Monitor performance in different market regimes",
            ]
        else:
            analysis = f"Factor {factor_name} failed evaluation with IC={ic:.4f} and IR={ir:.2f}, below required thresholds."
            insights = [
                f"IC of {ic:.4f} is below minimum threshold",
                "Factor may be capturing noise rather than signal",
            ]
            recommendations = [
                "Try different lookback periods",
                "Consider adding filters for market conditions",
                "Test alternative normalization methods",
            ]

        return {
            "analysis": analysis,
            "insights": insights,
            "recommendations": recommendations,
        }

    async def evaluate(self, state: AgentState) -> AgentState:
        """Evaluate factors from state.

        This is the main entry point for StateGraph integration.

        Args:
            state: Current agent state containing:
                - context["generated_factors"]: List of generated factors
                - context["evaluation_data"]: DataFrame for evaluation

        Returns:
            Updated state with evaluation results in context["evaluation_results"]

        Raises:
            EvaluationAgentError: On evaluation failure
        """
        logger.info("FactorEvaluationAgent: Starting evaluation")

        # Extract inputs from state
        context = state.context
        generated_factors = context.get("generated_factors", [])
        evaluation_data = context.get("evaluation_data")

        if not generated_factors:
            logger.warning("No factors to evaluate")
            return state.update(
                context={
                    **context,
                    "evaluation_results": [],
                    "evaluation_error": "No factors to evaluate",
                }
            )

        if evaluation_data is None:
            raise DataValidationError("No evaluation data provided in state")

        # Validate data
        self._validate_data(evaluation_data)

        # Evaluate each factor
        results: list[EvaluationAgentResult] = []

        for factor_info in generated_factors:
            factor_name = factor_info.get("name", "unknown")
            factor_family = factor_info.get("family", "unknown")
            factor_code = factor_info.get("code", "")

            try:
                # Prepare factor-specific data
                factor_data = self._prepare_factor_data(
                    evaluation_data, factor_code, factor_name
                )

                # Run evaluation
                eval_result = self.evaluator.evaluate(
                    factor_name=factor_name,
                    factor_family=factor_family,
                    data=factor_data,
                )

                # Generate report if configured
                report = None
                if self.config.generate_reports:
                    report = eval_result.generate_report()

                results.append(
                    EvaluationAgentResult(
                        factor_name=factor_name,
                        success=True,
                        result=eval_result,
                        report=report,
                    )
                )

                logger.info(
                    f"Evaluated {factor_name}: IC={eval_result.metrics.ic:.4f}, "
                    f"IR={eval_result.metrics.ir:.2f}, "
                    f"Passes={eval_result.passes_threshold}"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {factor_name}: {e}")
                results.append(
                    EvaluationAgentResult(
                        factor_name=factor_name,
                        success=False,
                        error=str(e),
                    )
                )

        # Calculate summary statistics
        summary = self._calculate_summary(results)

        # Update state
        new_context = {
            **context,
            "evaluation_results": [r.to_dict() for r in results],
            "evaluation_summary": summary,
            "factors_passed": [
                r.factor_name for r in results
                if r.success and r.result and r.result.passes_threshold
            ],
            "factors_failed": [
                r.factor_name for r in results
                if not r.success or (r.result and not r.result.passes_threshold)
            ],
        }

        logger.info(
            f"FactorEvaluationAgent: Completed. "
            f"Passed: {len(new_context['factors_passed'])}, "
            f"Failed: {len(new_context['factors_failed'])}"
        )

        return state.update(context=new_context)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate evaluation data.

        Args:
            data: DataFrame to validate

        Raises:
            DataValidationError: If validation fails
        """
        if data is None or data.empty:
            raise DataValidationError("Evaluation data is empty")

        if len(data) < self.config.min_data_points:
            raise DataValidationError(
                f"Insufficient data points: {len(data)} < {self.config.min_data_points}"
            )

        # Check for required columns
        required_cols = ["forward_return"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

    def _prepare_factor_data(
        self,
        data: pd.DataFrame,
        factor_code: str,
        factor_name: str,
    ) -> pd.DataFrame:
        """Prepare data for factor evaluation.

        If factor values are not in data, attempt to calculate them.

        Args:
            data: Base data DataFrame
            factor_code: Factor calculation code
            factor_name: Name of the factor

        Returns:
            DataFrame with factor_value column
        """
        df = data.copy()

        # Check if factor values already exist
        if "factor_value" in df.columns:
            return df

        if not factor_code:
            raise DataValidationError(f"No factor code provided for {factor_name}")

        # Determine if this looks like Python code (legacy) or Qlib expression (preferred).
        is_python_code = (
            factor_code.strip().startswith("def ")
            or factor_code.strip().startswith("import ")
            or factor_code.strip().startswith("from ")
            or "\ndef " in factor_code
        )

        if is_python_code:
            if not self.config.allow_python_factors:
                raise DataValidationError(
                    f"Python factor code is not allowed by default (factor={factor_name}). "
                    "Provide a Qlib expression or set allow_python_factors=True explicitly."
                )

            # =================================================================
            # P0 SECURITY: Mandatory AST security check before any exec
            # This prevents code injection through malicious factor code
            # =================================================================
            is_safe, violations = _security_checker.check(factor_code)
            if not is_safe:
                violation_details = "; ".join(violations[:5])
                raise DataValidationError(
                    f"SECURITY VIOLATION: Factor code failed security check "
                    f"(factor={factor_name}). Violations: {violation_details}"
                )

            # Best-effort sandboxed evaluation: no imports, no builtins beyond a small whitelist.
            local_vars: dict[str, Any] = {"df": df, "pd": pd}
            safe_builtins: dict[str, Any] = {
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
            }

            try:
                exec(factor_code, {"__builtins__": safe_builtins}, local_vars)
            except Exception as e:
                raise DataValidationError(
                    f"Python factor execution failed in sandbox (factor={factor_name}): {e}"
                ) from e

            series_obj = local_vars.get("factor_value")
            if series_obj is None:
                series_obj = local_vars.get("result")

            if series_obj is None:
                # Try to call a function (prefer matching name; otherwise first callable).
                candidate = local_vars.get(factor_name)
                if callable(candidate):
                    series_obj = candidate(df)
                else:
                    callables = [
                        v
                        for k, v in local_vars.items()
                        if callable(v) and k not in {"pd", "df"}
                    ]
                    if callables:
                        series_obj = callables[0](df)

            if series_obj is None:
                raise DataValidationError(
                    f"Python factor code did not produce 'factor_value'/'result' or a callable function (factor={factor_name})"
                )

            if isinstance(series_obj, pd.Series):
                series = series_obj
            else:
                series = pd.Series(series_obj, index=df.index)

            if len(series) != len(df):
                raise DataValidationError(
                    f"Python factor output length mismatch (factor={factor_name}): "
                    f"{len(series)} != {len(df)}"
                )

            if series.isna().all():
                raise DataValidationError(
                    f"Computed factor_value is all-NaN (factor={factor_name})"
                )

            df["factor_value"] = series
            return df

        # Qlib expression path
        series = self._expression_engine.compute_expression(
            expression=factor_code,
            df=df,
            result_name=factor_name,
        )

        if series.isna().all():
            raise DataValidationError(
                f"Computed factor_value is all-NaN (factor={factor_name}). "
                "Likely missing required fields for the expression."
            )

        df["factor_value"] = series

        return df

    def _calculate_summary(
        self, results: list[EvaluationAgentResult]
    ) -> dict[str, Any]:
        """Calculate summary statistics for evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Summary dictionary
        """
        total = len(results)
        successful = [r for r in results if r.success and r.result]
        passed = [r for r in successful if r.result.passes_threshold]

        ic_values = [r.result.metrics.ic for r in successful]
        ir_values = [r.result.metrics.ir for r in successful]
        sharpe_values = [r.result.metrics.sharpe_ratio for r in successful]

        summary = {
            "total_evaluated": total,
            "successful_count": len(successful),
            "passed_count": len(passed),
            "failed_count": total - len(successful),
            "pass_rate": len(passed) / total if total > 0 else 0,
            "success_rate": len(successful) / total if total > 0 else 0,
        }

        if ic_values:
            import numpy as np
            summary["avg_ic"] = float(np.mean(ic_values))
            summary["max_ic"] = float(np.max(np.abs(ic_values)))
            summary["avg_ir"] = float(np.mean(ir_values))
            summary["avg_sharpe"] = float(np.mean(sharpe_values))

        if passed:
            best = max(passed, key=lambda r: abs(r.result.metrics.ic))
            summary["best_factor"] = best.factor_name
            summary["best_ic"] = best.result.metrics.ic

        return summary

    def evaluate_single(
        self,
        factor_name: str,
        factor_family: str,
        data: pd.DataFrame,
    ) -> EvaluationAgentResult:
        """Evaluate a single factor directly.

        Convenience method for single-factor evaluation outside
        the StateGraph context.

        Args:
            factor_name: Name of the factor
            factor_family: Factor family/category
            data: DataFrame with factor values and returns

        Returns:
            EvaluationAgentResult with evaluation results
        """
        try:
            self._validate_data(data)

            result = self.evaluator.evaluate(
                factor_name=factor_name,
                factor_family=factor_family,
                data=data,
            )

            report = None
            if self.config.generate_reports:
                report = result.generate_report()

            return EvaluationAgentResult(
                factor_name=factor_name,
                success=True,
                result=result,
                report=report,
            )

        except Exception as e:
            return EvaluationAgentResult(
                factor_name=factor_name,
                success=False,
                error=str(e),
            )

    def evaluate_batch(
        self,
        factors: list[dict[str, str]],
        data: pd.DataFrame,
    ) -> list[EvaluationAgentResult]:
        """Evaluate multiple factors in batch.

        Args:
            factors: List of factor dicts with 'name' and 'family' keys
            data: DataFrame for evaluation

        Returns:
            List of evaluation results
        """
        # Limit batch size
        if len(factors) > self.config.max_factors_per_batch:
            logger.warning(
                f"Batch size {len(factors)} exceeds max {self.config.max_factors_per_batch}"
            )
            factors = factors[: self.config.max_factors_per_batch]

        results = []
        for factor in factors:
            result = self.evaluate_single(
                factor_name=factor["name"],
                factor_family=factor.get("family", "unknown"),
                data=data,
            )
            results.append(result)

        return results

    def get_ledger_statistics(self) -> dict[str, Any]:
        """Get statistics from the research ledger.

        Returns:
            Dictionary with ledger statistics
        """
        return self.ledger.get_statistics()


# Node function for StateGraph
async def evaluate_factors_node(state: AgentState) -> AgentState:
    """StateGraph node function for factor evaluation.

    This function creates and runs a FactorEvaluationAgent
    to evaluate factors in the current state.

    Args:
        state: Current agent state

    Returns:
        Updated state with evaluation results
    """
    agent = FactorEvaluationAgent()
    return await agent.evaluate(state)


# Factory function
def create_evaluation_agent(
    config: Optional[EvaluationAgentConfig] = None,
    ledger: Optional[ResearchLedger] = None,
) -> FactorEvaluationAgent:
    """Factory function to create a FactorEvaluationAgent.

    Args:
        config: Agent configuration
        ledger: Research ledger for trial tracking

    Returns:
        Configured FactorEvaluationAgent instance
    """
    return FactorEvaluationAgent(config=config, ledger=ledger)
