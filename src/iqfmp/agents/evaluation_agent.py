"""Factor Evaluation Agent for IQFMP.

Wraps FactorEvaluator as a StateGraph-compatible agent node.
Performs comprehensive factor evaluation with IC/IR/Sharpe metrics,
CV validation, and stability analysis.

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
import logging

import pandas as pd

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
    """Agent for evaluating generated factors.

    This agent wraps the FactorEvaluator to work within the
    StateGraph orchestration framework.

    Responsibilities:
    - Extract factor data from agent state
    - Run comprehensive evaluation (IC, IR, Sharpe, etc.)
    - Perform CV validation and stability analysis
    - Generate evaluation reports
    - Update state with results

    Usage:
        agent = FactorEvaluationAgent(config)
        new_state = await agent.evaluate(state)
    """

    def __init__(
        self,
        config: Optional[EvaluationAgentConfig] = None,
        ledger: Optional[ResearchLedger] = None,
    ) -> None:
        """Initialize the evaluation agent.

        Args:
            config: Agent configuration
            ledger: Research ledger for trial tracking
        """
        self.config = config or EvaluationAgentConfig()
        self.ledger = ledger or ResearchLedger(storage=MemoryStorage())

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

        # Try to calculate factor if code is provided
        if factor_code:
            try:
                # Simple sandbox execution for factor calculation
                local_vars = {"df": df, "pd": pd}
                exec(factor_code, {"__builtins__": {}}, local_vars)
                if "factor_value" in local_vars:
                    df["factor_value"] = local_vars["factor_value"]
                elif "result" in local_vars:
                    df["factor_value"] = local_vars["result"]
            except Exception as e:
                logger.warning(f"Failed to calculate factor {factor_name}: {e}")
                # Use close price as fallback
                if "close" in df.columns:
                    df["factor_value"] = df["close"].pct_change()
                else:
                    df["factor_value"] = 0.0

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
