"""RD Loop - Research-Development Loop for Factor Mining.

This module implements the main RD-Agent research loop:
1. Hypothesis Generation → 2. Factor Coding → 3. Evaluation →
4. Benchmark Comparison → 5. Feedback Analysis → 6. Factor Selection

Key components integrated:
- HypothesisAgent: Generate and manage hypotheses
- FactorEngine: Compute factor values
- FactorEvaluator: Evaluate factor metrics
- AlphaBenchmarker: Compare against Alpha158
- FactorCombiner: Combine factors using ML
- ResearchLedger: Track all experiments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import pandas as pd

from iqfmp.agents.hypothesis_agent import (
    Hypothesis,
    HypothesisAgent,
    HypothesisFamily,
    HypothesisStatus,
)
from iqfmp.core.factor_engine import FactorEngine, FactorEvaluator, get_default_data_path
from iqfmp.evaluation.alpha_benchmark import AlphaBenchmarker, BenchmarkResult
from iqfmp.evaluation.research_ledger import (
    DynamicThreshold,
    ResearchLedger,
    TrialRecord,
    MemoryStorage,
    PostgresStorage,
)
from iqfmp.models.factor_combiner import (
    FactorCombiner,
    EnsembleCombiner,
    CombinerConfig,
    ModelType,
)

logger = logging.getLogger(__name__)


class LoopPhase(str, Enum):
    """Current phase of the RD Loop."""
    INITIALIZING = "initializing"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    FACTOR_CODING = "factor_coding"
    FACTOR_EVALUATION = "factor_evaluation"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    FEEDBACK_ANALYSIS = "feedback_analysis"
    FACTOR_COMBINATION = "factor_combination"
    FACTOR_SELECTION = "factor_selection"
    COMPLETED = "completed"


@dataclass
class LoopConfig:
    """Configuration for RD Loop."""

    # Iteration limits
    max_iterations: int = 100
    max_hypotheses_per_iteration: int = 5
    target_core_factors: int = 10

    # Thresholds
    ic_threshold: float = 0.03
    ir_threshold: float = 1.0
    novelty_threshold: float = 0.7

    # Benchmark
    run_benchmark: bool = True
    benchmark_top_n: int = 10

    # Factor combination
    enable_combination: bool = True
    min_factors_for_combination: int = 3

    # Callbacks
    on_iteration_complete: Optional[Callable[[int, dict], None]] = None
    on_phase_change: Optional[Callable[[LoopPhase], None]] = None


@dataclass
class IterationResult:
    """Result of a single RD Loop iteration."""

    iteration: int
    hypotheses_tested: int
    factors_validated: int
    best_ic: float
    best_factor_name: str
    benchmark_rank: Optional[int] = None
    phase_durations: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "hypotheses_tested": self.hypotheses_tested,
            "factors_validated": self.factors_validated,
            "best_ic": self.best_ic,
            "best_factor_name": self.best_factor_name,
            "benchmark_rank": self.benchmark_rank,
            "phase_durations": self.phase_durations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LoopState:
    """Current state of the RD Loop."""

    phase: LoopPhase = LoopPhase.INITIALIZING
    iteration: int = 0
    total_hypotheses_tested: int = 0
    core_factors: list[str] = field(default_factory=list)
    iteration_results: list[IterationResult] = field(default_factory=list)
    is_running: bool = False
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "iteration": self.iteration,
            "total_hypotheses_tested": self.total_hypotheses_tested,
            "core_factors_count": len(self.core_factors),
            "core_factors": self.core_factors[:10],  # First 10 for display
            "is_running": self.is_running,
            "stop_requested": self.stop_requested,
        }


class RDLoop:
    """Main Research-Development Loop for automated factor mining.

    Example usage:
        ```python
        # Initialize
        loop = RDLoop(config=LoopConfig(max_iterations=50))

        # Load data
        loop.load_data("path/to/ohlcv.csv")

        # Run loop
        results = loop.run()

        # Get core factors
        core_factors = loop.get_core_factors()
        ```
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        hypothesis_agent: Optional[HypothesisAgent] = None,
        benchmarker: Optional[AlphaBenchmarker] = None,
        ledger: Optional[ResearchLedger] = None,
    ) -> None:
        """Initialize RD Loop.

        Args:
            config: Loop configuration
            hypothesis_agent: Agent for hypothesis generation
            benchmarker: Alpha158 benchmarker
            ledger: Research ledger for experiment tracking
        """
        self.config = config or LoopConfig()
        self.state = LoopState()

        # Components
        self.hypothesis_agent = hypothesis_agent or HypothesisAgent()
        self.benchmarker = benchmarker or AlphaBenchmarker(
            novelty_threshold=self.config.novelty_threshold
        )
        # Use PostgresStorage by default for persistence (falls back to memory if DB unavailable)
        self.ledger = ledger or self._create_ledger_with_storage()

        # Data and engines (set by load_data)
        self._df: Optional[pd.DataFrame] = None
        self._factor_engine: Optional[FactorEngine] = None
        self._factor_evaluator: Optional[FactorEvaluator] = None
        self._forward_returns: Optional[pd.Series] = None

        # Validated factors storage
        self._validated_factors: dict[str, pd.Series] = {}
        self._factor_metadata: dict[str, dict] = {}

    def _create_ledger_with_storage(self) -> ResearchLedger:
        """Create ResearchLedger with appropriate storage backend.

        Tries PostgresStorage first, falls back to MemoryStorage if DB is unavailable.

        Returns:
            ResearchLedger with storage backend
        """
        import os

        # Check if DATABASE_URL is configured
        if os.environ.get("DATABASE_URL"):
            try:
                storage = PostgresStorage()
                logger.info("RDLoop using PostgresStorage for persistence")
                return ResearchLedger(storage=storage)
            except Exception as e:
                logger.warning(f"PostgresStorage initialization failed: {e}, falling back to MemoryStorage")

        logger.info("RDLoop using MemoryStorage (no DB configured)")
        return ResearchLedger(storage=MemoryStorage())

    def load_data(
        self,
        data_source: str | Path | pd.DataFrame,
    ) -> None:
        """Load OHLCV data for factor mining.

        Args:
            data_source: Path to CSV/DB tag or DataFrame
        """
        if isinstance(data_source, pd.DataFrame):
            self._df = data_source.copy()
        else:
            # Support "db" or "auto" to load from TimescaleDB via DataProvider
            if isinstance(data_source, str) and data_source.lower() in {"db", "auto"}:
                try:
                    from iqfmp.core.data_provider import load_ohlcv_sync
                    self._df = load_ohlcv_sync(symbol="ETHUSDT", timeframe="1d", use_db=True)
                except Exception as e:
                    logger.warning(f"RD Loop DB load failed, fallback to CSV: {e}")
                    self._df = pd.read_csv(get_default_data_path())
            else:
                self._df = pd.read_csv(data_source)

        # Prepare data
        if "timestamp" in self._df.columns:
            self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])
            self._df = self._df.sort_values("timestamp").reset_index(drop=True)

        # Calculate forward returns
        self._df["fwd_returns_1d"] = self._df["close"].pct_change().shift(-1)

        # Initialize engines
        self._factor_engine = FactorEngine(df=self._df)
        self._factor_evaluator = FactorEvaluator(self._factor_engine)
        self._forward_returns = self._df["fwd_returns_1d"]

        logger.info(f"Loaded {len(self._df)} rows of data")

    def run(
        self,
        focus_families: Optional[list[HypothesisFamily]] = None,
    ) -> list[IterationResult]:
        """Run the RD Loop.

        Args:
            focus_families: Optional list of families to focus on

        Returns:
            List of iteration results
        """
        if self._df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.state.is_running = True
        self.state.stop_requested = False
        self.state.iteration = 0

        logger.info("Starting RD Loop")
        self._set_phase(LoopPhase.INITIALIZING)

        # Initialize benchmark
        if self.config.run_benchmark:
            logger.info("Computing benchmark factors...")
            self.benchmarker.compute_benchmark_factors(self._df)
            self.benchmarker.evaluate_benchmark_factors(self._df, self._forward_returns)

        try:
            while self._should_continue():
                self.state.iteration += 1
                logger.info(f"=== Iteration {self.state.iteration} ===")

                result = self._run_iteration(focus_families)
                self.state.iteration_results.append(result)

                if self.config.on_iteration_complete:
                    self.config.on_iteration_complete(self.state.iteration, result.to_dict())

        except Exception as e:
            logger.error(f"RD Loop failed: {e}")
            raise
        finally:
            self.state.is_running = False

        self._set_phase(LoopPhase.COMPLETED)
        logger.info(f"RD Loop completed. {len(self.state.core_factors)} core factors found.")

        return self.state.iteration_results

    def stop(self) -> None:
        """Request stop of running loop."""
        self.state.stop_requested = True
        logger.info("Stop requested")

    def get_core_factors(self) -> list[dict[str, Any]]:
        """Get all validated core factors.

        Returns:
            List of factor metadata dictionaries
        """
        return [
            self._factor_metadata[name]
            for name in self.state.core_factors
            if name in self._factor_metadata
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get loop statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "state": self.state.to_dict(),
            "hypothesis_stats": self.hypothesis_agent.get_statistics(),
            "ledger_stats": self.ledger.get_statistics(),
            "benchmark_top_factors": self.benchmarker.get_top_factors(5),
        }

    def combine_factors(
        self,
        factor_names: Optional[list[str]] = None,
        model_type: ModelType = ModelType.LIGHTGBM,
    ) -> pd.Series:
        """Combine validated factors using ML model.

        Args:
            factor_names: Factors to combine (defaults to all core factors)
            model_type: Model type to use

        Returns:
            Combined factor series
        """
        names = factor_names or self.state.core_factors

        if len(names) < self.config.min_factors_for_combination:
            raise ValueError(
                f"Need at least {self.config.min_factors_for_combination} factors, "
                f"got {len(names)}"
            )

        # Build factors DataFrame
        factors_df = pd.DataFrame({
            name: self._validated_factors[name]
            for name in names
            if name in self._validated_factors
        })

        # Create and fit combiner
        combiner = FactorCombiner(CombinerConfig(model_type=model_type))
        result = combiner.fit(factors_df, self._forward_returns)

        logger.info(f"Combined {len(names)} factors. Test IC: {result.test_metrics.get('test_ic', 0):.4f}")

        return result.combined_factor

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _should_continue(self) -> bool:
        """Check if loop should continue."""
        if self.state.stop_requested:
            return False

        if self.state.iteration >= self.config.max_iterations:
            logger.info(f"Reached max iterations ({self.config.max_iterations})")
            return False

        if len(self.state.core_factors) >= self.config.target_core_factors:
            logger.info(f"Reached target core factors ({self.config.target_core_factors})")
            return False

        return True

    def _set_phase(self, phase: LoopPhase) -> None:
        """Set current phase."""
        self.state.phase = phase
        if self.config.on_phase_change:
            self.config.on_phase_change(phase)

    def _run_iteration(
        self,
        focus_families: Optional[list[HypothesisFamily]] = None,
    ) -> IterationResult:
        """Run a single iteration of the RD Loop."""
        import time
        phase_times = {}

        # Phase 1: Generate hypotheses
        start = time.time()
        self._set_phase(LoopPhase.HYPOTHESIS_GENERATION)
        hypotheses = self.hypothesis_agent.generate_hypotheses(
            self._df,
            n_hypotheses=self.config.max_hypotheses_per_iteration,
            focus_family=focus_families[0] if focus_families else None,
        )
        phase_times["hypothesis_generation"] = time.time() - start

        # Phase 2: Convert to factor code
        start = time.time()
        self._set_phase(LoopPhase.FACTOR_CODING)
        factor_codes = self.hypothesis_agent.convert_to_factors(hypotheses)
        phase_times["factor_coding"] = time.time() - start

        # Phase 3: Evaluate factors
        start = time.time()
        self._set_phase(LoopPhase.FACTOR_EVALUATION)
        best_ic = 0.0
        best_factor_name = ""
        validated_count = 0

        for hypothesis, code in factor_codes:
            try:
                result = self._evaluate_factor(hypothesis, code)

                if result["passed"]:
                    validated_count += 1
                    if abs(result["ic"]) > abs(best_ic):
                        best_ic = result["ic"]
                        best_factor_name = hypothesis.factor_name or hypothesis.name

            except Exception as e:
                logger.warning(f"Factor evaluation failed for {hypothesis.name}: {e}")
                continue

        phase_times["factor_evaluation"] = time.time() - start
        self.state.total_hypotheses_tested += len(hypotheses)

        # Phase 4: Benchmark comparison (for best factor)
        benchmark_rank = None
        if self.config.run_benchmark and best_factor_name and best_factor_name in self._validated_factors:
            start = time.time()
            self._set_phase(LoopPhase.BENCHMARK_COMPARISON)
            try:
                benchmark_result = self.benchmarker.benchmark(
                    best_factor_name,
                    self._validated_factors[best_factor_name],
                    self._df,
                    self._forward_returns,
                )
                benchmark_rank = benchmark_result.rank_in_benchmark
            except Exception as e:
                logger.warning(f"Benchmark comparison failed: {e}")
            phase_times["benchmark_comparison"] = time.time() - start

        # Phase 5: Optional factor combination
        if (
            self.config.enable_combination
            and len(self.state.core_factors) >= self.config.min_factors_for_combination
        ):
            start = time.time()
            self._set_phase(LoopPhase.FACTOR_COMBINATION)
            try:
                combined = self.combine_factors()
                # Evaluate combined factor
                combined_result = self._factor_evaluator.evaluate(
                    combined,
                    forward_periods=[1, 5, 10],
                )
                logger.info(f"Combined factor IC: {combined_result['metrics']['ic_mean']:.4f}")
            except Exception as e:
                logger.warning(f"Factor combination failed: {e}")
            phase_times["factor_combination"] = time.time() - start

        return IterationResult(
            iteration=self.state.iteration,
            hypotheses_tested=len(hypotheses),
            factors_validated=validated_count,
            best_ic=best_ic,
            best_factor_name=best_factor_name,
            benchmark_rank=benchmark_rank,
            phase_durations=phase_times,
        )

    def _evaluate_factor(
        self,
        hypothesis: Hypothesis,
        code: str,
    ) -> dict[str, Any]:
        """Evaluate a single factor.

        Args:
            hypothesis: The hypothesis being tested
            code: Factor code

        Returns:
            Evaluation result dictionary
        """
        # Compute factor values
        factor_values = self._factor_engine.compute_factor(
            code, hypothesis.factor_name or "factor"
        )

        # Evaluate
        result = self._factor_evaluator.evaluate(
            factor_values,
            forward_periods=[1, 5, 10],
        )

        # Update hypothesis with results
        self.hypothesis_agent.process_results(hypothesis, result)

        # Record to ledger
        trial = TrialRecord(
            factor_name=hypothesis.factor_name or hypothesis.name,
            factor_family=hypothesis.family.value,
            sharpe_ratio=result["metrics"]["sharpe"],
            ic_mean=result["metrics"]["ic_mean"],
            ir=result["metrics"]["ir"],
        )
        self.ledger.record(trial)

        # Check threshold
        ic = abs(result["metrics"]["ic_mean"])
        ir = result["metrics"]["ir"]
        passed = ic >= self.config.ic_threshold and ir >= self.config.ir_threshold

        if passed:
            factor_name = hypothesis.factor_name or hypothesis.name
            self._validated_factors[factor_name] = factor_values
            self._factor_metadata[factor_name] = {
                "name": factor_name,
                "family": hypothesis.family.value,
                "code": code,
                "metrics": result["metrics"],
                "hypothesis": hypothesis.to_dict(),
            }
            if factor_name not in self.state.core_factors:
                self.state.core_factors.append(factor_name)
                logger.info(f"New core factor: {factor_name} (IC={ic:.4f}, IR={ir:.2f})")

        return {
            "passed": passed,
            "ic": result["metrics"]["ic_mean"],
            "ir": ir,
            "sharpe": result["metrics"]["sharpe"],
        }


# =============================================================================
# Factory function for easy creation
# =============================================================================

def create_rd_loop(
    data_path: Optional[str | Path] = None,
    max_iterations: int = 100,
    target_factors: int = 10,
    ic_threshold: float = 0.03,
    **kwargs,
) -> RDLoop:
    """Create and optionally initialize an RD Loop.

    Args:
        data_path: Optional path to OHLCV data
        max_iterations: Maximum iterations
        target_factors: Target number of core factors
        ic_threshold: IC threshold for factor validation
        **kwargs: Additional config options

    Returns:
        Configured RDLoop instance
    """
    config = LoopConfig(
        max_iterations=max_iterations,
        target_core_factors=target_factors,
        ic_threshold=ic_threshold,
        **kwargs,
    )

    loop = RDLoop(config=config)

    if data_path:
        loop.load_data(data_path)

    return loop
