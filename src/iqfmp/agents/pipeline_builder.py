"""Pipeline Builder for IQFMP Agent Orchestration.

Builds complete agent pipelines connecting all agents:
- Hypothesis Generation
- Factor Generation
- Factor Evaluation
- Strategy Assembly
- Backtest Optimization
- Risk Check

Implements the full IQFMP workflow as a StateGraph.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from iqfmp.agents.orchestrator import (
    AgentOrchestrator,
    AgentState,
    OrchestratorConfig,
    StateGraph,
    CheckpointSaver,
    MemorySaver,
)
from iqfmp.agents.hypothesis_agent import HypothesisAgent
from iqfmp.agents.factor_generation import FactorGenerationAgent, FactorGenerationConfig
from iqfmp.agents.evaluation_agent import (
    FactorEvaluationAgent,
    EvaluationAgentConfig,
    evaluate_factors_node,
)
from iqfmp.agents.strategy_agent import (
    StrategyAssemblyAgent,
    StrategyConfig,
    assemble_strategy_node,
)
from iqfmp.agents.backtest_agent import (
    BacktestOptimizationAgent,
    BacktestConfig,
    optimize_backtest_node,
)
from iqfmp.agents.risk_agent import (
    RiskCheckAgent,
    RiskConfig,
    check_risk_node,
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete agent pipeline."""

    name: str = "iqfmp_pipeline"

    # Pipeline options
    enable_hypothesis: bool = True
    enable_evaluation: bool = True
    enable_strategy: bool = True
    enable_backtest: bool = True
    enable_risk_check: bool = True

    # Orchestrator settings
    max_iterations: int = 100
    timeout: float = 600.0  # 10 minutes
    checkpoint_enabled: bool = True

    # Agent configs (optional - use defaults if not provided)
    factor_config: Optional[FactorGenerationConfig] = None
    evaluation_config: Optional[EvaluationAgentConfig] = None
    strategy_config: Optional[StrategyConfig] = None
    backtest_config: Optional[BacktestConfig] = None
    risk_config: Optional[RiskConfig] = None


class PipelineBuilder:
    """Builder for IQFMP agent pipelines.

    Creates StateGraph-based pipelines that connect all agents
    in the proper workflow order with conditional routing.

    The full pipeline flow:
    ```
    hypothesis → generate → evaluate → [check_passed?]
                                          ├── yes → strategy → backtest → risk → [approved?]
                                          │                                        ├── yes → finish
                                          │                                        └── no → iterate
                                          └── no → iterate/finish
    ```

    Usage:
        builder = PipelineBuilder(config)
        pipeline = builder.build()
        result = await pipeline.run(initial_state)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize the pipeline builder.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._graph: Optional[StateGraph] = None
        self._agents: dict[str, Any] = {}

    def build(
        self,
        checkpoint_saver: Optional[CheckpointSaver] = None,
    ) -> AgentOrchestrator:
        """Build the complete pipeline.

        Args:
            checkpoint_saver: Optional checkpoint persistence

        Returns:
            Configured AgentOrchestrator ready for execution
        """
        logger.info(f"Building pipeline: {self.config.name}")

        # Initialize agents
        self._initialize_agents()

        # Create graph
        self._graph = StateGraph(name=self.config.name)

        # Add nodes
        self._add_nodes()

        # Add edges
        self._add_edges()

        # Set entry and finish points
        self._set_entry_finish()

        # Create orchestrator
        orchestrator_config = OrchestratorConfig(
            name=self.config.name,
            max_iterations=self.config.max_iterations,
            timeout=self.config.timeout,
            checkpoint_enabled=self.config.checkpoint_enabled,
        )

        orchestrator = AgentOrchestrator(
            graph=self._graph,
            config=orchestrator_config,
            checkpoint_saver=checkpoint_saver or MemorySaver(),
        )

        logger.info(f"Pipeline built: {orchestrator.get_graph_structure()}")

        return orchestrator

    def _initialize_agents(self) -> None:
        """Initialize all agents with configs."""
        # Hypothesis agent
        if self.config.enable_hypothesis:
            self._agents["hypothesis"] = HypothesisAgent()

        # Factor generation agent
        self._agents["factor_generation"] = FactorGenerationAgent(
            config=self.config.factor_config or FactorGenerationConfig()
        )

        # Evaluation agent
        if self.config.enable_evaluation:
            self._agents["evaluation"] = FactorEvaluationAgent(
                config=self.config.evaluation_config or EvaluationAgentConfig()
            )

        # Strategy agent
        if self.config.enable_strategy:
            self._agents["strategy"] = StrategyAssemblyAgent(
                config=self.config.strategy_config or StrategyConfig()
            )

        # Backtest agent
        if self.config.enable_backtest:
            self._agents["backtest"] = BacktestOptimizationAgent(
                config=self.config.backtest_config or BacktestConfig()
            )

        # Risk agent
        if self.config.enable_risk_check:
            self._agents["risk"] = RiskCheckAgent(
                config=self.config.risk_config or RiskConfig()
            )

    def _add_nodes(self) -> None:
        """Add all nodes to the graph."""
        # Start node
        self._graph.add_node("start", self._start_node)

        # Hypothesis node
        if self.config.enable_hypothesis:
            self._graph.add_node("hypothesis", self._hypothesis_node)

        # Factor generation node
        self._graph.add_node("generate", self._generate_node)

        # Evaluation node
        if self.config.enable_evaluation:
            self._graph.add_node("evaluate", evaluate_factors_node)

        # Strategy assembly node
        if self.config.enable_strategy:
            self._graph.add_node("strategy", assemble_strategy_node)

        # Backtest optimization node
        if self.config.enable_backtest:
            self._graph.add_node("backtest", optimize_backtest_node)

        # Risk check node
        if self.config.enable_risk_check:
            self._graph.add_node("risk", check_risk_node)

        # Finish node
        self._graph.add_node("finish", self._finish_node)

    def _add_edges(self) -> None:
        """Add edges with conditional routing."""
        # Start → hypothesis or generate
        if self.config.enable_hypothesis:
            self._graph.add_edge("start", "hypothesis")
            self._graph.add_edge("hypothesis", "generate")
        else:
            self._graph.add_edge("start", "generate")

        # Generate → evaluate or strategy
        if self.config.enable_evaluation:
            self._graph.add_edge("generate", "evaluate")

            # Evaluate → conditional routing
            if self.config.enable_strategy:
                self._graph.add_conditional_edge(
                    "evaluate",
                    self._route_after_evaluation,
                    ["strategy", "finish"],
                )
            else:
                self._graph.add_edge("evaluate", "finish")
        elif self.config.enable_strategy:
            self._graph.add_edge("generate", "strategy")
        else:
            self._graph.add_edge("generate", "finish")

        # Strategy → backtest or finish
        if self.config.enable_strategy:
            if self.config.enable_backtest:
                self._graph.add_edge("strategy", "backtest")
            else:
                self._graph.add_edge("strategy", "finish")

        # Backtest → risk or finish
        if self.config.enable_backtest:
            if self.config.enable_risk_check:
                self._graph.add_edge("backtest", "risk")
            else:
                self._graph.add_edge("backtest", "finish")

        # Risk → conditional routing
        if self.config.enable_risk_check:
            self._graph.add_conditional_edge(
                "risk",
                self._route_after_risk,
                ["finish"],  # Always finish after risk for now
            )

    def _set_entry_finish(self) -> None:
        """Set entry and finish points."""
        self._graph.set_entry_point("start")
        self._graph.set_finish_point("finish")

    # Node implementations
    async def _start_node(self, state: AgentState) -> AgentState:
        """Initialize pipeline execution."""
        logger.info("Pipeline started")
        context = state.context.copy()
        context["pipeline_status"] = "started"
        context["pipeline_stage"] = "start"
        return state.update(context=context)

    async def _hypothesis_node(self, state: AgentState) -> AgentState:
        """Generate hypotheses."""
        logger.info("Generating hypotheses")
        agent = self._agents.get("hypothesis")
        if agent:
            # Use hypothesis agent
            context = state.context.copy()
            context["pipeline_stage"] = "hypothesis"
            # Generate hypotheses based on context
            hypotheses = agent.generate(
                market=context.get("market", "crypto"),
                n_hypotheses=context.get("n_hypotheses", 3),
            )
            context["hypotheses"] = [h.to_dict() for h in hypotheses]
            return state.update(context=context)
        return state

    async def _generate_node(self, state: AgentState) -> AgentState:
        """Generate factors."""
        logger.info("Generating factors")
        agent = self._agents.get("factor_generation")
        context = state.context.copy()
        context["pipeline_stage"] = "generate"

        # Get prompts from hypotheses or context
        prompts = context.get("factor_prompts", [])
        if not prompts and "hypotheses" in context:
            prompts = [h.get("description", "") for h in context["hypotheses"]]

        # Generate factors
        generated_factors = []
        for prompt in prompts[:5]:  # Limit to 5
            try:
                result = await agent.generate(
                    prompt=prompt,
                    factor_family=context.get("factor_family", "momentum"),
                )
                if result.success:
                    generated_factors.append({
                        "name": result.factor.name,
                        "family": result.factor.family,
                        "code": result.factor.code,
                        "description": result.factor.description,
                    })
            except Exception as e:
                logger.warning(f"Factor generation failed: {e}")

        context["generated_factors"] = generated_factors
        return state.update(context=context)

    async def _finish_node(self, state: AgentState) -> AgentState:
        """Finalize pipeline execution."""
        logger.info("Pipeline finished")
        context = state.context.copy()
        context["pipeline_status"] = "completed"
        context["pipeline_stage"] = "finish"

        # Generate summary
        summary = self._generate_summary(context)
        context["pipeline_summary"] = summary

        return state.update(context=context)

    # Routing functions
    def _route_after_evaluation(self, state: AgentState) -> str:
        """Route based on evaluation results."""
        context = state.context
        factors_passed = context.get("factors_passed", [])

        if len(factors_passed) > 0:
            return "strategy"
        else:
            logger.warning("No factors passed evaluation")
            return "finish"

    def _route_after_risk(self, state: AgentState) -> str:
        """Route based on risk assessment."""
        context = state.context
        approved = context.get("strategy_approved", False)

        if approved:
            return "finish"
        else:
            logger.warning("Strategy not approved by risk check")
            return "finish"

    def _generate_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate pipeline execution summary."""
        return {
            "generated_factors": len(context.get("generated_factors", [])),
            "factors_passed": len(context.get("factors_passed", [])),
            "strategy_result": context.get("strategy_result") is not None,
            "backtest_metrics": context.get("backtest_metrics"),
            "risk_level": context.get("risk_level", "unknown"),
            "approved": context.get("strategy_approved", False),
        }


def build_full_pipeline(
    config: Optional[PipelineConfig] = None,
    checkpoint_saver: Optional[CheckpointSaver] = None,
) -> AgentOrchestrator:
    """Convenience function to build a full pipeline.

    Args:
        config: Pipeline configuration
        checkpoint_saver: Optional checkpoint persistence

    Returns:
        Configured AgentOrchestrator
    """
    builder = PipelineBuilder(config)
    return builder.build(checkpoint_saver)


def build_evaluation_only_pipeline(
    checkpoint_saver: Optional[CheckpointSaver] = None,
) -> AgentOrchestrator:
    """Build a pipeline for evaluation only (no strategy/backtest/risk).

    Args:
        checkpoint_saver: Optional checkpoint persistence

    Returns:
        Configured AgentOrchestrator
    """
    config = PipelineConfig(
        name="evaluation_pipeline",
        enable_hypothesis=False,
        enable_evaluation=True,
        enable_strategy=False,
        enable_backtest=False,
        enable_risk_check=False,
    )
    builder = PipelineBuilder(config)
    return builder.build(checkpoint_saver)


def build_research_pipeline(
    checkpoint_saver: Optional[CheckpointSaver] = None,
) -> AgentOrchestrator:
    """Build a pipeline for research (hypothesis → generate → evaluate).

    Args:
        checkpoint_saver: Optional checkpoint persistence

    Returns:
        Configured AgentOrchestrator
    """
    config = PipelineConfig(
        name="research_pipeline",
        enable_hypothesis=True,
        enable_evaluation=True,
        enable_strategy=False,
        enable_backtest=False,
        enable_risk_check=False,
    )
    builder = PipelineBuilder(config)
    return builder.build(checkpoint_saver)


def build_production_pipeline(
    checkpoint_saver: Optional[CheckpointSaver] = None,
) -> AgentOrchestrator:
    """Build a full production pipeline with all checks.

    Args:
        checkpoint_saver: Optional checkpoint persistence

    Returns:
        Configured AgentOrchestrator
    """
    config = PipelineConfig(
        name="production_pipeline",
        enable_hypothesis=True,
        enable_evaluation=True,
        enable_strategy=True,
        enable_backtest=True,
        enable_risk_check=True,
        checkpoint_enabled=True,
    )
    builder = PipelineBuilder(config)
    return builder.build(checkpoint_saver)
