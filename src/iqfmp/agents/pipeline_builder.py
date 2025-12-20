"""Pipeline Builder for IQFMP Agent Orchestration.

Builds complete agent pipelines connecting all agents:
- Hypothesis Generation
- Factor Generation
- Factor Evaluation
- Strategy Assembly
- Backtest Optimization
- Risk Check

Implements the full IQFMP workflow as a StateGraph.

IMPORTANT: Production pipelines REQUIRE Qdrant for factor deduplication.
Factor mining without deduplication leads to redundant factors and wasted compute.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

# Vector database for factor deduplication
try:
    from iqfmp.vector.search import SimilaritySearcher
    from iqfmp.vector.store import FactorVectorStore
    VECTOR_AVAILABLE = True
except ImportError:
    SimilaritySearcher = None
    FactorVectorStore = None
    VECTOR_AVAILABLE = False

from iqfmp.agents.orchestrator import (
    AgentOrchestrator,
    AgentState,
    OrchestratorConfig,
    StateGraph,
    CheckpointSaver,
    MemorySaver,
    PostgresCheckpointSaver,
)
from iqfmp.agents.hypothesis_agent import HypothesisAgent
from iqfmp.agents.factor_generation import (
    FactorFamily,
    FactorGenerationAgent,
    FactorGenerationConfig,
)
from iqfmp.agents.evaluation_agent import (
    FactorEvaluationAgent,
    EvaluationAgentConfig,
)
from iqfmp.agents.strategy_agent import (
    StrategyAssemblyAgent,
    StrategyConfig,
)
from iqfmp.agents.backtest_agent import (
    BacktestOptimizationAgent,
    BacktestConfig,
)
from iqfmp.agents.risk_agent import (
    RiskCheckAgent,
    RiskConfig,
)


logger = logging.getLogger(__name__)


class VectorDBRequiredError(Exception):
    """Raised when Qdrant/vector database is required but not available."""
    pass


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

    # Vector database settings (Qdrant)
    require_vector_db: bool = False  # If True, pipeline fails if Qdrant unavailable
    dedup_threshold: float = 0.85  # Similarity threshold for deduplication

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
            checkpoint_saver=checkpoint_saver or self._default_checkpoint_saver(),
        )

        logger.info(f"Pipeline built: {orchestrator.get_graph_structure()}")

        return orchestrator

    def _default_checkpoint_saver(self) -> CheckpointSaver:
        """Pick a checkpoint saver based on environment (Postgres → Memory fallback)."""
        if not self.config.checkpoint_enabled:
            return MemorySaver()

        conn_str = os.getenv("CHECKPOINT_DB_URL") or os.getenv("DATABASE_URL")
        if conn_str:
            try:
                logger.info("Using Postgres checkpoint saver for pipeline")
                return PostgresCheckpointSaver(conn_str)
            except Exception as e:
                logger.warning(f"Postgres checkpoint saver unavailable ({e}), falling back to memory")

        return MemorySaver()

    def _initialize_agents(self) -> None:
        """Initialize all agents with configs."""
        # Hypothesis agent
        if self.config.enable_hypothesis:
            self._agents["hypothesis"] = HypothesisAgent()

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
            self._graph.add_node("evaluate", self._evaluate_node)

        # Strategy assembly node
        if self.config.enable_strategy:
            self._graph.add_node("strategy", self._strategy_node)

        # Backtest optimization node
        if self.config.enable_backtest:
            self._graph.add_node("backtest", self._backtest_node)

        # Risk check node
        if self.config.enable_risk_check:
            self._graph.add_node("risk", self._risk_node)

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
        context.setdefault("execution_id", str(uuid.uuid4()))
        context.setdefault("conversation_id", context["execution_id"])

        # P0: Provide default evaluation data if not injected by caller.
        # This keeps the pipeline runnable out-of-the-box for smoke tests and local dev.
        if self.config.enable_evaluation and context.get("evaluation_data") is None:
            try:
                import pandas as pd
                from iqfmp.core.factor_engine import get_default_data_path

                data_path = get_default_data_path()
                df = pd.read_csv(data_path)

                # Standardize columns for evaluator expectations
                if "timestamp" in df.columns and "date" not in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"])

                if "symbol" not in df.columns:
                    df["symbol"] = "ETHUSDT"

                if "forward_return" not in df.columns and "close" in df.columns:
                    df["forward_return"] = df["close"].shift(-1) / df["close"] - 1
                    df = df.dropna(subset=["forward_return"]).reset_index(drop=True)

                context["evaluation_data"] = df
                logger.info(f"Injected default evaluation_data: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to inject default evaluation_data: {e}")

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

    async def _evaluate_node(self, state: AgentState) -> AgentState:
        """Evaluate generated factors (StateGraph node)."""
        agent = self._agents.get("evaluation")
        if agent is None:
            agent = FactorEvaluationAgent(
                config=self.config.evaluation_config or EvaluationAgentConfig()
            )
            self._agents["evaluation"] = agent
        return await agent.evaluate(state)

    async def _strategy_node(self, state: AgentState) -> AgentState:
        """Assemble strategy from evaluated factors (StateGraph node)."""
        agent = self._agents.get("strategy")
        if agent is None:
            agent = StrategyAssemblyAgent(
                config=self.config.strategy_config or StrategyConfig()
            )
            self._agents["strategy"] = agent
        return await agent.assemble(state)

    async def _backtest_node(self, state: AgentState) -> AgentState:
        """Optimize/backtest assembled strategy (StateGraph node)."""
        agent = self._agents.get("backtest")
        if agent is None:
            agent = BacktestOptimizationAgent(
                config=self.config.backtest_config or BacktestConfig()
            )
            self._agents["backtest"] = agent
        return await agent.optimize(state)

    async def _risk_node(self, state: AgentState) -> AgentState:
        """Run centralized risk checks (StateGraph node)."""
        agent = self._agents.get("risk")
        if agent is None:
            agent = RiskCheckAgent(config=self.config.risk_config or RiskConfig())
            self._agents["risk"] = agent
        return await agent.check(state)

    async def _generate_node(self, state: AgentState) -> AgentState:
        """Generate factors with automatic deduplication.

        Uses vector database (Qdrant) to check for similar existing factors
        before generation and stores new unique factors after generation.

        IMPORTANT: Production pipelines REQUIRE Qdrant for factor deduplication.
        Set require_vector_db=True in PipelineConfig for production use.
        """
        logger.info("Generating factors")
        context = state.context.copy()
        context["pipeline_stage"] = "generate"

        # Initialize vector search for deduplication
        similarity_searcher = None
        vector_store = None
        vector_db_active = False

        if VECTOR_AVAILABLE:
            try:
                vector_store = FactorVectorStore()
                similarity_searcher = SimilaritySearcher(
                    similarity_threshold=float(
                        context.get("dedup_threshold", self.config.dedup_threshold)
                    )
                )
                vector_db_active = True
                logger.info("Qdrant vector database connected - deduplication enabled")
            except Exception as e:
                error_msg = f"Failed to initialize Qdrant vector search: {e}"
                if self.config.require_vector_db:
                    raise VectorDBRequiredError(
                        f"{error_msg}. "
                        "Production pipelines require Qdrant for factor deduplication. "
                        "Please start Qdrant (docker compose up -d qdrant) or set require_vector_db=False."
                    )
                logger.warning(error_msg)
        else:
            error_msg = "Qdrant vector module not available (import failed)"
            if self.config.require_vector_db:
                raise VectorDBRequiredError(
                    f"{error_msg}. "
                    "Production pipelines require Qdrant for factor deduplication. "
                    "Please install qdrant-client or set require_vector_db=False."
                )
            logger.warning(error_msg)

        # Record vector DB status in context
        context["vector_db_active"] = vector_db_active

        # Get prompts from hypotheses or context
        prompts = context.get("factor_prompts", [])
        if not prompts and "hypotheses" in context:
            prompts = [h.get("description", "") for h in context["hypotheses"]]

        # Deduplication threshold from context or default
        dedup_threshold = float(context.get("dedup_threshold", self.config.dedup_threshold))

        # Map factor_family string to enum (if provided)
        family_value = context.get("factor_family")
        family_enum: Optional[FactorFamily] = None
        if isinstance(family_value, FactorFamily):
            family_enum = family_value
        elif isinstance(family_value, str) and family_value.strip():
            family_map = {f.value: f for f in FactorFamily}
            family_enum = family_map.get(family_value.strip().lower())

        # Generate factors with deduplication
        generated_factors = []
        skipped_duplicates = []

        # Prefer caller-injected llm_provider for tests/offline runs.
        llm_provider_override = context.get("llm_provider")
        factor_config = self.config.factor_config or FactorGenerationConfig(
            name="pipeline_factor_generation",
            security_check_enabled=True,
            field_constraint_enabled=True,
            include_examples=True,
        )

        async def _generate_one(prompt: str, agent: FactorGenerationAgent) -> None:
            try:
                generated = await agent.generate(
                    user_request=prompt,
                    factor_family=family_enum,
                )
                factor_data = {
                    "name": generated.name,
                    "family": generated.family.value,
                    "code": generated.code,
                    "description": generated.description,
                }

                # Check for duplicates using vector similarity
                is_duplicate = False
                similar_factor: Optional[str] = None

                if similarity_searcher:
                    try:
                        is_duplicate, similar_result = similarity_searcher.check_duplicate(
                            code=generated.code,
                            name=generated.name,
                            hypothesis=generated.description or "",
                            threshold=dedup_threshold,
                        )
                        if is_duplicate and similar_result:
                            similar_factor = similar_result.name
                            logger.info(
                                f"Factor '{generated.name}' is duplicate of "
                                f"'{similar_factor}' (similarity: {similar_result.score:.2f})"
                            )
                    except Exception as e:
                        logger.warning(f"Duplicate check failed: {e}")

                if is_duplicate:
                    skipped_duplicates.append({
                        "name": generated.name,
                        "similar_to": similar_factor,
                    })
                    return

                generated_factors.append(factor_data)

                # Store new factor in vector database
                if vector_store:
                    try:
                        factor_id = str(uuid.uuid4())
                        vector_store.add_factor(
                            factor_id=factor_id,
                            name=generated.name,
                            code=generated.code,
                            hypothesis=generated.description or "",
                            family=generated.family.value,
                        )
                        logger.info(f"Stored factor '{generated.name}' in vector DB")
                    except Exception as e:
                        logger.warning(f"Failed to store factor in vector DB: {e}")

            except Exception as e:
                logger.warning(f"Factor generation failed: {e}")

        if llm_provider_override is not None:
            generation_agent = FactorGenerationAgent(
                config=factor_config,
                llm_provider=llm_provider_override,
            )
            for prompt in prompts[:5]:
                await _generate_one(prompt, generation_agent)
        else:
            # Fallback to environment-configured provider for production.
            from iqfmp.llm.provider import LLMConfig, LLMProvider

            llm_config = LLMConfig.from_env()
            async with LLMProvider(llm_config) as llm:
                from iqfmp.llm.trace import TracingLLMProvider

                traced_llm = TracingLLMProvider(
                    llm,
                    execution_id=str(context.get("execution_id", str(uuid.uuid4()))),
                    conversation_id=str(context.get("conversation_id", "")) or None,
                    agent="factor_generation",
                )
                generation_agent = FactorGenerationAgent(
                    config=factor_config,
                    llm_provider=traced_llm,
                )
                for prompt in prompts[:5]:
                    await _generate_one(prompt, generation_agent)

        context["generated_factors"] = generated_factors
        context["skipped_duplicates"] = skipped_duplicates
        context["dedup_stats"] = {
            "total_generated": len(generated_factors),
            "duplicates_skipped": len(skipped_duplicates),
            "dedup_threshold": dedup_threshold,
        }

        logger.info(
            f"Factor generation complete: {len(generated_factors)} unique, "
            f"{len(skipped_duplicates)} duplicates skipped"
        )

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

    IMPORTANT: Production pipelines REQUIRE:
    - Qdrant for factor deduplication (require_vector_db=True)
    - Qlib for backtesting (automatically enforced in BacktestOptimizationAgent)
    - All quality checks enabled

    Args:
        checkpoint_saver: Optional checkpoint persistence

    Returns:
        Configured AgentOrchestrator

    Raises:
        VectorDBRequiredError: If Qdrant is not available
    """
    config = PipelineConfig(
        name="production_pipeline",
        enable_hypothesis=True,
        enable_evaluation=True,
        enable_strategy=True,
        enable_backtest=True,
        enable_risk_check=True,
        require_vector_db=True,  # MANDATORY for production
        dedup_threshold=0.85,
        checkpoint_enabled=True,
    )
    builder = PipelineBuilder(config)
    return builder.build(checkpoint_saver)
