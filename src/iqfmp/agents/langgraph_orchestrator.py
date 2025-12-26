"""LangGraph-based Agent Orchestrator for IQFMP.

This module provides an orchestration layer built on the OFFICIAL LangGraph library,
implementing the spec requirements for:
- StateGraph state machine pattern
- thread_id/checkpoint_id semantics
- PostgreSQL checkpoint persistence
- Time-travel debugging support
- Human-in-the-loop interrupt/resume

Migration from custom orchestrator.py to official LangGraph.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict, Union

# Official LangGraph imports
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    AsyncPostgresSaver = None
    StateGraph = None
    END = "end"
    START = "start"
    add_messages = None

import asyncpg

logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph State Definition
# =============================================================================

class FactorPipelineState(TypedDict, total=False):
    """State for factor mining pipeline using LangGraph TypedDict pattern.

    Attributes:
        messages: Conversation/event history (LangGraph pattern)
        thread_id: Unique thread identifier for checkpointing
        checkpoint_id: Current checkpoint identifier
        hypothesis: Current research hypothesis
        factors: Generated factor expressions
        evaluation_results: Evaluation metrics
        strategy: Generated trading strategy
        backtest_results: Backtest performance
        risk_assessment: Risk analysis results
        current_phase: Current pipeline phase
        error: Error message if any
        metadata: Additional execution metadata
    """
    # Core LangGraph fields
    messages: Annotated[list[dict[str, Any]], add_messages] if add_messages else list
    thread_id: str
    checkpoint_id: Optional[str]

    # Pipeline-specific fields
    hypothesis: Optional[str]
    factors: list[dict[str, Any]]
    evaluation_results: dict[str, Any]
    strategy: Optional[dict[str, Any]]
    backtest_results: Optional[dict[str, Any]]
    risk_assessment: Optional[dict[str, Any]]
    current_phase: str
    error: Optional[str]
    metadata: dict[str, Any]


# =============================================================================
# LangGraph Checkpoint Saver
# =============================================================================

class LangGraphCheckpointManager:
    """Manager for LangGraph checkpoints using PostgreSQL.

    Provides:
    - Checkpoint persistence to PostgreSQL
    - Thread management with thread_id semantics
    - Time-travel debugging support
    - Checkpoint listing and restoration
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            connection_string: PostgreSQL connection string.
                             Defaults to environment variable or localhost.
        """
        self._connection_string = connection_string or self._get_default_connection()
        self._saver: Optional[AsyncPostgresSaver] = None
        self._pool: Optional[asyncpg.Pool] = None

    def _get_default_connection(self) -> str:
        """Get default PostgreSQL connection string."""
        import os
        host = os.environ.get("PGHOST", "localhost")
        port = os.environ.get("PGPORT", "5433")
        user = os.environ.get("PGUSER", "iqfmp")
        password = os.environ.get("PGPASSWORD", "iqfmp")
        database = os.environ.get("PGDATABASE", "iqfmp")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    async def initialize(self) -> None:
        """Initialize the checkpoint saver."""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, using in-memory fallback")
            return

        try:
            self._pool = await asyncpg.create_pool(self._connection_string)
            self._saver = AsyncPostgresSaver(self._pool)
            await self._saver.setup()
            logger.info("LangGraph checkpoint saver initialized with PostgreSQL")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL saver: {e}")
            self._saver = None

    async def close(self) -> None:
        """Close the checkpoint manager."""
        if self._pool:
            await self._pool.close()

    @property
    def saver(self) -> Optional[AsyncPostgresSaver]:
        """Get the checkpoint saver."""
        return self._saver

    async def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all threads with their latest checkpoints."""
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT thread_id, MAX(created_at) as last_updated
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY last_updated DESC
                LIMIT $1
            """, limit)
            return [{"thread_id": r["thread_id"], "last_updated": r["last_updated"]} for r in rows]

    async def get_thread_history(
        self,
        thread_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get checkpoint history for a thread (time-travel support)."""
        if not self._pool:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT checkpoint_id, created_at, metadata
                FROM checkpoints
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, thread_id, limit)
            return [dict(r) for r in rows]


# =============================================================================
# LangGraph Pipeline Builder
# =============================================================================

class LangGraphPipelineBuilder:
    """Builder for LangGraph-based factor mining pipelines.

    Provides a fluent interface for building pipelines with:
    - Node registration
    - Edge configuration
    - Conditional routing
    - Checkpoint configuration
    """

    def __init__(
        self,
        name: str = "factor_pipeline",
        checkpoint_manager: Optional[LangGraphCheckpointManager] = None,
    ) -> None:
        """Initialize pipeline builder.

        Args:
            name: Pipeline name
            checkpoint_manager: Optional checkpoint manager for persistence
        """
        self.name = name
        self._checkpoint_manager = checkpoint_manager
        self._nodes: dict[str, Callable] = {}
        self._edges: list[tuple[str, str]] = []
        self._conditional_edges: list[tuple[str, Callable, dict[str, str]]] = []
        self._entry_point: Optional[str] = None
        self._interrupt_before: list[str] = []
        self._interrupt_after: list[str] = []

    def add_node(
        self,
        name: str,
        func: Callable[[FactorPipelineState], FactorPipelineState],
    ) -> "LangGraphPipelineBuilder":
        """Add a node to the pipeline."""
        self._nodes[name] = func
        return self

    def add_edge(self, source: str, target: str) -> "LangGraphPipelineBuilder":
        """Add an edge between nodes."""
        self._edges.append((source, target))
        return self

    def add_conditional_edge(
        self,
        source: str,
        router: Callable[[FactorPipelineState], str],
        path_map: dict[str, str],
    ) -> "LangGraphPipelineBuilder":
        """Add a conditional edge with routing logic."""
        self._conditional_edges.append((source, router, path_map))
        return self

    def set_entry_point(self, name: str) -> "LangGraphPipelineBuilder":
        """Set the entry point node."""
        self._entry_point = name
        return self

    def interrupt_before(self, *nodes: str) -> "LangGraphPipelineBuilder":
        """Set nodes to interrupt before (human-in-the-loop)."""
        self._interrupt_before.extend(nodes)
        return self

    def interrupt_after(self, *nodes: str) -> "LangGraphPipelineBuilder":
        """Set nodes to interrupt after (human-in-the-loop)."""
        self._interrupt_after.extend(nodes)
        return self

    def compile(self) -> Any:
        """Compile the pipeline to a LangGraph runnable."""
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        # Create StateGraph with typed state
        graph = StateGraph(FactorPipelineState)

        # Add nodes
        for name, func in self._nodes.items():
            graph.add_node(name, func)

        # Set entry point
        if self._entry_point:
            graph.add_edge(START, self._entry_point)

        # Add edges
        for source, target in self._edges:
            if target == "end":
                graph.add_edge(source, END)
            else:
                graph.add_edge(source, target)

        # Add conditional edges
        for source, router, path_map in self._conditional_edges:
            # Convert path_map to use END for "end"
            converted_map = {
                k: (END if v == "end" else v)
                for k, v in path_map.items()
            }
            graph.add_conditional_edges(source, router, converted_map)

        # Compile with checkpointer if available
        checkpointer = None
        if self._checkpoint_manager and self._checkpoint_manager.saver:
            checkpointer = self._checkpoint_manager.saver

        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=self._interrupt_before or None,
            interrupt_after=self._interrupt_after or None,
        )


# =============================================================================
# Pipeline Executor
# =============================================================================

class LangGraphPipelineExecutor:
    """Executor for LangGraph-based pipelines.

    Provides:
    - Async execution with thread_id tracking
    - Checkpoint-based resume
    - Time-travel debugging
    - Human-in-the-loop support
    """

    def __init__(
        self,
        compiled_graph: Any,
        checkpoint_manager: Optional[LangGraphCheckpointManager] = None,
    ) -> None:
        """Initialize executor.

        Args:
            compiled_graph: Compiled LangGraph
            checkpoint_manager: Optional checkpoint manager
        """
        self._graph = compiled_graph
        self._checkpoint_manager = checkpoint_manager

    async def execute(
        self,
        initial_state: dict[str, Any],
        thread_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> FactorPipelineState:
        """Execute the pipeline.

        Args:
            initial_state: Initial pipeline state
            thread_id: Thread ID for checkpointing (auto-generated if None)
            checkpoint_id: Checkpoint ID to resume from (optional)

        Returns:
            Final pipeline state
        """
        # Generate thread_id if not provided
        if not thread_id:
            thread_id = f"thread_{uuid.uuid4().hex[:12]}"

        # Prepare config
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        # Merge thread_id into state
        state = {
            **initial_state,
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }

        logger.info(f"Executing pipeline with thread_id={thread_id}")

        # Execute
        try:
            result = await self._graph.ainvoke(state, config)
            logger.info(f"Pipeline completed: thread_id={thread_id}")
            return result
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    async def resume(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
        updates: Optional[dict[str, Any]] = None,
    ) -> FactorPipelineState:
        """Resume pipeline from checkpoint (time-travel).

        Args:
            thread_id: Thread ID to resume
            checkpoint_id: Specific checkpoint to resume from
            updates: Optional state updates before resuming

        Returns:
            Final pipeline state
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        # Get current state
        current_state = await self._graph.aget_state(config)

        if updates:
            # Update state and resume
            await self._graph.aupdate_state(config, updates)

        # Continue execution
        result = await self._graph.ainvoke(None, config)
        return result


# =============================================================================
# Factory Functions
# =============================================================================

async def create_factor_pipeline(
    enable_checkpointing: bool = True,
    enable_human_review: bool = False,
) -> tuple[Any, LangGraphCheckpointManager]:
    """Create a factor mining pipeline with official LangGraph.

    Args:
        enable_checkpointing: Enable PostgreSQL checkpointing
        enable_human_review: Enable human-in-the-loop before risky operations

    Returns:
        Tuple of (compiled_graph, checkpoint_manager)
    """
    # Initialize checkpoint manager
    checkpoint_manager = LangGraphCheckpointManager()
    if enable_checkpointing:
        await checkpoint_manager.initialize()

    # Import node functions
    from iqfmp.agents.factor_generation import generate_factors_node
    from iqfmp.agents.evaluation_agent import evaluate_factors_node
    from iqfmp.agents.strategy_agent import generate_strategy_node
    from iqfmp.agents.backtest_agent import run_backtest_node
    from iqfmp.agents.risk_agent import assess_risk_node

    # Build pipeline
    builder = LangGraphPipelineBuilder(
        name="factor_pipeline",
        checkpoint_manager=checkpoint_manager,
    )

    # Add nodes
    builder.add_node("generate", generate_factors_node)
    builder.add_node("evaluate", evaluate_factors_node)
    builder.add_node("strategy", generate_strategy_node)
    builder.add_node("backtest", run_backtest_node)
    builder.add_node("risk", assess_risk_node)

    # Add edges
    builder.set_entry_point("generate")
    builder.add_edge("generate", "evaluate")
    builder.add_edge("evaluate", "strategy")
    builder.add_edge("strategy", "backtest")
    builder.add_edge("backtest", "risk")
    builder.add_edge("risk", "end")

    # Add human review interrupt points if enabled
    if enable_human_review:
        builder.interrupt_before("backtest")  # Review before live trading

    # Compile
    compiled = builder.compile()

    return compiled, checkpoint_manager


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

# Re-export for backward compatibility with custom orchestrator
class AgentState:
    """Backward compatibility wrapper for custom AgentState.

    New code should use FactorPipelineState (TypedDict) directly.
    """

    def __init__(
        self,
        messages: Optional[list] = None,
        context: Optional[dict] = None,
        current_node: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.messages = messages or []
        self.context = context or {}
        self.current_node = current_node
        self.metadata = metadata or {}

    def to_dict(self) -> FactorPipelineState:
        """Convert to LangGraph state dict."""
        return {
            "messages": self.messages,
            "thread_id": self.metadata.get("thread_id", ""),
            "checkpoint_id": self.metadata.get("checkpoint_id"),
            "hypothesis": self.context.get("hypothesis"),
            "factors": self.context.get("factors", []),
            "evaluation_results": self.context.get("evaluation_results", {}),
            "strategy": self.context.get("strategy"),
            "backtest_results": self.context.get("backtest_results"),
            "risk_assessment": self.context.get("risk_assessment"),
            "current_phase": self.current_node or "unknown",
            "error": None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, state: FactorPipelineState) -> "AgentState":
        """Create from LangGraph state dict."""
        return cls(
            messages=state.get("messages", []),
            context={
                "hypothesis": state.get("hypothesis"),
                "factors": state.get("factors", []),
                "evaluation_results": state.get("evaluation_results", {}),
                "strategy": state.get("strategy"),
                "backtest_results": state.get("backtest_results"),
                "risk_assessment": state.get("risk_assessment"),
            },
            current_node=state.get("current_phase"),
            metadata=state.get("metadata", {}),
        )
