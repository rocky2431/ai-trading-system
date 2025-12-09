"""LangGraph-based Agent Orchestrator for IQFMP.

This module provides a flexible agent orchestration framework built on
LangGraph concepts, with support for:
- StateGraph state machine pattern
- Checkpoint persistence
- Time-travel debugging
- Conditional routing
"""

import asyncio
import inspect
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


# === Error Classes ===

class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class NodeExecutionError(OrchestratorError):
    """Raised when a node fails during execution."""
    pass


class StateValidationError(OrchestratorError):
    """Raised when state validation fails."""
    pass


# === State Management ===

@dataclass
class AgentState:
    """Immutable state container for agent execution.

    Attributes:
        messages: Conversation history
        context: Arbitrary context data
        current_node: Currently executing node name
        metadata: Execution metadata
    """
    messages: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_node: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> "AgentState":
        """Create a new state with updated values.

        This maintains immutability - original state is unchanged.
        """
        return AgentState(
            messages=kwargs.get("messages", self.messages.copy()),
            context=kwargs.get("context", self.context.copy()),
            current_node=kwargs.get("current_node", self.current_node),
            metadata=kwargs.get("metadata", self.metadata.copy()),
        )


# === Graph Abstractions ===

@dataclass
class Node:
    """A node in the state graph.

    Nodes are the processing units that transform state.
    """
    name: str
    func: Callable[[AgentState], Union[AgentState, Any]]
    is_async: bool = field(init=False)

    def __post_init__(self) -> None:
        """Detect if function is async."""
        self.is_async = asyncio.iscoroutinefunction(self.func)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute the node with given state."""
        try:
            if self.is_async:
                result = await self.func(state)
            else:
                result = self.func(state)

            if isinstance(result, AgentState):
                return result
            return state
        except Exception as e:
            raise NodeExecutionError(f"Node '{self.name}' failed: {e}") from e


@dataclass
class Edge:
    """A simple edge connecting two nodes."""
    source: str
    target: str


@dataclass
class ConditionalEdge:
    """An edge with conditional routing logic."""
    source: str
    router: Callable[[AgentState], str]
    targets: list[str]
    _is_async: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Detect if router is async."""
        self._is_async = asyncio.iscoroutinefunction(self.router)

    async def get_target(self, state: AgentState) -> str:
        """Determine target node based on state."""
        if self._is_async:
            return await self.router(state)
        return self.router(state)


# === State Graph ===

class StateGraph:
    """A state machine graph for agent orchestration.

    The graph consists of nodes (processing functions) connected by
    edges (transitions). Supports conditional routing and cycles.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Union[Edge, ConditionalEdge]] = []
        self.entry_point: Optional[str] = None
        self.finish_points: set[str] = set()

    @property
    def finish_point(self) -> Optional[str]:
        """Get first finish point (for compatibility)."""
        return next(iter(self.finish_points)) if self.finish_points else None

    def add_node(
        self,
        name: str,
        func: Callable[[AgentState], Union[AgentState, Any]],
    ) -> "StateGraph":
        """Add a node to the graph."""
        self.nodes[name] = Node(name=name, func=func)
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        """Add a simple edge between nodes."""
        if source not in self.nodes:
            raise ValueError(f"Node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Node '{target}' not found")

        self.edges.append(Edge(source=source, target=target))
        return self

    def add_conditional_edge(
        self,
        source: str,
        router: Callable[[AgentState], str],
        targets: list[str],
    ) -> "StateGraph":
        """Add a conditional edge with routing logic."""
        if source not in self.nodes:
            raise ValueError(f"Node '{source}' not found")
        for target in targets:
            if target not in self.nodes:
                raise ValueError(f"Node '{target}' not found")

        self.edges.append(ConditionalEdge(
            source=source,
            router=router,
            targets=targets,
        ))
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        """Set the entry point node."""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")
        self.entry_point = name
        return self

    def set_finish_point(self, name: str) -> "StateGraph":
        """Set a finish point node."""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")
        self.finish_points.add(name)
        return self

    def compile(self) -> "CompiledGraph":
        """Compile the graph for execution."""
        if not self.entry_point:
            raise ValueError("Entry point not set")
        if not self.finish_points:
            raise ValueError("No finish points set")
        return CompiledGraph(self)


class CompiledGraph:
    """A compiled, executable state graph."""

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph
        self._adjacency: dict[str, list[Union[Edge, ConditionalEdge]]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build adjacency list for efficient traversal."""
        for edge in self.graph.edges:
            source = edge.source
            if source not in self._adjacency:
                self._adjacency[source] = []
            self._adjacency[source].append(edge)

    async def get_next_node(
        self,
        current: str,
        state: AgentState,
    ) -> Optional[str]:
        """Get the next node to execute."""
        edges = self._adjacency.get(current, [])

        for edge in edges:
            if isinstance(edge, ConditionalEdge):
                return await edge.get_target(state)
            elif isinstance(edge, Edge):
                return edge.target

        return None


# === Checkpoint System ===

@dataclass
class Checkpoint:
    """A snapshot of execution state."""
    id: str
    state: AgentState
    node: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointSaver(ABC):
    """Abstract base class for checkpoint persistence."""

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        pass

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID."""
        pass

    @abstractmethod
    async def list(self) -> list[Checkpoint]:
        """List all checkpoints."""
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        pass


class MemorySaver(CheckpointSaver):
    """In-memory checkpoint storage for testing."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to memory."""
        self._checkpoints[checkpoint.id] = checkpoint

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from memory."""
        return self._checkpoints.get(checkpoint_id)

    async def list(self) -> list[Checkpoint]:
        """List all checkpoints."""
        return list(self._checkpoints.values())

    async def delete(self, checkpoint_id: str) -> None:
        """Delete checkpoint from memory."""
        self._checkpoints.pop(checkpoint_id, None)


# === Orchestrator Configuration ===

@dataclass
class OrchestratorConfig:
    """Configuration for AgentOrchestrator."""
    name: str
    max_iterations: int = 100
    timeout: float = 300.0  # 5 minutes
    checkpoint_enabled: bool = False
    checkpoint_interval: int = 1  # Checkpoint every N nodes
    api_key: Optional[str] = None  # For future LLM integration

    def to_safe_dict(self) -> dict[str, Any]:
        """Return config dict without sensitive values."""
        return {
            "name": self.name,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "checkpoint_enabled": self.checkpoint_enabled,
        }


# === Agent Orchestrator ===

class AgentOrchestrator:
    """Orchestrator for executing agent graphs.

    This class manages the execution of a StateGraph, handling:
    - State transitions
    - Checkpointing
    - Error handling
    - Timeout management
    """

    def __init__(
        self,
        graph: StateGraph,
        config: OrchestratorConfig,
        checkpoint_saver: Optional[CheckpointSaver] = None,
    ) -> None:
        self._graph = graph
        self._config = config
        self._compiled = graph.compile()
        self._checkpoint_saver = checkpoint_saver or MemorySaver()
        self._execution_id: Optional[str] = None

    def __repr__(self) -> str:
        """String representation without sensitive data."""
        return f"AgentOrchestrator(name={self._config.name})"

    async def run(
        self,
        initial_state: AgentState,
        execution_id: Optional[str] = None,
    ) -> AgentState:
        """Execute the graph from the beginning.

        Args:
            initial_state: Starting state
            execution_id: Optional execution identifier

        Returns:
            Final state after execution

        Raises:
            OrchestratorError: On execution failure
        """
        self._execution_id = execution_id or str(uuid.uuid4())
        current_node = self._graph.entry_point
        state = initial_state.update(current_node=current_node)
        iteration = 0
        nodes_since_checkpoint = 0

        try:
            async with asyncio.timeout(self._config.timeout):
                while current_node is not None:
                    # Check iteration limit
                    iteration += 1
                    if iteration > self._config.max_iterations:
                        raise OrchestratorError(
                            f"Max iterations ({self._config.max_iterations}) exceeded"
                        )

                    # Execute current node
                    node = self._graph.nodes[current_node]
                    state = await node.execute(state)
                    state = state.update(current_node=current_node)

                    # Checkpoint if enabled
                    if self._config.checkpoint_enabled:
                        nodes_since_checkpoint += 1
                        if nodes_since_checkpoint >= self._config.checkpoint_interval:
                            await self._save_checkpoint(state, current_node)
                            nodes_since_checkpoint = 0

                    # Check if we've reached a finish point
                    if current_node in self._graph.finish_points:
                        break

                    # Get next node
                    current_node = await self._compiled.get_next_node(
                        current_node, state
                    )

        except asyncio.TimeoutError:
            raise OrchestratorError(
                f"Timeout ({self._config.timeout}s) exceeded"
            )

        return state

    async def resume(self, checkpoint_id: str) -> AgentState:
        """Resume execution from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to resume from

        Returns:
            Final state after execution
        """
        checkpoint = await self._checkpoint_saver.load(checkpoint_id)
        if not checkpoint:
            raise OrchestratorError(f"Checkpoint '{checkpoint_id}' not found")

        return await self.run(
            checkpoint.state,
            execution_id=self._execution_id,
        )

    async def _save_checkpoint(self, state: AgentState, node: str) -> None:
        """Save a checkpoint."""
        checkpoint = Checkpoint(
            id=f"{self._execution_id}-{node}-{time.time()}",
            state=state,
            node=node,
            timestamp=time.time(),
            metadata={"execution_id": self._execution_id},
        )
        await self._checkpoint_saver.save(checkpoint)

    async def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints."""
        return await self._checkpoint_saver.list()

    async def get_state_at(self, checkpoint_id: str) -> Optional[AgentState]:
        """Get state at a specific checkpoint (time-travel)."""
        checkpoint = await self._checkpoint_saver.load(checkpoint_id)
        if checkpoint:
            return checkpoint.state
        return None

    def get_graph_structure(self) -> dict[str, Any]:
        """Get graph structure for visualization."""
        return {
            "name": self._graph.name,
            "nodes": list(self._graph.nodes.keys()),
            "edges": [
                {"source": e.source, "target": e.target if isinstance(e, Edge) else e.targets}
                for e in self._graph.edges
            ],
            "entry_point": self._graph.entry_point,
            "finish_points": list(self._graph.finish_points),
        }
