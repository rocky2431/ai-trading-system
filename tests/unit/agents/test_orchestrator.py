"""Tests for LangGraph Agent Orchestrator (Task 7).

Six-dimensional test coverage:
1. Functional: StateGraph, AgentOrchestrator, state management
2. Boundary: Edge cases for state and execution
3. Exception: Error handling for node failures
4. Performance: Execution time and concurrency
5. Security: State isolation
6. Compatibility: Different node types and state shapes
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, TypedDict
import asyncio
import time

from iqfmp.agents.orchestrator import (
    AgentOrchestrator,
    OrchestratorConfig,
    AgentState,
    StateGraph,
    Node,
    Edge,
    ConditionalEdge,
    Checkpoint,
    CheckpointSaver,
    MemorySaver,
    OrchestratorError,
    NodeExecutionError,
    StateValidationError,
)


class TestAgentState:
    """Tests for AgentState management."""

    def test_create_empty_state(self) -> None:
        """Test creating empty agent state."""
        state = AgentState()
        assert state.messages == []
        assert state.context == {}
        assert state.current_node is None

    def test_state_with_messages(self) -> None:
        """Test state with messages."""
        state = AgentState(
            messages=[
                {"role": "user", "content": "Generate a momentum factor"},
                {"role": "assistant", "content": "I'll create a momentum factor..."},
            ]
        )
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"

    def test_state_with_context(self) -> None:
        """Test state with context data."""
        state = AgentState(
            context={
                "task_type": "factor_generation",
                "factor_family": "momentum",
                "user_id": "user123",
            }
        )
        assert state.context["task_type"] == "factor_generation"

    def test_state_update(self) -> None:
        """Test updating state."""
        state = AgentState()
        updated = state.update(
            messages=[{"role": "user", "content": "Hello"}],
            current_node="greeting_node",
        )
        assert len(updated.messages) == 1
        assert updated.current_node == "greeting_node"

    def test_state_immutability(self) -> None:
        """Test state is immutable (creates new instance on update)."""
        state = AgentState()
        updated = state.update(context={"key": "value"})
        assert state.context == {}
        assert updated.context == {"key": "value"}


class TestNode:
    """Tests for Node abstraction."""

    def test_create_sync_node(self) -> None:
        """Test creating synchronous node."""
        def process(state: AgentState) -> AgentState:
            return state.update(context={"processed": True})

        node = Node(name="processor", func=process)
        assert node.name == "processor"
        assert node.is_async is False

    def test_create_async_node(self) -> None:
        """Test creating asynchronous node."""
        async def async_process(state: AgentState) -> AgentState:
            await asyncio.sleep(0.01)
            return state.update(context={"async_processed": True})

        node = Node(name="async_processor", func=async_process)
        assert node.name == "async_processor"
        assert node.is_async is True

    @pytest.mark.asyncio
    async def test_node_execution(self) -> None:
        """Test node execution."""
        def process(state: AgentState) -> AgentState:
            return state.update(context={"executed": True})

        node = Node(name="test_node", func=process)
        state = AgentState()
        result = await node.execute(state)
        assert result.context["executed"] is True


class TestEdge:
    """Tests for Edge abstraction."""

    def test_create_edge(self) -> None:
        """Test creating simple edge."""
        edge = Edge(source="node_a", target="node_b")
        assert edge.source == "node_a"
        assert edge.target == "node_b"

    def test_conditional_edge(self) -> None:
        """Test conditional edge routing."""
        def router(state: AgentState) -> str:
            if state.context.get("approved"):
                return "approved_node"
            return "review_node"

        edge = ConditionalEdge(
            source="check_node",
            router=router,
            targets=["approved_node", "review_node"],
        )
        assert edge.source == "check_node"
        assert "approved_node" in edge.targets

    @pytest.mark.asyncio
    async def test_conditional_edge_routing(self) -> None:
        """Test conditional edge actually routes correctly."""
        def router(state: AgentState) -> str:
            return "path_a" if state.context.get("flag") else "path_b"

        edge = ConditionalEdge(
            source="start",
            router=router,
            targets=["path_a", "path_b"],
        )

        state_with_flag = AgentState(context={"flag": True})
        state_without_flag = AgentState(context={"flag": False})

        assert await edge.get_target(state_with_flag) == "path_a"
        assert await edge.get_target(state_without_flag) == "path_b"


class TestStateGraph:
    """Tests for StateGraph construction."""

    def test_create_empty_graph(self) -> None:
        """Test creating empty state graph."""
        graph = StateGraph(name="test_graph")
        assert graph.name == "test_graph"
        assert len(graph.nodes) == 0

    def test_add_node(self) -> None:
        """Test adding node to graph."""
        graph = StateGraph(name="test")

        def process(state: AgentState) -> AgentState:
            return state

        graph.add_node("processor", process)
        assert "processor" in graph.nodes

    def test_add_edge(self) -> None:
        """Test adding edge to graph."""
        graph = StateGraph(name="test")

        def node_a(state: AgentState) -> AgentState:
            return state

        def node_b(state: AgentState) -> AgentState:
            return state

        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge("a", "b")

        assert any(e.source == "a" and e.target == "b" for e in graph.edges)

    def test_add_conditional_edge(self) -> None:
        """Test adding conditional edge."""
        graph = StateGraph(name="test")

        def router(state: AgentState) -> str:
            return "b"

        graph.add_node("a", lambda s: s)
        graph.add_node("b", lambda s: s)
        graph.add_node("c", lambda s: s)
        graph.add_conditional_edge("a", router, ["b", "c"])

        cond_edges = [e for e in graph.edges if isinstance(e, ConditionalEdge)]
        assert len(cond_edges) == 1

    def test_set_entry_point(self) -> None:
        """Test setting entry point."""
        graph = StateGraph(name="test")
        graph.add_node("start", lambda s: s)
        graph.set_entry_point("start")
        assert graph.entry_point == "start"

    def test_set_finish_point(self) -> None:
        """Test setting finish point."""
        graph = StateGraph(name="test")
        graph.add_node("end", lambda s: s)
        graph.set_finish_point("end")
        assert graph.finish_point == "end"

    def test_compile_graph(self) -> None:
        """Test compiling graph."""
        graph = StateGraph(name="test")
        graph.add_node("start", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_edge("start", "end")
        graph.set_entry_point("start")
        graph.set_finish_point("end")

        compiled = graph.compile()
        assert compiled is not None


class TestAgentOrchestratorFunctional:
    """Functional tests for AgentOrchestrator."""

    @pytest.fixture
    def simple_graph(self) -> StateGraph:
        """Create a simple test graph."""
        graph = StateGraph(name="simple_test")

        def node_a(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "a_executed": True})

        def node_b(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "b_executed": True})

        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge("a", "b")
        graph.set_entry_point("a")
        graph.set_finish_point("b")

        return graph

    @pytest.fixture
    def orchestrator(self, simple_graph: StateGraph) -> AgentOrchestrator:
        """Create orchestrator with simple graph."""
        config = OrchestratorConfig(
            name="test_orchestrator",
            max_iterations=10,
            timeout=30,
        )
        return AgentOrchestrator(graph=simple_graph, config=config)

    @pytest.mark.asyncio
    async def test_basic_execution(self, orchestrator: AgentOrchestrator) -> None:
        """Test basic graph execution."""
        initial_state = AgentState()
        result = await orchestrator.run(initial_state)

        assert result.context.get("a_executed") is True
        assert result.context.get("b_executed") is True

    @pytest.mark.asyncio
    async def test_execution_with_input(self, orchestrator: AgentOrchestrator) -> None:
        """Test execution with initial input."""
        initial_state = AgentState(
            messages=[{"role": "user", "content": "Test input"}],
            context={"input_provided": True},
        )
        result = await orchestrator.run(initial_state)

        assert result.context.get("input_provided") is True
        assert len(result.messages) == 1

    @pytest.mark.asyncio
    async def test_conditional_routing(self) -> None:
        """Test conditional edge routing during execution."""
        graph = StateGraph(name="conditional_test")

        def router(state: AgentState) -> str:
            return "success" if state.context.get("valid") else "failure"

        def check_node(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "checked": True})

        def success_node(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "result": "success"})

        def failure_node(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "result": "failure"})

        graph.add_node("check", check_node)
        graph.add_node("success", success_node)
        graph.add_node("failure", failure_node)
        graph.add_conditional_edge("check", router, ["success", "failure"])
        graph.set_entry_point("check")
        graph.set_finish_point("success")
        graph.set_finish_point("failure")

        config = OrchestratorConfig(name="cond_test")
        orchestrator = AgentOrchestrator(graph=graph, config=config)

        # Test success path
        state_valid = AgentState(context={"valid": True})
        result = await orchestrator.run(state_valid)
        assert result.context.get("result") == "success"

        # Test failure path
        state_invalid = AgentState(context={"valid": False})
        result = await orchestrator.run(state_invalid)
        assert result.context.get("result") == "failure"

    def test_get_graph_structure(self, orchestrator: AgentOrchestrator) -> None:
        """Test getting graph structure for visualization."""
        structure = orchestrator.get_graph_structure()
        assert "nodes" in structure
        assert "edges" in structure
        assert len(structure["nodes"]) == 2


class TestCheckpointSaver:
    """Tests for checkpoint persistence."""

    @pytest.fixture
    def memory_saver(self) -> MemorySaver:
        """Create in-memory checkpoint saver."""
        return MemorySaver()

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, memory_saver: MemorySaver) -> None:
        """Test saving checkpoint."""
        state = AgentState(context={"step": 1})
        checkpoint = Checkpoint(
            id="cp-001",
            state=state,
            node="test_node",
            timestamp=time.time(),
        )

        await memory_saver.save(checkpoint)
        loaded = await memory_saver.load("cp-001")

        assert loaded is not None
        assert loaded.id == "cp-001"
        assert loaded.state.context["step"] == 1

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, memory_saver: MemorySaver) -> None:
        """Test listing checkpoints."""
        for i in range(3):
            checkpoint = Checkpoint(
                id=f"cp-{i}",
                state=AgentState(context={"step": i}),
                node=f"node_{i}",
                timestamp=time.time(),
            )
            await memory_saver.save(checkpoint)

        checkpoints = await memory_saver.list()
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, memory_saver: MemorySaver) -> None:
        """Test deleting checkpoint."""
        checkpoint = Checkpoint(
            id="cp-delete",
            state=AgentState(),
            node="test",
            timestamp=time.time(),
        )
        await memory_saver.save(checkpoint)
        await memory_saver.delete("cp-delete")

        loaded = await memory_saver.load("cp-delete")
        assert loaded is None


class TestOrchestratorWithCheckpoints:
    """Tests for orchestrator with checkpoint support."""

    @pytest.fixture
    def orchestrator_with_checkpoints(self) -> AgentOrchestrator:
        """Create orchestrator with checkpoint support."""
        graph = StateGraph(name="checkpoint_test")

        def step1(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "step1": True})

        def step2(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "step2": True})

        def step3(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "step3": True})

        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_node("step3", step3)
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.set_entry_point("step1")
        graph.set_finish_point("step3")

        config = OrchestratorConfig(
            name="cp_test",
            checkpoint_enabled=True,
            checkpoint_interval=1,  # Checkpoint after each node
        )
        saver = MemorySaver()
        return AgentOrchestrator(graph=graph, config=config, checkpoint_saver=saver)

    @pytest.mark.asyncio
    async def test_checkpoints_created(
        self, orchestrator_with_checkpoints: AgentOrchestrator
    ) -> None:
        """Test checkpoints are created during execution."""
        await orchestrator_with_checkpoints.run(AgentState())
        checkpoints = await orchestrator_with_checkpoints.list_checkpoints()
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(
        self, orchestrator_with_checkpoints: AgentOrchestrator
    ) -> None:
        """Test resuming execution from checkpoint."""
        # Run initially
        await orchestrator_with_checkpoints.run(AgentState())
        checkpoints = await orchestrator_with_checkpoints.list_checkpoints()

        # Resume from first checkpoint
        if checkpoints:
            result = await orchestrator_with_checkpoints.resume(checkpoints[0].id)
            assert result is not None

    @pytest.mark.asyncio
    async def test_time_travel(
        self, orchestrator_with_checkpoints: AgentOrchestrator
    ) -> None:
        """Test time-travel to previous state."""
        initial_state = AgentState(context={"initial": True})
        await orchestrator_with_checkpoints.run(initial_state)

        checkpoints = await orchestrator_with_checkpoints.list_checkpoints()
        if checkpoints:
            # Get state at checkpoint
            state_at_checkpoint = await orchestrator_with_checkpoints.get_state_at(
                checkpoints[0].id
            )
            assert state_at_checkpoint is not None


class TestOrchestratorBoundary:
    """Boundary tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_state_execution(self) -> None:
        """Test execution with minimal state."""
        graph = StateGraph(name="minimal")
        graph.add_node("pass", lambda s: s)
        graph.set_entry_point("pass")
        graph.set_finish_point("pass")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="minimal_test"),
        )

        result = await orchestrator.run(AgentState())
        assert result is not None

    @pytest.mark.asyncio
    async def test_large_state(self) -> None:
        """Test handling large state objects."""
        graph = StateGraph(name="large_state")
        graph.add_node("process", lambda s: s)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="large_test"),
        )

        # Create large state
        large_context = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
        large_state = AgentState(context=large_context)

        result = await orchestrator.run(large_state)
        assert len(result.context) == 1000

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self) -> None:
        """Test max iterations limit prevents infinite loops."""
        graph = StateGraph(name="loop")

        def increment(state: AgentState) -> AgentState:
            count = state.context.get("count", 0)
            return state.update(context={"count": count + 1})

        def router(state: AgentState) -> str:
            # Always loop back - never reach end
            return "increment"

        graph.add_node("increment", increment)
        graph.add_node("end", lambda s: s)  # Unreachable end node
        graph.add_conditional_edge("increment", router, ["increment", "end"])
        graph.set_entry_point("increment")
        graph.set_finish_point("end")

        config = OrchestratorConfig(name="loop_test", max_iterations=5)
        orchestrator = AgentOrchestrator(graph=graph, config=config)

        with pytest.raises(OrchestratorError, match="Max iterations"):
            await orchestrator.run(AgentState())


class TestOrchestratorException:
    """Exception handling tests."""

    @pytest.mark.asyncio
    async def test_node_execution_error(self) -> None:
        """Test handling node execution errors."""
        graph = StateGraph(name="error_test")

        def failing_node(state: AgentState) -> AgentState:
            raise ValueError("Node failed!")

        graph.add_node("fail", failing_node)
        graph.set_entry_point("fail")
        graph.set_finish_point("fail")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="error_test"),
        )

        with pytest.raises(NodeExecutionError):
            await orchestrator.run(AgentState())

    @pytest.mark.asyncio
    async def test_invalid_node_reference(self) -> None:
        """Test handling invalid node references."""
        graph = StateGraph(name="invalid_ref")
        graph.add_node("start", lambda s: s)
        graph.set_entry_point("start")

        # Add edge to non-existent node
        with pytest.raises(ValueError, match="Node .* not found"):
            graph.add_edge("start", "nonexistent")

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test timeout handling for slow nodes."""
        graph = StateGraph(name="timeout_test")

        async def slow_node(state: AgentState) -> AgentState:
            await asyncio.sleep(10)  # Very slow
            return state

        graph.add_node("slow", slow_node)
        graph.set_entry_point("slow")
        graph.set_finish_point("slow")

        config = OrchestratorConfig(name="timeout_test", timeout=0.1)
        orchestrator = AgentOrchestrator(graph=graph, config=config)

        with pytest.raises(OrchestratorError, match="Timeout"):
            await orchestrator.run(AgentState())


class TestOrchestratorSecurity:
    """Security tests for state isolation."""

    @pytest.mark.asyncio
    async def test_state_isolation_between_runs(self) -> None:
        """Test state is isolated between different runs."""
        graph = StateGraph(name="isolation_test")

        def set_secret(state: AgentState) -> AgentState:
            # Merge with existing context
            return state.update(context={**state.context, "secret": "sensitive_data"})

        graph.add_node("set_secret", set_secret)
        graph.set_entry_point("set_secret")
        graph.set_finish_point("set_secret")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="isolation_test"),
        )

        # Run 1
        result1 = await orchestrator.run(AgentState(context={"run": 1}))

        # Run 2 should not see Run 1's data in initial state
        result2 = await orchestrator.run(AgentState(context={"run": 2}))

        assert result1.context.get("run") == 1
        assert result2.context.get("run") == 2
        # Both should have secret set
        assert result1.context.get("secret") == "sensitive_data"
        assert result2.context.get("secret") == "sensitive_data"

    def test_config_not_exposed(self) -> None:
        """Test sensitive config not exposed in repr."""
        config = OrchestratorConfig(
            name="test",
            api_key="secret-key-123",  # If there was an API key
        )
        graph = StateGraph(name="test")
        graph.add_node("n", lambda s: s)
        graph.set_entry_point("n")
        graph.set_finish_point("n")

        orchestrator = AgentOrchestrator(graph=graph, config=config)
        repr_str = repr(orchestrator)

        # Should not contain sensitive data
        assert "secret-key" not in repr_str


class TestOrchestratorPerformance:
    """Performance tests."""

    @pytest.mark.asyncio
    async def test_execution_time(self) -> None:
        """Test execution completes in reasonable time."""
        graph = StateGraph(name="perf_test")

        def quick_node(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "processed": True})

        for i in range(10):
            graph.add_node(f"node_{i}", quick_node)

        for i in range(9):
            graph.add_edge(f"node_{i}", f"node_{i+1}")

        graph.set_entry_point("node_0")
        graph.set_finish_point("node_9")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="perf_test"),
        )

        start = time.time()
        await orchestrator.run(AgentState())
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in under 1 second

    @pytest.mark.asyncio
    async def test_concurrent_runs(self) -> None:
        """Test concurrent orchestrator runs."""
        graph = StateGraph(name="concurrent_test")

        async def async_node(state: AgentState) -> AgentState:
            await asyncio.sleep(0.01)
            return state.update(context={"done": True})

        graph.add_node("async", async_node)
        graph.set_entry_point("async")
        graph.set_finish_point("async")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="concurrent_test"),
        )

        # Run 5 concurrent executions
        tasks = [orchestrator.run(AgentState(context={"id": i})) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.context.get("done") for r in results)


class TestOrchestratorCompatibility:
    """Compatibility tests for different configurations."""

    @pytest.mark.asyncio
    async def test_sync_and_async_nodes_mixed(self) -> None:
        """Test mixing sync and async nodes."""
        graph = StateGraph(name="mixed_test")

        def sync_node(state: AgentState) -> AgentState:
            return state.update(context={**state.context, "sync": True})

        async def async_node(state: AgentState) -> AgentState:
            await asyncio.sleep(0.01)
            return state.update(context={**state.context, "async": True})

        graph.add_node("sync", sync_node)
        graph.add_node("async", async_node)
        graph.add_edge("sync", "async")
        graph.set_entry_point("sync")
        graph.set_finish_point("async")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="mixed_test"),
        )

        result = await orchestrator.run(AgentState())
        assert result.context.get("sync") is True
        assert result.context.get("async") is True

    @pytest.mark.asyncio
    async def test_different_state_shapes(self) -> None:
        """Test handling different state shapes."""
        graph = StateGraph(name="shapes_test")

        def transform_state(state: AgentState) -> AgentState:
            # Transform state shape
            new_context = {
                "transformed": True,
                "nested": {"level1": {"level2": "value"}},
                "list": [1, 2, 3],
            }
            return state.update(context=new_context)

        graph.add_node("transform", transform_state)
        graph.set_entry_point("transform")
        graph.set_finish_point("transform")

        orchestrator = AgentOrchestrator(
            graph=graph,
            config=OrchestratorConfig(name="shapes_test"),
        )

        result = await orchestrator.run(AgentState())
        assert result.context["nested"]["level1"]["level2"] == "value"
        assert result.context["list"] == [1, 2, 3]

    def test_config_defaults(self) -> None:
        """Test configuration defaults."""
        config = OrchestratorConfig(name="defaults_test")
        assert config.max_iterations == 100  # Default
        assert config.timeout == 300  # Default 5 minutes
        assert config.checkpoint_enabled is False  # Default
