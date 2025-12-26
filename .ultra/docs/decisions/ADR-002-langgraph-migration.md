# ADR-002: LangGraph Official Library Migration

## Status
**PROPOSED** - Pending team review and approval

## Date
2025-12-26

## Context

The current `src/iqfmp/agents/orchestrator.py` implements a custom StateGraph pattern inspired by LangGraph concepts but does not use the official LangGraph library.

### Current Implementation
- Custom `StateGraph` class (~350 lines)
- Custom `AgentState` dataclass
- Custom `Node`, `Edge`, `ConditionalEdge` abstractions
- Custom `PostgresCheckpointSaver` for state persistence
- No official LangGraph dependencies

### Deviations from LangGraph
1. **State management**: Custom immutable state pattern vs LangGraph's `TypedDict` approach
2. **Checkpointing**: Custom PostgreSQL adapter vs `langgraph-checkpoint-postgres`
3. **Conditional routing**: Custom router pattern vs LangGraph's `add_conditional_edges`
4. **Human-in-the-loop**: Not implemented vs LangGraph's interrupt/resume

### Why This Matters
1. **Ecosystem compatibility**: Cannot use LangGraph extensions (human-in-the-loop, time-travel)
2. **Maintenance burden**: Custom code requires ongoing maintenance
3. **Community patterns**: Missing LangGraph best practices for agent patterns
4. **Testing**: Cannot leverage LangGraph testing utilities

## Decision

**MIGRATE** to official LangGraph library in phases:

### Phase 1: Core Migration (1-2 weeks)
- Install `langgraph`, `langgraph-checkpoint-postgres`
- Replace custom `StateGraph` with `langgraph.graph.StateGraph`
- Migrate `AgentState` to `TypedDict` pattern
- Preserve existing node functions (minimal changes)

### Phase 2: Checkpoint Migration (3-5 days)
- Replace custom `PostgresCheckpointSaver` with official adapter
- Validate state persistence compatibility
- Add migration script for existing checkpoints

### Phase 3: Advanced Features (1 week)
- Implement human-in-the-loop via `interrupt_before`
- Add time-travel debugging support
- Implement proper subgraph patterns for nested agents

### Files to Modify
| File | Changes |
|------|---------|
| `src/iqfmp/agents/orchestrator.py` | Replace with LangGraph imports |
| `src/iqfmp/agents/pipeline_builder.py` | Update graph construction |
| `src/iqfmp/agents/nodes/*.py` | Adapt node signatures if needed |
| `requirements.txt` | Add `langgraph>=0.2.0` |
| Tests | Update to use LangGraph test patterns |

## Consequences

### Positive
- Full LangGraph ecosystem compatibility
- Reduced maintenance (official library)
- Human-in-the-loop capability for review gates
- Better debugging with time-travel
- Community support and documentation

### Negative
- Migration effort required
- Potential breaking changes in node signatures
- Need to retest all agent workflows
- Learning curve for LangGraph patterns

### Risks
- Existing checkpoints may need migration
- Performance characteristics may differ
- Some custom features may not map directly

## Migration Code Example

```python
# Before (current custom implementation)
from iqfmp.agents.orchestrator import StateGraph, AgentState

graph = StateGraph("factor_pipeline")
graph.add_node("generate", generate_factor)
graph.add_edge("generate", "evaluate")
graph.set_entry_point("generate")

# After (official LangGraph)
from langgraph.graph import StateGraph
from typing import TypedDict

class FactorState(TypedDict):
    messages: list[dict]
    context: dict
    current_node: str | None

graph = StateGraph(FactorState)
graph.add_node("generate", generate_factor)
graph.add_edge("generate", "evaluate")
graph.set_entry_point("generate")
app = graph.compile()
```

## References
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Checkpoint Postgres](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres)
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
