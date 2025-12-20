"""Read-only tools for LLM-assisted workflows.

These helpers are designed to be safe to run inside agent loops:
- No file writes
- No code execution
- Best-effort external dependencies (e.g., vector DB) with graceful fallback
"""

from .read_only import (
    list_available_fields,
    search_similar_factors,
    summarize_dataframe,
)

__all__ = [
    "list_available_fields",
    "search_similar_factors",
    "summarize_dataframe",
]

