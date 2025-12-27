"""Prompts API module.

Provides REST endpoints for viewing and managing LLM prompt templates.
"""

from iqfmp.api.prompts.router import router

__all__ = ["router"]
