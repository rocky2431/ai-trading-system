"""Prompts API Router.

REST API endpoints for prompt template management and system configuration.
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from iqfmp.api.prompts.schemas import (
    PromptHistoryList,
    PromptTemplate,
    PromptTemplateList,
    SetSystemModeRequest,
    SystemModeConfigResponse,
)
from iqfmp.api.prompts.service import prompts_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prompts"])


class UpdatePromptRequest(BaseModel):
    """Request to update a prompt template."""

    system_prompt: str | None = None


class UpdatePromptResponse(BaseModel):
    """Response after updating a prompt."""

    success: bool
    message: str
    template: PromptTemplate | None = None


# ============== Prompt Templates ==============


@router.get("/templates", response_model=PromptTemplateList)
async def list_templates() -> PromptTemplateList:
    """List all prompt templates.

    Returns all agent prompt templates with current values
    (custom overrides or defaults).
    """
    return prompts_service.get_templates()


@router.get("/templates/{agent_id}", response_model=PromptTemplate)
async def get_template(agent_id: str) -> PromptTemplate:
    """Get a specific prompt template.

    Args:
        agent_id: Agent identifier (e.g., "factor_generation")

    Returns:
        The current prompt template for the agent
    """
    template = prompts_service.get_template(agent_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return template


@router.get("/templates/{agent_id}/default", response_model=PromptTemplate)
async def get_default_template(agent_id: str) -> PromptTemplate:
    """Get the default (built-in) prompt template.

    Args:
        agent_id: Agent identifier

    Returns:
        The default prompt template (ignoring any custom override)
    """
    template = prompts_service.get_default_template(agent_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return template


@router.put("/templates/{agent_id}", response_model=UpdatePromptResponse)
async def update_template(
    agent_id: str, request: UpdatePromptRequest
) -> UpdatePromptResponse:
    """Update a prompt template with custom override.

    Args:
        agent_id: Agent identifier
        request: Update request with new system_prompt

    Returns:
        Update result with the new template

    Notes:
        - Set system_prompt to null or empty string to reset to default
        - Changes are recorded in history
    """
    template = prompts_service.update_template(agent_id, request.system_prompt)
    if not template:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    action = "reset to default" if not request.system_prompt else "updated"
    return UpdatePromptResponse(
        success=True,
        message=f"Prompt template {action} for {agent_id}",
        template=template,
    )


@router.post("/templates/{agent_id}/reset", response_model=UpdatePromptResponse)
async def reset_template(agent_id: str) -> UpdatePromptResponse:
    """Reset a prompt template to default.

    Args:
        agent_id: Agent identifier

    Returns:
        The default template after reset
    """
    template = prompts_service.update_template(agent_id, None)
    if not template:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    return UpdatePromptResponse(
        success=True,
        message=f"Prompt template reset to default for {agent_id}",
        template=template,
    )


# ============== Prompt History ==============


@router.get("/history", response_model=PromptHistoryList)
async def get_history(
    agent_id: str | None = Query(None, description="Filter by agent ID"),
    limit: int = Query(50, ge=1, le=100),
) -> PromptHistoryList:
    """Get prompt change history.

    Args:
        agent_id: Optional filter by agent
        limit: Maximum entries to return

    Returns:
        List of prompt change history entries
    """
    return prompts_service.get_history(agent_id=agent_id, limit=limit)


# ============== System Mode Configuration ==============


@router.get("/system-mode", response_model=SystemModeConfigResponse)
async def get_system_mode() -> SystemModeConfigResponse:
    """Get system mode configuration.

    Returns current system mode settings including:
    - Strict mode (PostgreSQL requirement)
    - Sandbox settings (code execution limits)
    - Human review settings
    - Feature flags
    """
    return prompts_service.get_system_mode()


@router.patch("/system-mode", response_model=SystemModeConfigResponse)
async def update_system_mode(
    request: SetSystemModeRequest,
) -> SystemModeConfigResponse:
    """Update system mode configuration.

    Updates only the fields that are provided in the request.
    Other fields remain unchanged.
    """
    updates = request.model_dump(exclude_unset=True)
    return prompts_service.update_system_mode(updates)
