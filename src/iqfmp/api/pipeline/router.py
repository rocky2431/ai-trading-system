"""Pipeline API router with PostgreSQL persistence.

This module provides pipeline and RD Loop endpoints with full DB persistence.
Pipeline runs and RD Loop states survive server restarts.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.pipeline.schemas import (
    PipelineListResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatus,
    PipelineStatusResponse,
    RDLoopConfigRequest,
    RDLoopCoreFactorResponse,
    RDLoopIterationResult,
    RDLoopRunRequest,
    RDLoopRunResponse,
    RDLoopStateResponse,
    RDLoopStatisticsResponse,
)
from iqfmp.api.pipeline.service import PipelineNotFoundError, get_pipeline_service
from iqfmp.api.system.websocket import manager as ws_manager, WebSocketMessage
from iqfmp.db.database import get_db
from iqfmp.db.models import RDLoopRunORM

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pipeline"])

# =============================================================================
# RD Loop State Management (memory cache + DB persistence)
# =============================================================================

# Memory cache for running loops (hot state)
_rd_loop_instances: dict[str, Any] = {}
# Memory cache for results (also persisted to DB)
_rd_loop_results: dict[str, dict[str, Any]] = {}


@router.post("/run", response_model=PipelineRunResponse, status_code=status.HTTP_201_CREATED)
async def run_pipeline(
    request: PipelineRunRequest,
    session: AsyncSession = Depends(get_db),
) -> PipelineRunResponse:
    """Start a new pipeline run.

    Args:
        request: Pipeline run request

    Returns:
        Pipeline run response with run_id
    """
    service = get_pipeline_service()
    return await service.create_run_async(
        pipeline_type=request.pipeline_type,
        config=request.config,
        session=session,
    )


@router.get("/{run_id}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    run_id: str,
    session: AsyncSession = Depends(get_db),
) -> PipelineStatusResponse:
    """Get pipeline run status.

    Args:
        run_id: Pipeline run ID

    Returns:
        Pipeline status response

    Raises:
        HTTPException: If run not found
    """
    service = get_pipeline_service()
    status_resp = await service.get_run_status_async(run_id, session)

    if not status_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run {run_id} not found",
        )

    return status_resp


@router.get("/runs", response_model=PipelineListResponse)
async def list_pipeline_runs(
    status_filter: Optional[PipelineStatus] = Query(
        default=None, alias="status", description="Filter by status"
    ),
    session: AsyncSession = Depends(get_db),
) -> PipelineListResponse:
    """List pipeline runs.

    Args:
        status_filter: Filter by status

    Returns:
        List of pipeline runs
    """
    service = get_pipeline_service()
    runs = await service.list_runs_async(status=status_filter, session=session)

    return PipelineListResponse(
        runs=runs,
        total=len(runs),
    )


@router.post("/{run_id}/cancel")
async def cancel_pipeline(run_id: str) -> dict:
    """Cancel a pipeline run.

    Args:
        run_id: Pipeline run ID

    Returns:
        Cancellation result

    Raises:
        HTTPException: If run not found or cannot be cancelled
    """
    service = get_pipeline_service()

    try:
        cancelled = service.cancel_run(run_id)
    except PipelineNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run {run_id} not found",
        )

    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline run cannot be cancelled (already completed/failed)",
        )

    return {"message": "Pipeline run cancelled", "run_id": run_id}


@router.websocket("/{run_id}/ws")
async def pipeline_websocket(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for pipeline progress updates.

    Args:
        websocket: WebSocket connection
        run_id: Pipeline run ID
    """
    service = get_pipeline_service()

    # Check if run exists
    status_resp = service.get_run_status(run_id)
    if not status_resp:
        await websocket.close(code=4004, reason="Pipeline run not found")
        return

    await websocket.accept()

    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "run_id": status_resp.run_id,
                "status": status_resp.status,
                "progress": status_resp.progress,
                "current_step": status_resp.current_step,
            },
        })

        # Keep connection alive and send updates
        while True:
            # Wait for messages from client (ping/pong or commands)
            try:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "get_status":
                    current_status = service.get_run_status(run_id)
                    if current_status:
                        await websocket.send_json({
                            "type": "status",
                            "data": {
                                "run_id": current_status.run_id,
                                "status": current_status.status,
                                "progress": current_status.progress,
                                "current_step": current_status.current_step,
                            },
                        })
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass


# =============================================================================
# RD Loop Endpoints
# =============================================================================


async def _execute_rd_loop(run_id: str, config: RDLoopConfigRequest, data_source: Optional[str]) -> None:
    """Execute RD Loop in background with DB persistence and broadcast progress."""
    from iqfmp.agents.hypothesis_agent import HypothesisFamily
    from iqfmp.core.rd_loop import RDLoop, LoopConfig, LoopPhase
    from iqfmp.core.data_provider import DataProvider
    from iqfmp.db.database import get_async_session

    try:
        # Create loop config
        loop_config = LoopConfig(
            max_iterations=config.max_iterations,
            max_hypotheses_per_iteration=config.max_hypotheses_per_iteration,
            target_core_factors=config.target_core_factors,
            ic_threshold=config.ic_threshold,
            ir_threshold=config.ir_threshold,
            novelty_threshold=config.novelty_threshold,
            run_benchmark=config.run_benchmark,
            enable_combination=config.enable_combination,
        )

        # Create phase change callback with DB update
        def on_phase_change(phase: LoopPhase) -> None:
            async def update_phase():
                try:
                    async with get_async_session() as session:
                        await session.execute(
                            update(RDLoopRunORM)
                            .where(RDLoopRunORM.id == run_id)
                            .values(phase=phase.value)
                        )
                        await session.commit()
                except Exception:
                    pass  # Non-critical, continue

                await ws_manager.broadcast(WebSocketMessage(
                    type="rd_loop_phase",
                    data={"run_id": run_id, "phase": phase.value},
                ))

            asyncio.create_task(update_phase())

        # Create iteration callback with DB update
        def on_iteration_complete(iteration: int, result: dict) -> None:
            async def update_iteration():
                try:
                    async with get_async_session() as session:
                        # Get current run
                        db_result = await session.execute(
                            select(RDLoopRunORM).where(RDLoopRunORM.id == run_id)
                        )
                        db_run = db_result.scalar_one_or_none()
                        if db_run:
                            # Update iteration results
                            iteration_results = db_run.iteration_results or []
                            iteration_results.append(result)
                            await session.execute(
                                update(RDLoopRunORM)
                                .where(RDLoopRunORM.id == run_id)
                                .values(
                                    iteration=iteration,
                                    total_hypotheses_tested=result.get("total_tested", 0),
                                    iteration_results=iteration_results,
                                )
                            )
                            await session.commit()
                except Exception as e:
                    logger.warning(f"Failed to update iteration in DB: {e}")

                await ws_manager.broadcast(WebSocketMessage(
                    type="rd_loop_progress",
                    data={"run_id": run_id, "iteration": iteration, "result": result},
                ))

            asyncio.create_task(update_iteration())

        loop_config.on_phase_change = on_phase_change
        loop_config.on_iteration_complete = on_iteration_complete

        # Create RD Loop
        loop = RDLoop(config=loop_config)
        _rd_loop_instances[run_id] = loop

        # Update status to running in DB
        async with get_async_session() as session:
            await session.execute(
                update(RDLoopRunORM)
                .where(RDLoopRunORM.id == run_id)
                .values(status="running", started_at=datetime.now(timezone.utc))
            )
            await session.commit()

        # Load data - prioritize DB, fallback to CSV/synthetic
        df = None
        if data_source:
            loop.load_data(data_source)
        else:
            # Try loading from TimescaleDB first
            try:
                async with get_async_session() as session:
                    provider = DataProvider(session=session)
                    df = await provider.load_ohlcv(
                        symbol="ETH/USDT",
                        timeframe="1d",
                    )
                    logger.info(f"RD Loop: Loaded {len(df)} rows from DB")
            except Exception as e:
                logger.warning(f"RD Loop: DB data load failed: {e}, using fallback")
                df = None

            # Fallback to synthetic data if DB failed
            if df is None or len(df) < 100:
                import pandas as pd
                import numpy as np

                logger.info("RD Loop: Using synthetic data")
                np.random.seed(42)
                n_rows = 1000
                dates = pd.date_range("2023-01-01", periods=n_rows, freq="1h")
                close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
                high = close + np.abs(np.random.randn(n_rows) * 0.2)
                low = close - np.abs(np.random.randn(n_rows) * 0.2)
                open_price = low + (high - low) * np.random.rand(n_rows)
                volume = np.random.randint(1000, 10000, n_rows)

                df = pd.DataFrame({
                    "timestamp": dates,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                })

            loop.load_data(df)

        # Prepare focus families
        focus_families = None
        if config.focus_families:
            focus_families = [HypothesisFamily(f) for f in config.focus_families]

        # Run loop
        results = loop.run(focus_families=focus_families)

        # Store results in memory
        core_factors = loop.get_core_factors()
        statistics = loop.get_statistics()
        results_list = [r.to_dict() for r in results]

        _rd_loop_results[run_id] = {
            "status": "completed",
            "results": results_list,
            "core_factors": core_factors,
            "statistics": statistics,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Persist to DB
        async with get_async_session() as session:
            await session.execute(
                update(RDLoopRunORM)
                .where(RDLoopRunORM.id == run_id)
                .values(
                    status="completed",
                    completed_at=datetime.now(timezone.utc),
                    core_factors_count=len(core_factors),
                    core_factors=[f.get("name", "") for f in core_factors],
                    statistics=statistics,
                    iteration_results=results_list,
                )
            )
            await session.commit()

        # Broadcast completion
        await ws_manager.broadcast(WebSocketMessage(
            type="rd_loop_complete",
            data={
                "run_id": run_id,
                "status": "completed",
                "core_factors_count": len(core_factors),
            },
        ))

    except Exception as e:
        logger.error(f"RD Loop {run_id} failed: {e}")
        _rd_loop_results[run_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Persist failure to DB
        try:
            async with get_async_session() as session:
                await session.execute(
                    update(RDLoopRunORM)
                    .where(RDLoopRunORM.id == run_id)
                    .values(
                        status="failed",
                        completed_at=datetime.now(timezone.utc),
                        error_message=str(e),
                    )
                )
                await session.commit()
        except Exception:
            pass  # Already handling error

        await ws_manager.broadcast(WebSocketMessage(
            type="rd_loop_error",
            data={"run_id": run_id, "error": str(e)},
        ))

    finally:
        # Clean up instance but keep results
        if run_id in _rd_loop_instances:
            del _rd_loop_instances[run_id]


@router.post("/rd-loop/run", response_model=RDLoopRunResponse, status_code=status.HTTP_201_CREATED)
async def run_rd_loop(
    request: RDLoopRunRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> RDLoopRunResponse:
    """Start a new RD Loop run with DB persistence.

    The RD Loop executes the research-development cycle for factor mining:
    1. Hypothesis Generation
    2. Factor Coding
    3. Factor Evaluation
    4. Benchmark Comparison
    5. Factor Selection

    Progress updates are broadcast via WebSocket.
    State is persisted to PostgreSQL for recovery after restart.
    """
    run_id = str(uuid4())

    # Create DB record
    db_run = RDLoopRunORM(
        id=run_id,
        config=request.config.model_dump() if request.config else None,
        data_source=request.data_source,
        status="starting",
        phase="initialization",
    )
    session.add(db_run)
    await session.commit()

    # Also cache in memory
    _rd_loop_results[run_id] = {
        "status": "starting",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Start background execution
    background_tasks.add_task(
        _execute_rd_loop,
        run_id=run_id,
        config=request.config,
        data_source=request.data_source,
    )

    return RDLoopRunResponse(
        run_id=run_id,
        status="started",
        message="RD Loop started. Monitor progress via WebSocket.",
        created_at=datetime.now(timezone.utc),
    )


@router.get("/rd-loop/{run_id}/state", response_model=RDLoopStateResponse)
async def get_rd_loop_state(
    run_id: str,
    session: AsyncSession = Depends(get_db),
) -> RDLoopStateResponse:
    """Get current state of an RD Loop run from DB or memory cache."""
    # First check memory cache for running loop
    loop = _rd_loop_instances.get(run_id)

    if loop:
        # Running loop - use real-time state
        state = loop.state
        return RDLoopStateResponse(
            run_id=run_id,
            phase=state.phase.value,
            iteration=state.iteration,
            total_hypotheses_tested=state.total_hypotheses_tested,
            core_factors_count=len(state.core_factors),
            core_factors=state.core_factors[:20],  # Limit for response
            is_running=state.is_running,
            stop_requested=state.stop_requested,
        )

    # Check memory cache for completed/failed runs
    result = _rd_loop_results.get(run_id)
    if result:
        stats = result.get("statistics", {})
        state_dict = stats.get("state", {})
        return RDLoopStateResponse(
            run_id=run_id,
            phase=state_dict.get("phase", "completed"),
            iteration=state_dict.get("iteration", 0),
            total_hypotheses_tested=state_dict.get("total_hypotheses_tested", 0),
            core_factors_count=state_dict.get("core_factors_count", 0),
            core_factors=state_dict.get("core_factors", []),
            is_running=False,
            stop_requested=False,
        )

    # Fall back to DB lookup (for runs from previous server sessions)
    db_result = await session.execute(
        select(RDLoopRunORM).where(RDLoopRunORM.id == run_id)
    )
    db_run = db_result.scalar_one_or_none()

    if not db_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RD Loop run {run_id} not found",
        )

    return RDLoopStateResponse(
        run_id=run_id,
        phase=db_run.phase,
        iteration=db_run.iteration,
        total_hypotheses_tested=db_run.total_hypotheses_tested,
        core_factors_count=db_run.core_factors_count,
        core_factors=db_run.core_factors[:20] if db_run.core_factors else [],
        is_running=db_run.status == "running",
        stop_requested=False,
    )


@router.get("/rd-loop/{run_id}/statistics", response_model=RDLoopStatisticsResponse)
async def get_rd_loop_statistics(run_id: str) -> RDLoopStatisticsResponse:
    """Get detailed statistics for an RD Loop run."""
    loop = _rd_loop_instances.get(run_id)

    if loop:
        stats = loop.get_statistics()
        state = loop.state

        return RDLoopStatisticsResponse(
            run_id=run_id,
            state=RDLoopStateResponse(
                run_id=run_id,
                phase=state.phase.value,
                iteration=state.iteration,
                total_hypotheses_tested=state.total_hypotheses_tested,
                core_factors_count=len(state.core_factors),
                core_factors=state.core_factors[:20],
                is_running=state.is_running,
                stop_requested=state.stop_requested,
            ),
            hypothesis_stats=stats.get("hypothesis_stats", {}),
            ledger_stats=stats.get("ledger_stats", {}),
            benchmark_top_factors=stats.get("benchmark_top_factors", []),
            iteration_results=[
                RDLoopIterationResult(**r.to_dict())
                for r in state.iteration_results
            ],
        )

    # Check completed results
    result = _rd_loop_results.get(run_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RD Loop run {run_id} not found",
        )

    stats = result.get("statistics", {})
    state_dict = stats.get("state", {})

    return RDLoopStatisticsResponse(
        run_id=run_id,
        state=RDLoopStateResponse(
            run_id=run_id,
            phase=state_dict.get("phase", "completed"),
            iteration=state_dict.get("iteration", 0),
            total_hypotheses_tested=state_dict.get("total_hypotheses_tested", 0),
            core_factors_count=state_dict.get("core_factors_count", 0),
            core_factors=state_dict.get("core_factors", []),
            is_running=False,
            stop_requested=False,
        ),
        hypothesis_stats=stats.get("hypothesis_stats", {}),
        ledger_stats=stats.get("ledger_stats", {}),
        benchmark_top_factors=stats.get("benchmark_top_factors", []),
        iteration_results=[
            RDLoopIterationResult(**r) for r in result.get("results", [])
        ],
    )


@router.get("/rd-loop/{run_id}/factors", response_model=list[RDLoopCoreFactorResponse])
async def get_rd_loop_factors(run_id: str) -> list[RDLoopCoreFactorResponse]:
    """Get core factors discovered by an RD Loop run."""
    loop = _rd_loop_instances.get(run_id)

    if loop:
        factors = loop.get_core_factors()
        return [
            RDLoopCoreFactorResponse(
                name=f["name"],
                family=f["family"],
                code=f["code"],
                metrics=f["metrics"],
                hypothesis=f["hypothesis"],
            )
            for f in factors
        ]

    # Check completed results
    result = _rd_loop_results.get(run_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RD Loop run {run_id} not found",
        )

    factors = result.get("core_factors", [])
    return [
        RDLoopCoreFactorResponse(
            name=f["name"],
            family=f["family"],
            code=f["code"],
            metrics=f["metrics"],
            hypothesis=f["hypothesis"],
        )
        for f in factors
    ]


@router.post("/rd-loop/{run_id}/stop")
async def stop_rd_loop(run_id: str) -> dict:
    """Request to stop a running RD Loop."""
    loop = _rd_loop_instances.get(run_id)

    if not loop:
        # Check if it exists in results (already completed)
        if run_id in _rd_loop_results:
            return {"message": "RD Loop already completed", "run_id": run_id}

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RD Loop run {run_id} not found",
        )

    loop.stop()
    return {"message": "Stop requested", "run_id": run_id}


@router.get("/rdloop/state")
async def get_current_rd_loop_state() -> dict:
    """Get current RD Loop state (for monitoring dashboard).

    Returns state of any running RD Loop, or idle status if none running.
    """
    # Check for any running loop
    for run_id, loop in _rd_loop_instances.items():
        state = loop.state
        return {
            "run_id": run_id,
            "is_running": True,
            "phase": state.phase.value,
            "iteration": state.iteration,
            "total_hypotheses_tested": state.total_hypotheses_tested,
            "core_factors_count": len(state.core_factors),
        }

    # No running loop - return idle state
    return {
        "run_id": None,
        "is_running": False,
        "phase": "idle",
        "iteration": 0,
        "total_hypotheses_tested": 0,
        "core_factors_count": 0,
    }


@router.get("/rd-loop/runs")
async def list_rd_loop_runs(
    session: AsyncSession = Depends(get_db),
) -> dict:
    """List all RD Loop runs from DB (including historical runs).

    Combines running loops from memory cache with persisted runs from DB.
    """
    runs = []
    seen_ids = set()

    # Add running loops from memory cache (highest priority - real-time state)
    for run_id, loop in _rd_loop_instances.items():
        runs.append({
            "run_id": run_id,
            "status": "running",
            "phase": loop.state.phase.value,
            "iteration": loop.state.iteration,
            "core_factors_count": len(loop.state.core_factors),
        })
        seen_ids.add(run_id)

    # Add completed/failed results from memory cache
    for run_id, result in _rd_loop_results.items():
        if run_id not in seen_ids:
            stats = result.get("statistics", {})
            state = stats.get("state", {})
            runs.append({
                "run_id": run_id,
                "status": result.get("status", "unknown"),
                "phase": state.get("phase", "completed"),
                "iteration": state.get("iteration", 0),
                "core_factors_count": state.get("core_factors_count", 0),
                "error": result.get("error"),
            })
            seen_ids.add(run_id)

    # Fetch historical runs from DB (not in memory cache)
    db_result = await session.execute(
        select(RDLoopRunORM).order_by(RDLoopRunORM.created_at.desc()).limit(100)
    )
    db_runs = db_result.scalars().all()

    for db_run in db_runs:
        if db_run.id not in seen_ids:
            runs.append({
                "run_id": db_run.id,
                "status": db_run.status,
                "phase": db_run.phase,
                "iteration": db_run.iteration,
                "core_factors_count": db_run.core_factors_count,
                "error": db_run.error_message,
                "created_at": db_run.created_at.isoformat() if db_run.created_at else None,
            })

    return {"runs": runs, "total": len(runs)}
