"""Data API router for IQFMP."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from iqfmp.db.database import get_db
from iqfmp.api.data.schemas import (
    DataStatusResponse,
    SymbolListResponse,
    AddSymbolRequest,
    AddSymbolResponse,
    DownloadTaskListResponse,
    DownloadTaskStatus,
    StartDownloadRequest,
    StartDownloadResponse,
    CancelDownloadResponse,
    OHLCVDataRequest,
    OHLCVDataResponse,
    DataRangeResponse,
    DataOptionsResponse,
)
from iqfmp.api.data.service import DataService

router = APIRouter(tags=["data"])


# ============== Dependency ==============

async def get_data_service():
    """Get data service with database session."""
    async for session in get_db():
        yield DataService(session)


# ============== Status ==============

@router.get("/status", response_model=DataStatusResponse)
async def get_data_status(
    service: DataService = Depends(get_data_service),
) -> DataStatusResponse:
    """Get data status overview including database connection and data statistics."""
    return await service.get_status()


@router.get("/options", response_model=DataOptionsResponse)
async def get_data_options(
    service: DataService = Depends(get_data_service),
) -> DataOptionsResponse:
    """Get available exchanges and timeframes."""
    return service.get_options()


# ============== Symbols ==============

@router.get("/symbols", response_model=SymbolListResponse)
async def list_symbols(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    service: DataService = Depends(get_data_service),
) -> SymbolListResponse:
    """List available trading symbols."""
    return await service.list_symbols(exchange=exchange, page=page, page_size=page_size)


@router.post("/symbols", response_model=AddSymbolResponse)
async def add_symbol(
    request: AddSymbolRequest,
    service: DataService = Depends(get_data_service),
) -> AddSymbolResponse:
    """Add a new trading symbol."""
    return await service.add_symbol(symbol=request.symbol, exchange=request.exchange)


@router.delete("/symbols/{symbol}", response_model=AddSymbolResponse)
async def remove_symbol(
    symbol: str,
    service: DataService = Depends(get_data_service),
) -> AddSymbolResponse:
    """Remove a trading symbol (mark as inactive)."""
    return await service.remove_symbol(symbol=symbol)


# ============== Download Tasks ==============

@router.get("/downloads", response_model=DownloadTaskListResponse)
async def list_downloads(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
    service: DataService = Depends(get_data_service),
) -> DownloadTaskListResponse:
    """List data download tasks."""
    return await service.list_download_tasks(status=status, limit=limit)


@router.post("/downloads", response_model=StartDownloadResponse)
async def start_download(
    request: StartDownloadRequest,
    service: DataService = Depends(get_data_service),
) -> StartDownloadResponse:
    """Start a data download task."""
    return await service.start_download(
        symbol=request.symbol,
        timeframe=request.timeframe,
        exchange=request.exchange,
        start_date=request.start_date,
        end_date=request.end_date,
    )


@router.get("/downloads/{task_id}", response_model=DownloadTaskStatus)
async def get_download_status(
    task_id: str,
    service: DataService = Depends(get_data_service),
) -> DownloadTaskStatus:
    """Get download task status."""
    task = await service.get_download_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.delete("/downloads/{task_id}", response_model=CancelDownloadResponse)
async def cancel_download(
    task_id: str,
    service: DataService = Depends(get_data_service),
) -> CancelDownloadResponse:
    """Cancel a download task."""
    return await service.cancel_download(task_id)


# ============== OHLCV Data ==============

@router.get("/ohlcv/{symbol}", response_model=OHLCVDataResponse)
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    start_date: datetime = Query(..., description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date (default: now)"),
    limit: int = Query(1000, ge=1, le=10000, description="Max bars to return"),
    service: DataService = Depends(get_data_service),
) -> OHLCVDataResponse:
    """Get OHLCV data for a symbol."""
    return await service.get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


@router.post("/ohlcv/query", response_model=OHLCVDataResponse)
async def query_ohlcv_data(
    request: OHLCVDataRequest,
    service: DataService = Depends(get_data_service),
) -> OHLCVDataResponse:
    """Query OHLCV data with POST body."""
    return await service.get_ohlcv_data(
        symbol=request.symbol,
        timeframe=request.timeframe,
        start_date=request.start_date,
        end_date=request.end_date,
        limit=request.limit,
    )


# ============== Data Ranges ==============

@router.get("/ranges", response_model=DataRangeResponse)
async def get_data_ranges(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    service: DataService = Depends(get_data_service),
) -> DataRangeResponse:
    """Get data availability ranges."""
    return await service.get_data_ranges(symbol=symbol, timeframe=timeframe)
