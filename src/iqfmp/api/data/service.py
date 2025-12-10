"""Data API service for IQFMP."""

import uuid
from datetime import datetime, timezone
from typing import Optional
import asyncio

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.models import OHLCVDataORM, DataDownloadTaskORM, SymbolInfoORM
from iqfmp.api.data.schemas import (
    DataStatusResponse,
    DatabaseStatus,
    DataOverview,
    SymbolInfo,
    SymbolListResponse,
    AddSymbolResponse,
    DownloadTaskStatus,
    DownloadTaskListResponse,
    StartDownloadResponse,
    CancelDownloadResponse,
    OHLCVBar,
    OHLCVDataResponse,
    DataRangeInfo,
    DataRangeResponse,
    DataOptionsResponse,
    ExchangeOption,
    TimeframeOption,
)


# Available exchanges
EXCHANGES = [
    {"id": "binance", "name": "Binance", "supported": True},
    {"id": "okx", "name": "OKX", "supported": True},
    {"id": "bybit", "name": "Bybit", "supported": True},
    {"id": "gate", "name": "Gate.io", "supported": True},
]

# Available timeframes
TIMEFRAMES = [
    {"id": "1m", "name": "1 Minute", "minutes": 1},
    {"id": "5m", "name": "5 Minutes", "minutes": 5},
    {"id": "15m", "name": "15 Minutes", "minutes": 15},
    {"id": "1h", "name": "1 Hour", "minutes": 60},
    {"id": "4h", "name": "4 Hours", "minutes": 240},
    {"id": "1d", "name": "1 Day", "minutes": 1440},
]


class DataService:
    """Service for data management operations."""

    def __init__(self, session: AsyncSession):
        """Initialize service with database session."""
        self.session = session
        self._download_tasks: dict[str, asyncio.Task] = {}

    # ==================== Status ====================

    async def get_status(self) -> DataStatusResponse:
        """Get data status overview."""
        # Check database connection
        db_status = DatabaseStatus(connected=False)
        try:
            result = await self.session.execute(text("SELECT version()"))
            version = result.scalar()
            db_status.connected = True
            db_status.version = version

            # Check if TimescaleDB is enabled
            try:
                await self.session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                )
                db_status.hypertables_enabled = True
            except Exception:
                pass

            # Get database size
            result = await self.session.execute(
                text("SELECT pg_database_size(current_database()) / (1024.0 * 1024.0)")
            )
            db_status.total_size_mb = result.scalar() or 0.0

        except Exception:
            pass

        # Get data overview
        overview = DataOverview()
        try:
            # Total symbols
            result = await self.session.execute(
                select(func.count()).select_from(SymbolInfoORM)
            )
            overview.total_symbols = result.scalar() or 0

            # Total rows
            result = await self.session.execute(
                select(func.count()).select_from(OHLCVDataORM)
            )
            overview.total_rows = result.scalar() or 0

            # Date range
            result = await self.session.execute(
                select(func.min(OHLCVDataORM.timestamp), func.max(OHLCVDataORM.timestamp))
            )
            row = result.one_or_none()
            if row:
                overview.oldest_data = row[0]
                overview.newest_data = row[1]

        except Exception:
            pass

        # Active downloads
        active_downloads = 0
        try:
            result = await self.session.execute(
                select(func.count())
                .select_from(DataDownloadTaskORM)
                .where(DataDownloadTaskORM.status.in_(["pending", "running"]))
            )
            active_downloads = result.scalar() or 0
        except Exception:
            pass

        return DataStatusResponse(
            database=db_status,
            overview=overview,
            active_downloads=active_downloads,
        )

    # ==================== Symbols ====================

    async def list_symbols(
        self,
        exchange: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> SymbolListResponse:
        """List available symbols."""
        query = select(SymbolInfoORM)

        if exchange:
            query = query.where(SymbolInfoORM.exchange == exchange)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        # Pagination
        offset = (page - 1) * page_size
        query = query.order_by(SymbolInfoORM.symbol).offset(offset).limit(page_size)

        result = await self.session.execute(query)
        symbols_orm = result.scalars().all()

        symbols = [
            SymbolInfo(
                symbol=s.symbol,
                exchange=s.exchange,
                base_asset=s.base_asset,
                quote_asset=s.quote_asset,
                is_active=s.is_active,
                has_1m=s.has_1m,
                has_5m=s.has_5m,
                has_15m=s.has_15m,
                has_1h=s.has_1h,
                has_4h=s.has_4h,
                has_1d=s.has_1d,
                data_start=s.data_start,
                data_end=s.data_end,
                total_rows=s.total_rows,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
            for s in symbols_orm
        ]

        return SymbolListResponse(symbols=symbols, total=total)

    async def add_symbol(self, symbol: str, exchange: str = "binance") -> AddSymbolResponse:
        """Add a new symbol."""
        # Parse symbol
        symbol = symbol.upper().replace("-", "/")
        if "/" not in symbol:
            # Try to guess format
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"
            elif symbol.endswith("BTC"):
                symbol = symbol[:-3] + "/BTC"

        parts = symbol.split("/")
        if len(parts) != 2:
            return AddSymbolResponse(
                success=False,
                message=f"Invalid symbol format: {symbol}. Expected format: BTC/USDT",
            )

        base_asset, quote_asset = parts

        # Check if already exists
        result = await self.session.execute(
            select(SymbolInfoORM).where(SymbolInfoORM.symbol == symbol)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return AddSymbolResponse(
                success=False,
                message=f"Symbol {symbol} already exists",
            )

        # Create new symbol
        new_symbol = SymbolInfoORM(
            symbol=symbol,
            exchange=exchange,
            base_asset=base_asset,
            quote_asset=quote_asset,
            is_active=True,
        )
        self.session.add(new_symbol)
        await self.session.commit()

        return AddSymbolResponse(
            success=True,
            message=f"Symbol {symbol} added successfully",
            symbol=SymbolInfo(
                symbol=symbol,
                exchange=exchange,
                base_asset=base_asset,
                quote_asset=quote_asset,
                is_active=True,
            ),
        )

    async def remove_symbol(self, symbol: str) -> AddSymbolResponse:
        """Remove a symbol (mark as inactive)."""
        result = await self.session.execute(
            select(SymbolInfoORM).where(SymbolInfoORM.symbol == symbol)
        )
        symbol_orm = result.scalar_one_or_none()

        if not symbol_orm:
            return AddSymbolResponse(
                success=False,
                message=f"Symbol {symbol} not found",
            )

        symbol_orm.is_active = False
        await self.session.commit()

        return AddSymbolResponse(
            success=True,
            message=f"Symbol {symbol} removed",
        )

    # ==================== Download Tasks ====================

    async def list_download_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> DownloadTaskListResponse:
        """List download tasks."""
        query = select(DataDownloadTaskORM)

        if status:
            query = query.where(DataDownloadTaskORM.status == status)

        query = query.order_by(DataDownloadTaskORM.created_at.desc()).limit(limit)

        result = await self.session.execute(query)
        tasks_orm = result.scalars().all()

        tasks = [
            DownloadTaskStatus(
                id=t.id,
                symbol=t.symbol,
                timeframe=t.timeframe,
                exchange=t.exchange,
                start_date=t.start_date,
                end_date=t.end_date,
                status=t.status,
                progress=t.progress,
                rows_downloaded=t.rows_downloaded,
                error_message=t.error_message,
                created_at=t.created_at,
                started_at=t.started_at,
                completed_at=t.completed_at,
            )
            for t in tasks_orm
        ]

        return DownloadTaskListResponse(tasks=tasks, total=len(tasks))

    async def start_download(
        self,
        symbol: str,
        timeframe: str,
        exchange: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> StartDownloadResponse:
        """Start a data download task."""
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Validate timeframe
        valid_timeframes = [t["id"] for t in TIMEFRAMES]
        if timeframe not in valid_timeframes:
            return StartDownloadResponse(
                success=False,
                message=f"Invalid timeframe: {timeframe}. Valid options: {valid_timeframes}",
            )

        # Create download task
        task_id = str(uuid.uuid4())
        task = DataDownloadTaskORM(
            id=task_id,
            symbol=symbol.upper(),
            timeframe=timeframe,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            status="pending",
            progress=0.0,
            rows_downloaded=0,
        )
        self.session.add(task)
        await self.session.commit()

        # Note: Actual download would be handled by a background worker
        # For now, we just create the task record

        return StartDownloadResponse(
            success=True,
            message=f"Download task created for {symbol} ({timeframe})",
            task_id=task_id,
        )

    async def get_download_task(self, task_id: str) -> Optional[DownloadTaskStatus]:
        """Get download task status."""
        result = await self.session.execute(
            select(DataDownloadTaskORM).where(DataDownloadTaskORM.id == task_id)
        )
        task = result.scalar_one_or_none()

        if not task:
            return None

        return DownloadTaskStatus(
            id=task.id,
            symbol=task.symbol,
            timeframe=task.timeframe,
            exchange=task.exchange,
            start_date=task.start_date,
            end_date=task.end_date,
            status=task.status,
            progress=task.progress,
            rows_downloaded=task.rows_downloaded,
            error_message=task.error_message,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
        )

    async def cancel_download(self, task_id: str) -> CancelDownloadResponse:
        """Cancel a download task."""
        result = await self.session.execute(
            select(DataDownloadTaskORM).where(DataDownloadTaskORM.id == task_id)
        )
        task = result.scalar_one_or_none()

        if not task:
            return CancelDownloadResponse(
                success=False,
                message=f"Task {task_id} not found",
            )

        if task.status not in ["pending", "running"]:
            return CancelDownloadResponse(
                success=False,
                message=f"Task cannot be cancelled (status: {task.status})",
            )

        task.status = "cancelled"
        await self.session.commit()

        return CancelDownloadResponse(
            success=True,
            message=f"Task {task_id} cancelled",
        )

    # ==================== OHLCV Data ====================

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> OHLCVDataResponse:
        """Get OHLCV data for a symbol."""
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        query = (
            select(OHLCVDataORM)
            .where(
                OHLCVDataORM.symbol == symbol.upper(),
                OHLCVDataORM.timeframe == timeframe,
                OHLCVDataORM.timestamp >= start_date,
                OHLCVDataORM.timestamp <= end_date,
            )
            .order_by(OHLCVDataORM.timestamp)
            .limit(limit)
        )

        result = await self.session.execute(query)
        data_orm = result.scalars().all()

        bars = [
            OHLCVBar(
                timestamp=d.timestamp,
                open=d.open,
                high=d.high,
                low=d.low,
                close=d.close,
                volume=d.volume,
            )
            for d in data_orm
        ]

        return OHLCVDataResponse(
            symbol=symbol.upper(),
            timeframe=timeframe,
            data=bars,
            total_rows=len(bars),
        )

    async def get_data_ranges(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> DataRangeResponse:
        """Get data availability ranges."""
        query = select(
            OHLCVDataORM.symbol,
            OHLCVDataORM.timeframe,
            func.min(OHLCVDataORM.timestamp),
            func.max(OHLCVDataORM.timestamp),
            func.count(),
        ).group_by(OHLCVDataORM.symbol, OHLCVDataORM.timeframe)

        if symbol:
            query = query.where(OHLCVDataORM.symbol == symbol.upper())
        if timeframe:
            query = query.where(OHLCVDataORM.timeframe == timeframe)

        result = await self.session.execute(query)
        rows = result.all()

        ranges = [
            DataRangeInfo(
                symbol=row[0],
                timeframe=row[1],
                start_date=row[2],
                end_date=row[3],
                total_rows=row[4],
            )
            for row in rows
        ]

        return DataRangeResponse(ranges=ranges)

    # ==================== Options ====================

    def get_options(self) -> DataOptionsResponse:
        """Get available data options."""
        return DataOptionsResponse(
            exchanges=[ExchangeOption(**e) for e in EXCHANGES],
            timeframes=[TimeframeOption(**t) for t in TIMEFRAMES],
        )


# Dependency injection
async def get_data_service(session: AsyncSession) -> DataService:
    """Get data service instance."""
    return DataService(session)
