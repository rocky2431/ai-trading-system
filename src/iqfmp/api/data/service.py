"""Data API service for IQFMP."""

import uuid
from datetime import datetime, timezone
from typing import Optional
import asyncio
import httpx
import logging

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.models import OHLCVDataORM, DataDownloadTaskORM, SymbolInfoORM
from iqfmp.data.downloader import execute_download_task
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
    BinanceSymbolInfo,
    BinanceSymbolListResponse,
    DataTypeOption,
    MarketTypeOption,
    ExtendedDataOptionsResponse,
)

logger = logging.getLogger(__name__)


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

# Binance supported data types
# supported=True means download is implemented, False means planned but not yet implemented
DATA_TYPES = [
    {
        "id": "ohlcv",
        "name": "OHLCV Candlestick",
        "description": "Open, High, Low, Close, Volume",
        "requires_futures": False,
        "min_interval": "1m",
        "supported": True,  # Implemented via CCXT
    },
    {
        "id": "agg_trades",
        "name": "Aggregated Trades (Coming Soon)",
        "description": "[NOT IMPLEMENTED] Trade records aggregated by price. Planned for future release.",
        "requires_futures": False,
        "min_interval": "tick",
        "supported": False,
    },
    {
        "id": "trades",
        "name": "Tick Trades (Coming Soon)",
        "description": "[NOT IMPLEMENTED] Detailed record of each trade. Planned for future release.",
        "requires_futures": False,
        "min_interval": "tick",
        "supported": False,
    },
    {
        "id": "depth",
        "name": "Order Book Snapshot (Coming Soon)",
        "description": "[NOT IMPLEMENTED] Order book bid/ask snapshot. Planned for future release.",
        "requires_futures": False,
        "min_interval": "snapshot",
        "supported": False,
    },
    {
        "id": "funding_rate",
        "name": "Funding Rate",
        "description": "Perpetual contract funding rate history",
        "requires_futures": True,
        "min_interval": "8h",
        "supported": True,  # Implemented for futures via CCXT
    },
    {
        "id": "open_interest",
        "name": "Open Interest",
        "description": "Contract open interest quantity",
        "requires_futures": True,
        "min_interval": "5m",
        "supported": True,  # Implemented for futures via CCXT
    },
    {
        "id": "long_short_ratio",
        "name": "Long/Short Ratio",
        "description": "Large/retail trader long/short position ratio",
        "requires_futures": True,
        "min_interval": "5m",
        "supported": True,  # Implemented for Binance futures HTTP endpoint
    },
    {
        "id": "liquidation",
        "name": "Liquidation",
        "description": "Liquidation orders volume",
        "requires_futures": True,
        "min_interval": "1m",
        "supported": True,  # Implemented for Binance futures HTTP endpoint
    },
]

# Market types
MARKET_TYPES = [
    {
        "id": "spot",
        "name": "Spot",
        "description": "Spot market trading pairs, supports all basic data types",
    },
    {
        "id": "futures",
        "name": "USDT-M Futures",
        "description": "USDT-margined perpetual contracts, supports funding rate, open interest, etc.",
    },
]

# Stablecoins to exclude from top tokens
STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "FRAX",
    "LUSD", "USDD", "SUSD", "EURS", "UST", "USTC", "FDUSD", "PYUSD",
}


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
            size_value = result.scalar()
            db_status.total_size_mb = float(size_value) if size_value else 0.0

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
                data_type=getattr(t, 'data_type', 'ohlcv'),  # Backward compatible
                market_type=getattr(t, 'market_type', 'spot'),  # Backward compatible
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
        data_type: str = "ohlcv",
        market_type: str = "spot",
    ) -> StartDownloadResponse:
        """Start a data download task."""
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Validate timeframe (skip for data types that don't require it)
        valid_timeframes = [t["id"] for t in TIMEFRAMES]
        if data_type not in ["funding_rate", "liquidation", "long_short_ratio"]:
            if timeframe not in valid_timeframes:
                return StartDownloadResponse(
                    success=False,
                    message=f"Invalid timeframe: {timeframe}. Valid options: {valid_timeframes}",
                )

        # Validate data_type is supported
        supported_data_types = [d["id"] for d in DATA_TYPES if d.get("supported", False)]
        if data_type not in supported_data_types:
            return StartDownloadResponse(
                success=False,
                message=f"Data type '{data_type}' is not yet supported. Supported types: {supported_data_types}",
            )

        # Futures-only validation
        futures_only = {d["id"] for d in DATA_TYPES if d.get("requires_futures")}
        if data_type in futures_only and market_type != "futures":
            return StartDownloadResponse(
                success=False,
                message=f"Data type '{data_type}' requires futures market_type.",
            )

        # Create download task
        task_id = str(uuid.uuid4())
        task = DataDownloadTaskORM(
            id=task_id,
            symbol=symbol.upper(),
            timeframe=timeframe,
            exchange=exchange,
            data_type=data_type,
            market_type=market_type,
            start_date=start_date,
            end_date=end_date,
            status="pending",
            progress=0.0,
            rows_downloaded=0,
        )
        self.session.add(task)
        await self.session.commit()

        # Start background download task
        asyncio.create_task(self._run_download_task(task_id))

        return StartDownloadResponse(
            success=True,
            message=f"Download task started for {symbol} ({timeframe}, {data_type})",
            task_id=task_id,
        )

    async def _run_download_task(self, task_id: str):
        """Run download task in background with a new session."""
        from iqfmp.db.database import get_async_session

        try:
            async with get_async_session() as session:
                await execute_download_task(task_id, session)
        except Exception as e:
            logger.error(f"Background download task {task_id} failed: {e}")

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
            OHLCVDataORM.market_type,
            func.min(OHLCVDataORM.timestamp),
            func.max(OHLCVDataORM.timestamp),
            func.count(),
        ).group_by(OHLCVDataORM.symbol, OHLCVDataORM.timeframe, OHLCVDataORM.market_type)

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
                market_type=row[2] or "spot",
                data_type="ohlcv",
                start_date=row[3],
                end_date=row[4],
                total_rows=row[5],
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

    def get_extended_options(self) -> ExtendedDataOptionsResponse:
        """Get extended data options including data types and market types."""
        return ExtendedDataOptionsResponse(
            exchanges=[ExchangeOption(**e) for e in EXCHANGES],
            timeframes=[TimeframeOption(**t) for t in TIMEFRAMES],
            data_types=[DataTypeOption(**d) for d in DATA_TYPES],
            market_types=[MarketTypeOption(**m) for m in MARKET_TYPES],
        )

    # ==================== Binance Exchange Info ====================

    async def get_binance_symbols(
        self,
        quote_asset: str = "USDT",
        limit: int = 200,
        search: Optional[str] = None,
    ) -> BinanceSymbolListResponse:
        """Get top trading pairs from Binance by volume, excluding stablecoins.

        Args:
            quote_asset: Quote asset filter (default: USDT)
            limit: Max number of symbols to return (default: 200)
            search: Optional search term to filter by base asset

        Returns:
            List of top trading pairs sorted by 24h volume
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get exchange info for symbol details
                exchange_info_resp = await client.get(
                    "https://api.binance.com/api/v3/exchangeInfo"
                )
                exchange_info = exchange_info_resp.json()

                # Get 24h ticker for volume data
                ticker_resp = await client.get(
                    "https://api.binance.com/api/v3/ticker/24hr"
                )
                tickers = ticker_resp.json()

            # Build symbol info map
            symbol_info_map = {}
            for s in exchange_info.get("symbols", []):
                if s.get("status") == "TRADING" and s.get("quoteAsset") == quote_asset:
                    base_asset = s.get("baseAsset", "")
                    # Exclude stablecoins
                    if base_asset not in STABLECOINS:
                        symbol_info_map[s["symbol"]] = {
                            "symbol": s["symbol"],
                            "base_asset": base_asset,
                            "quote_asset": s.get("quoteAsset", ""),
                            "status": s.get("status", ""),
                        }

            # Get volume data and merge
            symbols_with_volume = []
            for ticker in tickers:
                symbol = ticker.get("symbol", "")
                if symbol in symbol_info_map:
                    info = symbol_info_map[symbol]
                    # Calculate USD volume (quoteVolume for USDT pairs)
                    volume_24h = float(ticker.get("quoteVolume", 0))
                    symbols_with_volume.append({
                        **info,
                        "volume_24h_usd": volume_24h,
                    })

            # Sort by volume descending
            symbols_with_volume.sort(key=lambda x: x["volume_24h_usd"], reverse=True)

            # Apply search filter if provided
            if search:
                search_upper = search.upper()
                symbols_with_volume = [
                    s for s in symbols_with_volume
                    if search_upper in s["base_asset"] or search_upper in s["symbol"]
                ]

            # Take top N and add rank
            top_symbols = []
            for i, s in enumerate(symbols_with_volume[:limit]):
                top_symbols.append(BinanceSymbolInfo(
                    symbol=s["symbol"],
                    base_asset=s["base_asset"],
                    quote_asset=s["quote_asset"],
                    status=s["status"],
                    rank=i + 1,
                    volume_24h_usd=s["volume_24h_usd"],
                ))

            return BinanceSymbolListResponse(
                symbols=top_symbols,
                total=len(top_symbols),
                updated_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Failed to fetch Binance symbols: {e}")
            return BinanceSymbolListResponse(
                symbols=[],
                total=0,
                updated_at=datetime.now(timezone.utc),
            )


# Dependency injection
async def get_data_service(session: AsyncSession) -> DataService:
    """Get data service instance."""
    return DataService(session)
