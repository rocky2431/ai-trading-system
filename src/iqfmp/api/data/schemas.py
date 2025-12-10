"""Data API schemas for IQFMP."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============== Database Status ==============

class DatabaseStatus(BaseModel):
    """Database connection status."""
    connected: bool = False
    version: Optional[str] = None
    hypertables_enabled: bool = False
    total_size_mb: float = 0.0


class DataOverview(BaseModel):
    """Data overview statistics."""
    total_symbols: int = 0
    total_rows: int = 0
    data_size_mb: float = 0.0
    oldest_data: Optional[datetime] = None
    newest_data: Optional[datetime] = None


class DataStatusResponse(BaseModel):
    """Data status response."""
    database: DatabaseStatus = Field(default_factory=DatabaseStatus)
    overview: DataOverview = Field(default_factory=DataOverview)
    active_downloads: int = 0


# ============== Symbol Management ==============

class SymbolInfo(BaseModel):
    """Symbol information."""
    symbol: str
    exchange: str
    base_asset: str
    quote_asset: str
    is_active: bool = True

    # Data availability by timeframe
    has_1m: bool = False
    has_5m: bool = False
    has_15m: bool = False
    has_1h: bool = False
    has_4h: bool = False
    has_1d: bool = False

    # Data range
    data_start: Optional[datetime] = None
    data_end: Optional[datetime] = None
    total_rows: int = 0

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SymbolListResponse(BaseModel):
    """Symbol list response."""
    symbols: list[SymbolInfo] = Field(default_factory=list)
    total: int = 0


class AddSymbolRequest(BaseModel):
    """Add symbol request."""
    symbol: str
    exchange: str = "binance"


class AddSymbolResponse(BaseModel):
    """Add symbol response."""
    success: bool
    message: str
    symbol: Optional[SymbolInfo] = None


# ============== Download Tasks ==============

class DownloadTaskStatus(BaseModel):
    """Download task status."""
    id: str
    symbol: str
    timeframe: str
    exchange: str
    start_date: datetime
    end_date: datetime
    status: str  # pending, running, completed, failed
    progress: float = 0.0  # 0-100
    rows_downloaded: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DownloadTaskListResponse(BaseModel):
    """Download task list response."""
    tasks: list[DownloadTaskStatus] = Field(default_factory=list)
    total: int = 0


class StartDownloadRequest(BaseModel):
    """Start download request."""
    symbol: str
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    exchange: str = "binance"
    start_date: datetime
    end_date: Optional[datetime] = None  # Defaults to now


class StartDownloadResponse(BaseModel):
    """Start download response."""
    success: bool
    message: str
    task_id: Optional[str] = None


class CancelDownloadResponse(BaseModel):
    """Cancel download response."""
    success: bool
    message: str


# ============== OHLCV Data Query ==============

class OHLCVBar(BaseModel):
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVDataRequest(BaseModel):
    """OHLCV data query request."""
    symbol: str
    timeframe: str = "1h"
    start_date: datetime
    end_date: Optional[datetime] = None
    limit: int = Field(default=1000, le=10000)


class OHLCVDataResponse(BaseModel):
    """OHLCV data response."""
    symbol: str
    timeframe: str
    data: list[OHLCVBar] = Field(default_factory=list)
    total_rows: int = 0


# ============== Data Range Query ==============

class DataRangeInfo(BaseModel):
    """Data range info for a symbol."""
    symbol: str
    timeframe: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_rows: int = 0
    gaps: list[tuple[datetime, datetime]] = Field(default_factory=list)


class DataRangeResponse(BaseModel):
    """Data range response."""
    ranges: list[DataRangeInfo] = Field(default_factory=list)


# ============== Available Exchanges & Timeframes ==============

class ExchangeOption(BaseModel):
    """Exchange option."""
    id: str
    name: str
    supported: bool = True


class TimeframeOption(BaseModel):
    """Timeframe option."""
    id: str
    name: str
    minutes: int


class DataOptionsResponse(BaseModel):
    """Data options response."""
    exchanges: list[ExchangeOption] = Field(default_factory=list)
    timeframes: list[TimeframeOption] = Field(default_factory=list)
