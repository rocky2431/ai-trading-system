"""SQLAlchemy ORM models for TimescaleDB."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class FactorORM(Base):
    """Factor table - stores generated factors."""

    __tablename__ = "factors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    family: Mapped[list[str]] = mapped_column(ARRAY(String(50)), default=list)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_task: Mapped[str] = mapped_column(String(50), default="prediction")

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="candidate", index=True
    )  # candidate, rejected, core, redundant
    cluster_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Metrics (stored as JSONB for flexibility)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    stability: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Experiment tracking
    experiment_number: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("ix_factors_family", "family", postgresql_using="gin"),
        Index("ix_factors_created_at", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family or [],
            "code": self.code,
            "code_hash": self.code_hash,
            "target_task": self.target_task,
            "status": self.status,
            "cluster_id": self.cluster_id,
            "metrics": self.metrics,
            "stability": self.stability,
            "experiment_number": self.experiment_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ResearchTrialORM(Base):
    """Research trial table - tracks all factor experiments."""

    __tablename__ = "research_trials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    trial_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    factor_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("factors.id", ondelete="SET NULL"), nullable=True
    )
    factor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    factor_family: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")

    # Metrics
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    ic_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Dynamic threshold (DSR)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    passed_threshold: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    evaluation_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_research_trials_created_at", "created_at"),
        Index("ix_research_trials_factor_family", "factor_family"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for TrialRecord compatibility."""
        return {
            "trial_id": self.trial_id,
            "factor_name": self.factor_name,
            "factor_family": self.factor_family,
            "sharpe_ratio": self.sharpe_ratio,
            "ic_mean": self.ic_mean,
            "ir": self.ir,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.evaluation_config or {},
        }


class StrategyORM(Base):
    """Strategy table - stores trading strategies."""

    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Factor combination
    factor_ids: Mapped[list[str]] = mapped_column(ARRAY(String(36)), default=list)
    factor_weights: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Strategy code
    code: Mapped[str] = mapped_column(Text, nullable=False)

    # Configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="draft", index=True
    )  # draft, backtested, live, stopped

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class BacktestResultORM(Base):
    """Backtest result table - stores backtest results."""

    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("strategies.id", ondelete="CASCADE"), index=True
    )

    # Period
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Performance metrics
    total_return: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float] = mapped_column(Float, nullable=True)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=True)

    # Full results stored as JSONB
    full_results: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PipelineRunORM(Base):
    """Pipeline run table - tracks pipeline execution state."""

    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    pipeline_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # factor_mining, strategy_backtest, full_pipeline

    # Configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="pending", index=True
    )  # pending, running, completed, failed, cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 to 1.0
    current_step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Results
    result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Task IDs (for Celery integration)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_pipeline_runs_status_created", "status", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.id,
            "pipeline_type": self.pipeline_type,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "result": self.result,
            "error": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class RDLoopRunORM(Base):
    """RD Loop run table - stores RD Loop execution state for persistence."""

    __tablename__ = "rd_loop_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # Configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    data_source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        String(20), default="pending", index=True
    )  # pending, running, completed, failed, stopped
    phase: Mapped[str] = mapped_column(String(50), default="initialization")
    iteration: Mapped[int] = mapped_column(Integer, default=0)
    total_hypotheses_tested: Mapped[int] = mapped_column(Integer, default=0)
    core_factors_count: Mapped[int] = mapped_column(Integer, default=0)

    # Core factors (list of factor names/IDs)
    core_factors: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Statistics
    statistics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Iteration results
    iteration_results: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Error
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_rd_loop_runs_status_created", "status", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.id,
            "status": self.status,
            "phase": self.phase,
            "iteration": self.iteration,
            "total_hypotheses_tested": self.total_hypotheses_tested,
            "core_factors_count": self.core_factors_count,
            "core_factors": self.core_factors or [],
            "statistics": self.statistics,
            "iteration_results": self.iteration_results or [],
            "error": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class FactorValueORM(Base):
    """Factor value table - stores computed factor time-series (TimescaleDB hypertable)."""

    __tablename__ = "factor_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Factor value
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # Additional context (for multi-timeframe factors)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="1d")

    # Timestamp - hypertable partition column
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_factor_values_factor_symbol_ts", "factor_id", "symbol", "timestamp"),
        Index("ix_factor_values_ts", "timestamp"),
    )


class MiningTaskORM(Base):
    """Mining task table - stores factor mining task state."""

    __tablename__ = "mining_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration
    factor_families: Mapped[list[str]] = mapped_column(ARRAY(String(50)), default=list)
    target_count: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    auto_evaluate: Mapped[bool] = mapped_column(Boolean, default=True)

    # Progress
    generated_count: Mapped[int] = mapped_column(Integer, default=0)
    passed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Celery task ID for tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "factor_families": self.factor_families or [],
            "target_count": self.target_count,
            "auto_evaluate": self.auto_evaluate,
            "generated_count": self.generated_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "status": self.status,
            "progress": self.progress,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TradeORM(Base):
    """Trade table - stores trading records (TimescaleDB hypertable)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Order info
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # buy, sell
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # market, limit
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    commission: Mapped[float] = mapped_column(Float, default=0.0)

    # PnL
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=True)

    # Timestamp - this will be the time column for hypertable
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_trades_strategy_timestamp", "strategy_id", "timestamp"),
    )


class OHLCVDataORM(Base):
    """OHLCV data table - stores historical price data (TimescaleDB hypertable)."""

    __tablename__ = "ohlcv_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # 1m, 5m, 1h, 1d
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, default="binance", index=True)
    market_type: Mapped[str] = mapped_column(String(10), nullable=False, default="spot")  # spot, futures

    # OHLCV data
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamp - this will be the time column for hypertable
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_ohlcv_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
        Index("ix_ohlcv_exchange_symbol", "exchange", "symbol"),
    )


class DataDownloadTaskORM(Base):
    """Data download task table - tracks data download jobs."""

    __tablename__ = "data_download_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), nullable=False, default="ohlcv")  # ohlcv, agg_trades, etc.
    market_type: Mapped[str] = mapped_column(String(10), nullable=False, default="spot")  # spot, futures

    # Period
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Progress
    status: Mapped[str] = mapped_column(
        String(20), default="pending", index=True
    )  # pending, running, completed, failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    rows_downloaded: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class SymbolInfoORM(Base):
    """Symbol info table - stores available trading symbols."""

    __tablename__ = "symbol_info"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    base_asset: Mapped[str] = mapped_column(String(10), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(10), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Data availability
    has_1m: Mapped[bool] = mapped_column(Boolean, default=False)
    has_5m: Mapped[bool] = mapped_column(Boolean, default=False)
    has_15m: Mapped[bool] = mapped_column(Boolean, default=False)
    has_1h: Mapped[bool] = mapped_column(Boolean, default=False)
    has_4h: Mapped[bool] = mapped_column(Boolean, default=False)
    has_1d: Mapped[bool] = mapped_column(Boolean, default=False)

    # Data range
    data_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    data_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    total_rows: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AgentConfigORM(Base):
    """Agent configuration table - stores AI agent settings and prompts."""

    __tablename__ = "agent_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_type: Mapped[str] = mapped_column(
        String(30), nullable=False, unique=True, index=True
    )  # factor_generation, evaluation, strategy, backtest
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Prompts
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_prompt_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    examples: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration (JSON)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Expected config keys:
    # - security_check_enabled: bool
    # - field_constraint_enabled: bool
    # - max_retries: int
    # - timeout_seconds: float
    # - include_examples: bool
    # - temperature: float
    # - max_tokens: int

    # Status
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "examples": self.examples,
            "config": self.config or {},
            "is_enabled": self.is_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# SQL to create TimescaleDB hypertable (run after creating tables)
CREATE_HYPERTABLE_SQL = """
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert trades table to hypertable
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- Convert ohlcv_data table to hypertable
SELECT create_hypertable('ohlcv_data', 'timestamp', if_not_exists => TRUE);

-- Convert factor_values table to hypertable
SELECT create_hypertable('factor_values', 'timestamp', if_not_exists => TRUE);

-- Add compression policy (compress chunks older than 7 days)
SELECT add_compression_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('factor_values', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy (drop chunks older than 1 year)
SELECT add_retention_policy('trades', INTERVAL '1 year', if_not_exists => TRUE);
-- Keep OHLCV data for 3 years
SELECT add_retention_policy('ohlcv_data', INTERVAL '3 years', if_not_exists => TRUE);
-- Keep factor values for 2 years
SELECT add_retention_policy('factor_values', INTERVAL '2 years', if_not_exists => TRUE);
"""
