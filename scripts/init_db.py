#!/usr/bin/env python3
"""Initialize database schema for IQFMP."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from iqfmp.db.database import get_settings
from iqfmp.db.models import Base, CREATE_HYPERTABLE_SQL


async def init_database():
    """Create all tables and TimescaleDB hypertables."""
    settings = get_settings()

    print(f"Connecting to database: {settings.PGHOST}:{settings.PGPORT}/{settings.PGDATABASE}")

    engine = create_async_engine(settings.database_url, echo=True)

    try:
        async with engine.begin() as conn:
            # Create all tables
            print("\n=== Creating tables ===")
            await conn.run_sync(Base.metadata.create_all)
            print("Tables created successfully!")

            # Create TimescaleDB extension and hypertable
            print("\n=== Setting up TimescaleDB ===")
            try:
                # Enable TimescaleDB extension
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                print("TimescaleDB extension enabled")

                # ========================================
                # 1. trades table (交易记录)
                # ========================================
                await conn.execute(text(
                    "SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);"
                ))
                print("trades hypertable created")

                await conn.execute(text(
                    "SELECT add_compression_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);"
                ))
                print("trades compression policy added")

                # ========================================
                # 2. factor_values table (因子值时序数据)
                # ========================================
                await conn.execute(text(
                    "SELECT create_hypertable('factor_values', 'timestamp', if_not_exists => TRUE);"
                ))
                print("factor_values hypertable created")

                # Add compression policy (compress data older than 30 days)
                await conn.execute(text(
                    "SELECT add_compression_policy('factor_values', INTERVAL '30 days', if_not_exists => TRUE);"
                ))
                print("factor_values compression policy added")

                # Create index for factor_id + symbol queries
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_factor_values_factor_symbol
                    ON factor_values (factor_id, symbol, timestamp DESC);
                """))
                print("factor_values index created")

                # ========================================
                # 3. ohlcv_data table (K线数据)
                # ========================================
                await conn.execute(text(
                    "SELECT create_hypertable('ohlcv_data', 'timestamp', if_not_exists => TRUE);"
                ))
                print("ohlcv_data hypertable created")

                # Add compression policy (compress data older than 90 days)
                await conn.execute(text(
                    "SELECT add_compression_policy('ohlcv_data', INTERVAL '90 days', if_not_exists => TRUE);"
                ))
                print("ohlcv_data compression policy added")

                # Create index for symbol + timeframe queries
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe
                    ON ohlcv_data (symbol, timeframe, timestamp DESC);
                """))
                print("ohlcv_data index created")

                # ========================================
                # 4. Create indexes for other tables
                # ========================================
                # mining_tasks index
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_mining_tasks_status
                    ON mining_tasks (status, created_at DESC);
                """))
                print("mining_tasks index created")

                # pipeline_runs index
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
                    ON pipeline_runs (status, created_at DESC);
                """))
                print("pipeline_runs index created")

                # factors index
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_factors_family_status
                    ON factors (family, status, created_at DESC);
                """))
                print("factors index created")

                # research_trials index
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_research_trials_created
                    ON research_trials (created_at DESC);
                """))
                print("research_trials index created")

                # rd_loop_runs index
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_rd_loop_runs_status
                    ON rd_loop_runs (status, created_at DESC);
                """))
                print("rd_loop_runs index created")

            except Exception as e:
                print(f"Warning: TimescaleDB setup failed (may not be installed): {e}")
                print("Continuing without TimescaleDB features...")

            print("\n=== Database initialization complete! ===")

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
    finally:
        await engine.dispose()


async def drop_all_tables():
    """Drop all tables (for development/testing)."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)

    try:
        async with engine.begin() as conn:
            print("Dropping all tables...")
            await conn.run_sync(Base.metadata.drop_all)
            print("All tables dropped!")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize IQFMP database")
    parser.add_argument("--drop", action="store_true", help="Drop all tables first")
    args = parser.parse_args()

    if args.drop:
        print("WARNING: This will drop all tables!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() == "yes":
            asyncio.run(drop_all_tables())
        else:
            print("Aborted.")
            sys.exit(0)

    asyncio.run(init_database())
