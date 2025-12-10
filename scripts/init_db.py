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

                # Convert trades table to hypertable
                await conn.execute(text(
                    "SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);"
                ))
                print("Trades hypertable created")

                # Add compression policy
                await conn.execute(text(
                    "SELECT add_compression_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);"
                ))
                print("Compression policy added")

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
