#!/usr/bin/env python3
"""Sync TimescaleDB data to Qlib binary format.

This script synchronizes OHLCV data from the database to Qlib binary format
for use with D.features() API.

Usage:
    # Sync all symbols
    python scripts/sync_db_to_qlib.py --sync

    # Sync specific symbol
    python scripts/sync_db_to_qlib.py --sync --symbol ETHUSDT

    # Check sync status
    python scripts/sync_db_to_qlib.py --status

    # Force full resync
    python scripts/sync_db_to_qlib.py --sync --force
"""

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Database connection
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://iqfmp:iqfmp@localhost:5433/iqfmp"
)

# Default output directory
DEFAULT_QLIB_DATA_DIR = Path(__file__).parent.parent / "data" / "qlib_data"


async def get_db_session() -> AsyncSession:
    """Create async database session."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    return async_session()


async def fetch_ohlcv_data(
    session: AsyncSession,
    symbol: Optional[str] = None,
    timeframe: str = "1d",
    market_type: str = "futures",
) -> pd.DataFrame:
    """Fetch OHLCV data from database.

    Args:
        session: Database session
        symbol: Optional symbol filter (e.g., 'ETHUSDT')
        timeframe: Timeframe (e.g., '1d', '1h')
        market_type: Market type ('spot', 'futures')

    Returns:
        DataFrame with OHLCV data
    """
    query = text("""
        SELECT
            symbol,
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM ohlcv_data
        WHERE timeframe = :timeframe
          AND market_type = :market_type
          {symbol_filter}
        ORDER BY symbol, timestamp
    """.format(
        symbol_filter="AND symbol = :symbol" if symbol else ""
    ))

    params = {"timeframe": timeframe, "market_type": market_type}
    if symbol:
        params["symbol"] = symbol

    result = await session.execute(query, params)
    rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


async def fetch_funding_rates(
    session: AsyncSession,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch funding rate data from database.

    Args:
        session: Database session
        symbol: Optional symbol filter

    Returns:
        DataFrame with funding rate data
    """
    query = text("""
        SELECT
            symbol,
            timestamp,
            funding_rate
        FROM funding_rates
        {symbol_filter}
        ORDER BY symbol, timestamp
    """.format(
        symbol_filter="WHERE symbol = :symbol" if symbol else ""
    ))

    params = {}
    if symbol:
        params["symbol"] = symbol

    result = await session.execute(query, params)
    rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


async def fetch_open_interest(
    session: AsyncSession,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch open interest data from database.

    Args:
        session: Database session
        symbol: Optional symbol filter

    Returns:
        DataFrame with open interest data
    """
    query = text("""
        SELECT
            symbol,
            timestamp,
            open_interest
        FROM open_interest
        {symbol_filter}
        ORDER BY symbol, timestamp
    """.format(
        symbol_filter="WHERE symbol = :symbol" if symbol else ""
    ))

    params = {}
    if symbol:
        params["symbol"] = symbol

    result = await session.execute(query, params)
    rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def convert_to_qlib_binary(
    df: pd.DataFrame,
    output_dir: Path,
    symbol: str,
    include_derivatives: bool = True,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Convert DataFrame to Qlib binary format.

    Args:
        df: OHLCV DataFrame
        output_dir: Output directory for Qlib data
        symbol: Symbol name
        include_derivatives: Include funding_rate and open_interest
        funding_df: Funding rate DataFrame
        oi_df: Open interest DataFrame

    Returns:
        Dict with conversion stats
    """
    # Normalize symbol name for Qlib (lowercase, no underscore)
    instrument = symbol.lower().replace("_", "")

    # Create directories
    features_dir = output_dir / "features" / instrument
    features_dir.mkdir(parents=True, exist_ok=True)

    calendars_dir = output_dir / "calendars"
    calendars_dir.mkdir(exist_ok=True)

    instruments_dir = output_dir / "instruments"
    instruments_dir.mkdir(exist_ok=True)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove timezone info
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Qlib uses calendar position index (0, 1, 2, ...)
    start_index = 0

    # Field mapping (no $ prefix in file names - Qlib strips $ from expressions)
    field_mapping = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    stats = {"fields": [], "rows": len(df)}

    for csv_col, qlib_field in field_mapping.items():
        if csv_col not in df.columns:
            continue

        values = df[csv_col].values.astype(np.float32)
        bin_path = features_dir / f"{qlib_field}.day.bin"

        with open(bin_path, "wb") as f:
            np.array([start_index], dtype="<f").tofile(f)
            values.astype("<f").tofile(f)

        stats["fields"].append(qlib_field)

    # Add derivative data if available
    if include_derivatives:
        if funding_df is not None and len(funding_df) > 0:
            # Merge funding rate with OHLCV timestamps
            funding_merged = pd.merge_asof(
                df[["timestamp"]].sort_values("timestamp"),
                funding_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            if "funding_rate" in funding_merged.columns:
                values = funding_merged["funding_rate"].fillna(0).values.astype(np.float32)
                bin_path = features_dir / "funding_rate.day.bin"
                with open(bin_path, "wb") as f:
                    np.array([start_index], dtype="<f").tofile(f)
                    values.astype("<f").tofile(f)
                stats["fields"].append("funding_rate")

        if oi_df is not None and len(oi_df) > 0:
            # Merge open interest with OHLCV timestamps
            oi_merged = pd.merge_asof(
                df[["timestamp"]].sort_values("timestamp"),
                oi_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            if "open_interest" in oi_merged.columns:
                values = oi_merged["open_interest"].fillna(0).values.astype(np.float32)
                bin_path = features_dir / "open_interest.day.bin"
                with open(bin_path, "wb") as f:
                    np.array([start_index], dtype="<f").tofile(f)
                    values.astype("<f").tofile(f)
                stats["fields"].append("open_interest")

    # Create/update calendar file
    calendar_file = calendars_dir / "day.txt"
    calendar_dates = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()

    # Read existing calendar and merge
    existing_dates = set()
    if calendar_file.exists():
        with open(calendar_file, "r") as f:
            existing_dates = set(line.strip() for line in f if line.strip())

    all_dates = sorted(existing_dates | set(calendar_dates))
    with open(calendar_file, "w") as f:
        f.write("\n".join(all_dates))

    stats["calendar_days"] = len(all_dates)

    # Create/update instruments file
    instruments_file = instruments_dir / "all.txt"
    start_str = df["timestamp"].min().strftime("%Y-%m-%d")
    end_str = df["timestamp"].max().strftime("%Y-%m-%d")

    # Read existing instruments and update
    instruments = {}
    if instruments_file.exists():
        with open(instruments_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    instruments[parts[0]] = (parts[1], parts[2])

    instruments[instrument] = (start_str, end_str)

    with open(instruments_file, "w") as f:
        for inst, (start, end) in sorted(instruments.items()):
            f.write(f"{inst}\t{start}\t{end}\n")

    stats["instrument"] = instrument
    stats["start_date"] = start_str
    stats["end_date"] = end_str

    return stats


async def sync_symbol(
    session: AsyncSession,
    symbol: str,
    output_dir: Path,
    timeframe: str = "1d",
    market_type: str = "futures",
) -> dict:
    """Sync single symbol from database to Qlib binary.

    Args:
        session: Database session
        symbol: Symbol to sync
        output_dir: Output directory
        timeframe: Timeframe
        market_type: Market type

    Returns:
        Sync stats
    """
    print(f"Syncing {symbol}...")

    # Fetch OHLCV data
    ohlcv_df = await fetch_ohlcv_data(session, symbol, timeframe, market_type)
    if len(ohlcv_df) == 0:
        print(f"  ⚠️  No OHLCV data found for {symbol}")
        return {"symbol": symbol, "status": "no_data"}

    # Fetch derivative data
    funding_df = await fetch_funding_rates(session, symbol)
    oi_df = await fetch_open_interest(session, symbol)

    # Convert to Qlib binary
    stats = convert_to_qlib_binary(
        df=ohlcv_df,
        output_dir=output_dir,
        symbol=symbol,
        include_derivatives=True,
        funding_df=funding_df,
        oi_df=oi_df,
    )

    print(f"  ✅ {symbol}: {stats['rows']} rows, fields: {stats['fields']}")
    return {"symbol": symbol, "status": "synced", **stats}


async def sync_all_symbols(
    output_dir: Path,
    timeframe: str = "1d",
    market_type: str = "futures",
    force: bool = False,
) -> list:
    """Sync all symbols from database to Qlib binary.

    Args:
        output_dir: Output directory
        timeframe: Timeframe
        market_type: Market type
        force: Force full resync

    Returns:
        List of sync stats
    """
    session = await get_db_session()

    try:
        # Get list of symbols
        query = text("""
            SELECT DISTINCT symbol
            FROM ohlcv_data
            WHERE timeframe = :timeframe
              AND market_type = :market_type
            ORDER BY symbol
        """)
        result = await session.execute(query, {"timeframe": timeframe, "market_type": market_type})
        symbols = [row[0] for row in result.fetchall()]

        if not symbols:
            print("No symbols found in database")
            return []

        print(f"Found {len(symbols)} symbols to sync")
        print("=" * 60)

        results = []
        for symbol in symbols:
            stats = await sync_symbol(session, symbol, output_dir, timeframe, market_type)
            results.append(stats)

        print("=" * 60)
        synced = sum(1 for r in results if r.get("status") == "synced")
        print(f"Sync complete: {synced}/{len(symbols)} symbols synced")

        return results

    finally:
        await session.close()


async def show_status(output_dir: Path):
    """Show current sync status.

    Args:
        output_dir: Qlib data directory
    """
    print("=" * 60)
    print("Qlib Data Sync Status")
    print("=" * 60)

    if not output_dir.exists():
        print(f"❌ Data directory not found: {output_dir}")
        return

    # Check calendar
    calendar_file = output_dir / "calendars" / "day.txt"
    if calendar_file.exists():
        with open(calendar_file, "r") as f:
            dates = [line.strip() for line in f if line.strip()]
        print(f"📅 Calendar: {len(dates)} days")
        if dates:
            print(f"   Range: {dates[0]} ~ {dates[-1]}")
    else:
        print("📅 Calendar: Not found")

    # Check instruments
    instruments_file = output_dir / "instruments" / "all.txt"
    if instruments_file.exists():
        with open(instruments_file, "r") as f:
            instruments = [line.strip() for line in f if line.strip()]
        print(f"📊 Instruments: {len(instruments)}")
        for inst in instruments:
            print(f"   - {inst}")
    else:
        print("📊 Instruments: Not found")

    # Check features
    features_dir = output_dir / "features"
    if features_dir.exists():
        print(f"📦 Features:")
        for inst_dir in sorted(features_dir.iterdir()):
            if inst_dir.is_dir():
                bins = list(inst_dir.glob("*.bin"))
                print(f"   {inst_dir.name}: {len(bins)} fields")
                for b in sorted(bins):
                    size = b.stat().st_size
                    rows = (size - 4) // 4  # 4 bytes header + 4 bytes per float32
                    print(f"      - {b.name}: {rows} values")
    else:
        print("📦 Features: Not found")

    # Check database connection
    print()
    print("📡 Database Status:")
    try:
        session = await get_db_session()
        query = text("""
            SELECT
                symbol,
                COUNT(*) as rows,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM ohlcv_data
            WHERE timeframe = '1d' AND market_type = 'futures'
            GROUP BY symbol
            ORDER BY symbol
            LIMIT 10
        """)
        result = await session.execute(query)
        rows = result.fetchall()
        if rows:
            print(f"   Found {len(rows)} symbols in database (showing first 10):")
            for row in rows:
                print(f"   - {row[0]}: {row[1]} rows ({row[2]} ~ {row[3]})")
        else:
            print("   No OHLCV data found in database")
        await session.close()
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Sync database to Qlib binary format")
    parser.add_argument("--sync", action="store_true", help="Run sync")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument("--symbol", type=str, help="Sync specific symbol")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (default: 1d)")
    parser.add_argument("--market-type", type=str, default="futures", help="Market type (default: futures)")
    parser.add_argument("--force", action="store_true", help="Force full resync")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_QLIB_DATA_DIR

    if args.status or (not args.sync):
        asyncio.run(show_status(output_dir))

    if args.sync:
        if args.symbol:
            # Sync single symbol
            async def sync_one():
                session = await get_db_session()
                try:
                    return await sync_symbol(
                        session, args.symbol, output_dir,
                        args.timeframe, args.market_type
                    )
                finally:
                    await session.close()
            asyncio.run(sync_one())
        else:
            # Sync all symbols
            asyncio.run(sync_all_symbols(
                output_dir, args.timeframe, args.market_type, args.force
            ))


if __name__ == "__main__":
    main()
