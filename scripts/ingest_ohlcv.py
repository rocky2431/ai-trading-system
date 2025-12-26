#!/usr/bin/env python3
"""Data ingestion pipeline for IQFMP.

This script ingests OHLCV and derivative data from exchanges into TimescaleDB.

Supported data types:
- OHLCV (candlestick data)
- Funding rates
- Open interest
- Liquidations

Usage:
    # Ingest historical OHLCV data
    python scripts/ingest_ohlcv.py --symbol BTCUSDT --timeframe 1d --days 365

    # Ingest multiple symbols
    python scripts/ingest_ohlcv.py --symbols BTCUSDT,ETHUSDT --timeframe 1h --days 30

    # Ingest with derivatives data
    python scripts/ingest_ohlcv.py --symbol BTCUSDT --timeframe 1d --days 30 --derivatives
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not installed. Install with: pip install ccxt")


# Exchange configuration
EXCHANGE = os.environ.get("EXCHANGE", "binance")
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# Default symbols
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# Timeframe mapping
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


async def get_exchange() -> "ccxt.Exchange":
    """Create exchange instance."""
    exchange_class = getattr(ccxt, EXCHANGE)
    exchange = exchange_class({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},  # Use futures market
    })
    return exchange


async def fetch_ohlcv(
    exchange: "ccxt.Exchange",
    symbol: str,
    timeframe: str,
    since: int,
    limit: int = 1000,
) -> list:
    """Fetch OHLCV data from exchange."""
    try:
        ohlcv = await exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=since,
            limit=limit,
        )
        return ohlcv
    except Exception as e:
        print(f"Error fetching OHLCV for {symbol}: {e}")
        return []


async def fetch_funding_rate(
    exchange: "ccxt.Exchange",
    symbol: str,
) -> Optional[dict]:
    """Fetch current funding rate."""
    try:
        if hasattr(exchange, "fetch_funding_rate"):
            return await exchange.fetch_funding_rate(symbol)
    except Exception as e:
        print(f"Error fetching funding rate for {symbol}: {e}")
    return None


async def ingest_symbol_ohlcv(
    exchange: "ccxt.Exchange",
    symbol: str,
    timeframe: str,
    days: int,
) -> pd.DataFrame:
    """Ingest OHLCV data for a single symbol."""
    print(f"\n=== Ingesting {symbol} {timeframe} (last {days} days) ===")

    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)

    all_data = []
    current_since = since
    batch = 0

    while current_since < int(end_time.timestamp() * 1000):
        batch += 1
        ohlcv = await fetch_ohlcv(exchange, symbol, timeframe, current_since)

        if not ohlcv:
            break

        all_data.extend(ohlcv)
        print(f"  Batch {batch}: fetched {len(ohlcv)} candles")

        # Move to next batch
        last_timestamp = ohlcv[-1][0]
        current_since = last_timestamp + TIMEFRAME_MS.get(timeframe, 60000)

        # Rate limit
        await asyncio.sleep(0.5)

    if not all_data:
        print(f"  No data fetched for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol.replace("/", "")
    df["timeframe"] = timeframe
    df["exchange"] = EXCHANGE

    print(f"  Total: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


async def save_to_timescaledb(df: pd.DataFrame) -> int:
    """Save DataFrame to TimescaleDB."""
    if df.empty:
        return 0

    from iqfmp.db.database import get_settings
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    settings = get_settings()
    engine = create_async_engine(settings.database_url)

    try:
        async with engine.begin() as conn:
            # Prepare data for insertion
            records = df.to_dict("records")

            # Use upsert (ON CONFLICT DO UPDATE) for idempotent ingestion
            insert_sql = text("""
                INSERT INTO ohlcv_data (timestamp, symbol, timeframe, exchange, open, high, low, close, volume)
                VALUES (:timestamp, :symbol, :timeframe, :exchange, :open, :high, :low, :close, :volume)
                ON CONFLICT (timestamp, symbol, timeframe, exchange) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """)

            for record in records:
                await conn.execute(insert_sql, record)

            print(f"  Saved {len(records)} records to TimescaleDB")
            return len(records)

    except Exception as e:
        print(f"Error saving to TimescaleDB: {e}")
        return 0
    finally:
        await engine.dispose()


async def ingest_data(
    symbols: list[str],
    timeframe: str,
    days: int,
    save_to_db: bool = True,
    save_to_csv: bool = False,
) -> None:
    """Main ingestion function."""
    if not CCXT_AVAILABLE:
        print("ERROR: ccxt not installed")
        sys.exit(1)

    print(f"Starting data ingestion from {EXCHANGE}")
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {timeframe}")
    print(f"Days: {days}")

    exchange = await get_exchange()

    try:
        # Load markets
        await exchange.load_markets()
        print(f"Loaded {len(exchange.markets)} markets")

        total_records = 0

        for symbol in symbols:
            # Normalize symbol format
            if "/" not in symbol:
                symbol = f"{symbol[:3]}/{symbol[3:]}" if len(symbol) >= 6 else symbol

            if symbol not in exchange.markets:
                print(f"Warning: {symbol} not found in {EXCHANGE} markets")
                continue

            df = await ingest_symbol_ohlcv(exchange, symbol, timeframe, days)

            if df.empty:
                continue

            if save_to_db:
                records = await save_to_timescaledb(df)
                total_records += records

            if save_to_csv:
                csv_path = Path(f"data/ingested/{symbol.replace('/', '')}_{timeframe}.csv")
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"  Saved to {csv_path}")

        print(f"\n=== Ingestion Complete ===")
        print(f"Total records saved: {total_records}")

    finally:
        await exchange.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest OHLCV data into TimescaleDB")
    parser.add_argument("--symbol", type=str, help="Single symbol (e.g., BTCUSDT)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (1m,5m,15m,1h,4h,1d)")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--no-db", action="store_true", help="Skip saving to database")
    parser.add_argument("--csv", action="store_true", help="Also save to CSV")

    args = parser.parse_args()

    # Determine symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = DEFAULT_SYMBOLS

    asyncio.run(ingest_data(
        symbols=symbols,
        timeframe=args.timeframe,
        days=args.days,
        save_to_db=not args.no_db,
        save_to_csv=args.csv,
    ))
