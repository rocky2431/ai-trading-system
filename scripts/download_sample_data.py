#!/usr/bin/env python3
"""Download sample data for IQFMP testing.

Downloads 3 years of ETH/USDT perpetual futures daily data from Binance.
Saves to both CSV and database (if available).
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("Installing ccxt...")
    os.system("pip install ccxt")
    import ccxt.async_support as ccxt


async def download_futures_ohlcv(
    symbol: str = "ETH/USDT:USDT",  # Perpetual futures symbol
    timeframe: str = "1d",
    years: int = 3,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Download historical futures OHLCV data.

    Args:
        symbol: Trading pair (ETH/USDT:USDT for perpetual)
        timeframe: Timeframe (1d for daily)
        years: Number of years to download
        exchange_id: Exchange to use

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Initializing {exchange_id} exchange...")

    # Create exchange with futures market type
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # perpetual futures
        }
    })

    try:
        # Load markets
        await exchange.load_markets()
        print(f"Loaded {len(exchange.markets)} markets")

        # Check if symbol exists
        if symbol not in exchange.markets:
            # Try alternative symbol format
            alt_symbols = [
                "ETH/USDT:USDT",
                "ETHUSDT",
                "ETH/USDT",
            ]
            for alt in alt_symbols:
                if alt in exchange.markets:
                    symbol = alt
                    break
            else:
                print(f"Available futures symbols: {[s for s in exchange.markets if 'ETH' in s and 'USDT' in s][:10]}")
                raise ValueError(f"Symbol {symbol} not found")

        print(f"Using symbol: {symbol}")

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=365 * years)

        print(f"Downloading {timeframe} data from {start_time.date()} to {end_time.date()}")

        # Download in batches
        all_data = []
        current_time = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)

        batch_count = 0
        while current_time < end_timestamp:
            batch_count += 1
            print(f"Fetching batch {batch_count}...", end=" ")

            try:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_time,
                    limit=1000,
                )

                if not ohlcv:
                    print("No more data")
                    break

                all_data.extend(ohlcv)
                print(f"Got {len(ohlcv)} candles (total: {len(all_data)})")

                # Move to next batch
                current_time = ohlcv[-1][0] + 1

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(2)
                continue

        print(f"\nTotal candles downloaded: {len(all_data)}")

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add metadata columns
        df["symbol"] = symbol.replace(":USDT", "").replace("/", "")
        df["exchange"] = exchange_id
        df["market_type"] = "futures"
        df["timeframe"] = timeframe

        print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total rows: {len(df)}")

        return df

    finally:
        await exchange.close()


def save_to_csv(df: pd.DataFrame, filepath: Path):
    """Save DataFrame to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")


async def save_to_database(df: pd.DataFrame, batch_size: int = 500):
    """Save DataFrame to database using batch inserts for better performance.

    Args:
        df: DataFrame with OHLCV data
        batch_size: Number of rows per batch insert
    """
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text

        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            print("DATABASE_URL not set, skipping database save")
            return False

        # Ensure async driver
        if "asyncpg" not in database_url:
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

        print(f"Connecting to database...")
        engine = create_async_engine(database_url, echo=False)

        async with engine.begin() as conn:
            # Ensure table exists (should be created by init_db.py)
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
                    market_type VARCHAR(10) NOT NULL DEFAULT 'spot',
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL
                )
            """))

            # Create unique constraint for upsert
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'ohlcv_unique_symbol_timeframe_timestamp'
                    ) THEN
                        ALTER TABLE ohlcv_data
                        ADD CONSTRAINT ohlcv_unique_symbol_timeframe_timestamp
                        UNIQUE (symbol, timeframe, timestamp);
                    END IF;
                END $$;
            """))

            print("Table verified")

        # Batch insert with upsert (ON CONFLICT UPDATE)
        async with engine.begin() as conn:
            inserted = 0
            total_rows = len(df)

            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                batch_values = []

                for _, row in batch.iterrows():
                    batch_values.append({
                        "symbol": row["symbol"],
                        "timeframe": row["timeframe"],
                        "exchange": row["exchange"],
                        "market_type": row["market_type"],
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    })

                if batch_values:
                    # Build batch insert statement with upsert
                    placeholders = []
                    params = {}
                    for j, val in enumerate(batch_values):
                        idx = i + j
                        placeholders.append(
                            f"(:symbol_{idx}, :timeframe_{idx}, :exchange_{idx}, :market_type_{idx}, "
                            f":timestamp_{idx}, :open_{idx}, :high_{idx}, :low_{idx}, :close_{idx}, :volume_{idx})"
                        )
                        for k, v in val.items():
                            params[f"{k}_{idx}"] = v

                    sql = f"""
                        INSERT INTO ohlcv_data
                        (symbol, timeframe, exchange, market_type, timestamp, open, high, low, close, volume)
                        VALUES {', '.join(placeholders)}
                        ON CONFLICT (symbol, timeframe, timestamp)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """

                    try:
                        await conn.execute(text(sql), params)
                        inserted += len(batch_values)
                        print(f"  Inserted batch {i // batch_size + 1} ({inserted}/{total_rows} rows)")
                    except Exception as e:
                        print(f"  Batch insert error: {e}")
                        # Fallback to row-by-row insert for this batch
                        for val in batch_values:
                            try:
                                await conn.execute(
                                    text("""
                                        INSERT INTO ohlcv_data
                                        (symbol, timeframe, exchange, market_type, timestamp, open, high, low, close, volume)
                                        VALUES (:symbol, :timeframe, :exchange, :market_type, :timestamp, :open, :high, :low, :close, :volume)
                                        ON CONFLICT (symbol, timeframe, timestamp)
                                        DO UPDATE SET
                                            open = EXCLUDED.open,
                                            high = EXCLUDED.high,
                                            low = EXCLUDED.low,
                                            close = EXCLUDED.close,
                                            volume = EXCLUDED.volume
                                    """),
                                    val
                                )
                                inserted += 1
                            except Exception:
                                pass

            print(f"Total inserted/updated: {inserted} rows")

        await engine.dispose()
        return True

    except Exception as e:
        print(f"Database save failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    print("=" * 60)
    print("IQFMP Sample Data Downloader")
    print("=" * 60)
    print()

    # Download data
    df = await download_futures_ohlcv(
        symbol="ETH/USDT:USDT",
        timeframe="1d",
        years=3,
        exchange_id="binance",
    )

    # Save to CSV (always works)
    data_dir = Path(__file__).parent.parent / "data" / "sample"
    csv_path = data_dir / "eth_usdt_futures_daily.csv"
    save_to_csv(df, csv_path)

    # Try to save to database
    print("\nAttempting database save...")
    await save_to_database(df)

    print()
    print("=" * 60)
    print("Download complete!")
    print(f"CSV file: {csv_path}")
    print("=" * 60)

    # Print sample data
    print("\nSample data (first 5 rows):")
    print(df.head())
    print("\nSample data (last 5 rows):")
    print(df.tail())

    # Print statistics
    print("\nData statistics:")
    print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Total days: {len(df)}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"  Average volume: {df['volume'].mean():,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
