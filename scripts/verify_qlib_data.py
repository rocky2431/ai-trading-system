#!/usr/bin/env python3
"""Verify Qlib binary data loading capabilities.

This script verifies that:
1. Qlib is properly initialized
2. Binary data can be loaded via D.features() API
3. Crypto data handler works with vendor/qlib deep fork

Usage:
    # Check current status
    python scripts/verify_qlib_data.py --check

    # Convert CSV to Qlib binary format
    python scripts/verify_qlib_data.py --convert data/sample/eth_usdt_futures_daily.csv

    # Test D.features() API
    python scripts/verify_qlib_data.py --test-features
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def check_qlib_status() -> bool:
    """Check Qlib installation and configuration status."""
    print("=" * 60)
    print("Qlib 状态检查")
    print("=" * 60)

    # 1. Check Qlib import
    try:
        import qlib
        print(f"✅ Qlib 版本: {qlib.__version__}")
        print(f"   路径: {qlib.__file__}")
    except ImportError as e:
        print(f"❌ Qlib 导入失败: {e}")
        return False

    # 2. Check C++ extensions
    try:
        from qlib.data._libs.rolling import rolling_resi
        from qlib.data._libs.expanding import expanding_resi
        print("✅ C++ 扩展 (rolling/expanding) 可用")
    except ImportError as e:
        print(f"❌ C++ 扩展不可用: {e}")
        return False

    # 3. Check provider_uri configuration
    qlib_data_dir = os.environ.get("QLIB_DATA_DIR")
    print(f"\n📁 数据目录配置:")
    print(f"   QLIB_DATA_DIR: {qlib_data_dir or '未设置'}")

    # 4. Check if data directory exists
    if qlib_data_dir:
        data_path = Path(qlib_data_dir)
        if data_path.exists():
            print(f"   ✅ 目录存在")
            # List instruments
            instruments = list(data_path.glob("*"))
            if instruments:
                print(f"   📊 发现 {len(instruments)} 个品种目录:")
                for inst in instruments[:5]:
                    if inst.is_dir():
                        print(f"      - {inst.name}")
        else:
            print(f"   ❌ 目录不存在")
    else:
        print("   ⚠️  未设置 QLIB_DATA_DIR，将使用默认路径")

    # 5. Check CryptoDataHandler
    try:
        from qlib.contrib.crypto import CryptoDataHandler, QLIB_AVAILABLE
        print(f"\n📦 Crypto 扩展:")
        print(f"   ✅ CryptoDataHandler 可用")
        print(f"   Qlib backend: {'启用' if QLIB_AVAILABLE else '禁用'}")
    except ImportError as e:
        print(f"   ❌ Crypto 扩展不可用: {e}")

    print("=" * 60)
    return True


def convert_csv_to_qlib_binary(csv_path: str, output_dir: str = None) -> bool:
    """Convert CSV data to Qlib binary format.

    Qlib binary format:
    - Directory structure: {instrument}/{field}.{freq}.bin
    - Example: ethusdt/$close.1d.bin
    - Binary format: float32 little-endian with start_index prefix
    """
    print("=" * 60)
    print(f"转换 CSV 到 Qlib 二进制格式")
    print("=" * 60)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV 文件不存在: {csv_path}")
        return False

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"✅ 读取 CSV: {len(df)} 行")
    print(f"   列: {list(df.columns)}")

    # Determine instrument name
    if "symbol" in df.columns:
        instrument = df["symbol"].iloc[0].lower()
    else:
        instrument = csv_path.stem.replace("_", "").lower()
    print(f"   品种: {instrument}")

    # Output directory
    if output_dir is None:
        output_dir = Path("data/qlib_data")
    else:
        output_dir = Path(output_dir)

    # Qlib expects features in features/{instrument}/ subdirectory
    inst_dir = output_dir / "features" / instrument
    inst_dir.mkdir(parents=True, exist_ok=True)
    print(f"   输出目录: {inst_dir}")

    # Convert timestamp to Qlib calendar index
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        print("❌ 找不到时间列 (timestamp 或 datetime)")
        return False

    # Create calendar (trading days)
    df = df.sort_values("datetime").reset_index(drop=True)
    # Remove timezone info for consistency
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    start_date = df["datetime"].min()
    print(f"   日期范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # Map to calendar index (days since epoch)
    epoch = pd.Timestamp("1970-01-01")
    df["_cal_idx"] = (df["datetime"] - epoch).dt.days
    start_index = int(df["_cal_idx"].iloc[0])

    # Fields to convert
    field_mapping = {
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
        "funding_rate": "$funding_rate",
        "open_interest": "$open_interest",
    }

    converted = []
    for csv_col, qlib_field in field_mapping.items():
        if csv_col not in df.columns:
            continue

        # Get values
        values = df[csv_col].values.astype(np.float32)

        # Write to binary file
        # Format: [start_index (float32)] + [values (float32 array)]
        # Use .day.bin for daily data (Qlib convention)
        bin_path = inst_dir / f"{qlib_field}.day.bin"
        with open(bin_path, "wb") as f:
            np.array([start_index], dtype="<f").tofile(f)
            values.astype("<f").tofile(f)

        converted.append(qlib_field)
        print(f"   ✅ {qlib_field} -> {bin_path.name} ({len(values)} 值)")

    # Create calendar file
    calendar_dir = output_dir / "calendars"
    calendar_dir.mkdir(exist_ok=True)
    calendar_file = calendar_dir / "day.txt"
    calendar_dates = df["datetime"].dt.strftime("%Y-%m-%d").tolist()
    with open(calendar_file, "w") as f:
        f.write("\n".join(calendar_dates))
    print(f"   ✅ calendar -> {calendar_file} ({len(calendar_dates)} 天)")

    # Create instruments file
    instruments_dir = output_dir / "instruments"
    instruments_dir.mkdir(exist_ok=True)
    instruments_file = instruments_dir / "all.txt"
    start_str = df["datetime"].min().strftime("%Y-%m-%d")
    end_str = df["datetime"].max().strftime("%Y-%m-%d")
    with open(instruments_file, "w") as f:
        f.write(f"{instrument}\t{start_str}\t{end_str}\n")
    print(f"   ✅ instruments -> {instruments_file}")

    print(f"\n🎉 转换完成！共 {len(converted)} 个字段")
    print(f"\n使用方法:")
    print(f"  export QLIB_DATA_DIR={output_dir.absolute()}")
    print(f"  python -c \"import qlib; qlib.init(provider_uri='{output_dir.absolute()}')\"")

    print("=" * 60)
    return True


def test_features_api() -> bool:
    """Test D.features() API with real data."""
    print("=" * 60)
    print("测试 D.features() API")
    print("=" * 60)

    qlib_data_dir = os.environ.get("QLIB_DATA_DIR")
    if not qlib_data_dir:
        print("⚠️  QLIB_DATA_DIR 未设置")
        print("   请先运行: python scripts/verify_qlib_data.py --convert <csv_file>")
        print("   然后设置: export QLIB_DATA_DIR=data/qlib_data")
        return False

    data_path = Path(qlib_data_dir)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        return False

    # Initialize Qlib
    import qlib
    from qlib.data import D

    try:
        qlib.init(provider_uri=str(data_path))
        print("✅ Qlib 初始化成功")
    except Exception as e:
        print(f"❌ Qlib 初始化失败: {e}")
        return False

    # List instruments
    try:
        instruments = D.instruments(market="all")
        print(f"✅ 发现品种: {instruments}")
    except Exception as e:
        print(f"❌ 获取品种列表失败: {e}")
        return False

    # Test D.features()
    try:
        features = D.features(
            instruments=instruments,
            fields=["$close", "$volume"],
            start_time="2022-01-01",
            end_time="2024-12-31",
        )
        print(f"✅ D.features() 返回: {features.shape}")
        print(f"   列: {list(features.columns)}")
        print(f"   前5行:\n{features.head()}")
    except Exception as e:
        print(f"❌ D.features() 失败: {e}")
        return False

    # Test expression evaluation
    try:
        # Simple moving average
        features = D.features(
            instruments=instruments,
            fields=["Mean($close, 5)", "Std($close, 20)", "Resi($close, 10)"],
            start_time="2022-01-01",
            end_time="2024-12-31",
        )
        print(f"\n✅ 表达式计算成功:")
        print(f"   Mean($close, 5), Std($close, 20), Resi($close, 10)")
        print(f"   形状: {features.shape}")
        print(f"   前5行:\n{features.head()}")
    except Exception as e:
        print(f"⚠️  表达式计算失败: {e}")

    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify Qlib data loading")
    parser.add_argument("--check", action="store_true", help="Check Qlib status")
    parser.add_argument("--convert", type=str, help="Convert CSV to Qlib binary format")
    parser.add_argument("--output", type=str, help="Output directory for conversion")
    parser.add_argument("--test-features", action="store_true", help="Test D.features() API")

    args = parser.parse_args()

    if args.check or (not args.convert and not args.test_features):
        check_qlib_status()

    if args.convert:
        convert_csv_to_qlib_binary(args.convert, args.output)

    if args.test_features:
        test_features_api()


if __name__ == "__main__":
    main()
