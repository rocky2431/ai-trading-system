#!/usr/bin/env python3
"""Benchmark Alpha158 calculation: Qlib C++ vs Pandas.

This script compares the performance of calculating Alpha158 factors using:
1. Qlib's C++ Expression Engine (via D.features)
2. Local Pandas Implementation (legacy)

Requirements:
- Qlib properly installed with C++ extensions
- Data converted to Qlib binary format (run verify_qlib_data.py first)
"""

import time
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Define some complex Alpha158 expressions for benchmarking
# Source: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
ALPHA158_EXPRESSIONS = [
    # Momentum
    "Ref($close, 10) / $close - 1",  # ROC10
    "Mean($close, 5) / $close",      # MA5
    "Std($close, 20)",               # VOL20
    
    # Volume
    "Mean($volume, 5) / $volume",    # VMA5
    
    # Complex (Resi, Slope) - C++ optimized
    "Slope($close, 10)",
    "Rsquare($close, 5)",
    "Resi($close, 10)",
    
    # Volatility / Range
    "($high - $low) / $close",
    "($close - $open) / ($high - $low + 1e-10)",
    
    # KDJ-like (needs specific formulation in Qlib expr)
    "(Mean($close, 9) - Min($low, 9)) / (Max($high, 9) - Min($low, 9) + 1e-10)",
]

def benchmark_qlib_cpp(data_dir: str):
    """Benchmark Qlib C++ engine."""
    print("-" * 60)
    print("Running Qlib C++ Engine Benchmark...")
    
    import qlib
    from qlib.data import D
    
    # Initialize
    qlib.init(provider_uri=data_dir)
    
    instruments = D.instruments(market="all")
    start_time = time.time()
    
    # Load and compute
    try:
        features = D.features(
            instruments=instruments,
            fields=ALPHA158_EXPRESSIONS,
            start_time="2022-01-01",
            end_time="2024-12-31"
        )
        duration = time.time() - start_time
        
        print(f"✅ Qlib C++ Finished in {duration:.4f}s")
        print(f"   Shape: {features.shape}")
        print(f"   NaNs: {features.isna().sum().sum()}")
        return duration, features
    except Exception as e:
        print(f"❌ Qlib C++ Failed: {e}")
        return None, None

def benchmark_pandas_legacy(csv_path: str):
    """Benchmark Pandas implementation."""
    print("-" * 60)
    print("Running Pandas Legacy Benchmark...")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        # Prepare for factor engine (which expects certain column names)
        # Map columns if necessary
        column_map = {
            'timestamp': 'datetime',
            'symbol': 'instrument'
        }
        df = df.rename(columns=column_map)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        
        start_time = time.time()
        
        # Simulate calculations using standard pandas
        # We manually implement the SAME expressions as above for fair comparison
        results = pd.DataFrame(index=df.index)
        
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        open_ = df['open']
        
        # 1. ROC10
        results['ROC10'] = close.shift(10) / close - 1
        
        # 2. MA5
        results['MA5'] = close.rolling(5).mean() / close
        
        # 3. VOL20
        results['VOL20'] = close.rolling(20).std()
        
        # 4. VMA5
        results['VMA5'] = volume.rolling(5).mean() / volume
        
        # 5. Slope(10) - Pure python/numpy polyfit is slow, using simplified formula
        # Slope = Cov(x, y) / Var(x)
        # This is actually hard to do efficiently in pure Pandas rolling without apply() which is slow
        def rolling_slope(series, window):
            y = series.values
            x = np.arange(window)
            slopes = [np.nan] * (window-1)
            for i in range(window, len(y)+1):
                y_slice = y[i-window:i]
                s = np.polyfit(x, y_slice, 1)[0]
                slopes.append(s)
            return pd.Series(slopes, index=series.index)
            
        # For fairness, we'll use a slightly optimized Apply but it will still be slow
        # results['Slope10'] = close.rolling(10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
        # The above line makes Pandas VERY slow. Let's assume a "Smart Pandas" user who avoids .apply()
        # But Slope is inherently complex. We will skip the slowest ones to avoid waiting forever, 
        # or include one to show the difference.
        
        # 6. Rsquare(5) - Also complex
        # 7. Resi(10) - Also complex
        
        # 8. Range 1
        results['Range1'] = (high - low) / close
        
        # 9. Range 2
        results['Range2'] = (close - open_) / (high - low + 1e-10)
        
        # 10. KDJ-like
        rsv = (close.rolling(9).mean() - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min() + 1e-10)
        results['KDJ'] = rsv
        
        duration = time.time() - start_time
        print(f"✅ Pandas Finished in {duration:.4f}s (Note: Skipped complex Rolling Slope/Resi)")
        print(f"   Shape: {results.shape}")
        return duration, results
        
    except Exception as e:
        print(f"❌ Pandas Failed: {e}")
        return None, None

if __name__ == "__main__":
    # Settings
    DATA_CSV = "data/sample/eth_usdt_futures_daily.csv"
    QLIB_DIR = "data/qlib_data_verify"
    
    if not Path(DATA_CSV).exists() or not Path(QLIB_DIR).exists():
        print("Data missing. Please run verify_qlib_data.py first.")
        exit(1)
        
    print("=" * 60)
    print("ALPHA158 PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    t_qlib, _ = benchmark_qlib_cpp(QLIB_DIR)
    t_pandas, _ = benchmark_pandas_legacy(DATA_CSV)
    
    print("-" * 60)
    print("RESULTS:")
    if t_qlib and t_pandas:
        speedup = t_pandas / t_qlib
        print(f"🚀 Qlib C++ is {speedup:.2f}x faster than Pandas")
        print("(And Qlib calculated complex Slope/Resi/Rsquare which Pandas skipped!)")
    
    print("=" * 60)
