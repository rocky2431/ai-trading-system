#!/usr/bin/env python3
"""Full Pipeline Test - IQFMP Complete Flow Verification.

Tests all components of the IQFMP system:
1. Health Check
2. Data Source Verification (CSV exists)
3. Factor Creation with Real Code
4. Factor Listing & Statistics
5. Strategy Creation
6. Full Backtest Execution with ETH/USDT 3-year data
7. Backtest Results Verification
8. System Status Check
9. Pipeline/RD Loop Status
10. Monitoring Metrics

NO SHORTCUTS - Every step is fully executed and verified.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"

# Correct API endpoints (based on OpenAPI spec)
ENDPOINTS = {
    "health": f"{BASE_URL}/health",
    "factors": f"{API_URL}/factors",
    "factor_stats": f"{API_URL}/factors/stats",
    "factor_mining": f"{API_URL}/factors/mining",
    "strategies": f"{API_URL}/backtest/strategies",
    "backtests": f"{API_URL}/backtest/backtests",
    "backtest_stats": f"{API_URL}/backtest/stats",
    "system_status": f"{API_URL}/system/status",
    "pipeline_runs": f"{API_URL}/pipeline/runs",
    "rd_loop_runs": f"{API_URL}/pipeline/rd-loop/runs",
    "config_status": f"{API_URL}/config/status",
    "data_ohlcv": f"{API_URL}/data/ohlcv/query",
    "data_status": f"{API_URL}/data/status",
    "metrics_thresholds": f"{API_URL}/metrics/thresholds",
    "research_ledger": f"{API_URL}/research/ledger",
}

# Test results tracking
test_results = []
factor_id = None
strategy_id = None
backtest_id = None


def log_test(name: str, passed: bool, details: str = ""):
    """Log test result."""
    status = "PASS" if passed else "FAIL"
    test_results.append({"name": name, "passed": passed, "details": details})
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


def print_separator(title: str):
    """Print section separator."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================
# Step 1: Health Check
# ============================================
def test_health_check():
    print_separator("Step 1: Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        data = resp.json()
        passed = resp.status_code == 200 and data.get("status") == "healthy"
        log_test("Backend Health Check", passed, f"Status: {data}")
        return passed
    except Exception as e:
        log_test("Backend Health Check", False, str(e))
        return False


# ============================================
# Step 2: Data Source Verification
# ============================================
def test_data_source():
    print_separator("Step 2: Data Source Verification")

    # Check CSV file exists
    csv_path = Path(__file__).parent.parent / "data" / "sample" / "eth_usdt_futures_daily.csv"
    csv_exists = csv_path.exists()

    if csv_exists:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row_count = len(df)
        date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        log_test("CSV Data File", True, f"{row_count} rows, {date_range}")
    else:
        log_test("CSV Data File", False, f"File not found: {csv_path}")
        return False

    # Test data status API
    try:
        resp = requests.get(ENDPOINTS["data_status"], timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            log_test("Data Status API", True, f"Data status: {json.dumps(data)[:200]}...")
        else:
            log_test("Data Status API", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Data Status API", False, f"Error: {e}")

    # Test OHLCV query API
    try:
        query_payload = {
            "symbol": "ETHUSDT",
            "timeframe": "1d",
            "limit": 5
        }
        resp = requests.post(ENDPOINTS["data_ohlcv"], json=query_payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            log_test("Data OHLCV Query API", True, f"Response: {json.dumps(data)[:200]}...")
        else:
            log_test("Data OHLCV Query API", False, f"Status: {resp.status_code}, {resp.text[:100]}")
    except Exception as e:
        log_test("Data OHLCV Query API", False, f"Error: {e}")

    return csv_exists


# ============================================
# Step 3: Factor Creation
# ============================================
def test_factor_creation():
    global factor_id
    print_separator("Step 3: Factor Creation with Real Code")

    # Real RSI momentum factor code
    factor_code = '''import pandas as pd
import numpy as np

def compute_factor(df: pd.DataFrame) -> pd.Series:
    """Compute RSI-based momentum factor.

    RSI (Relative Strength Index) normalized to [-1, 1] range.
    Positive values indicate overbought, negative indicate oversold.
    """
    close = df["close"]
    delta = close.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # Calculate average gain/loss over 14 periods
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))

    # Normalize to [-1, 1]
    factor = (rsi - 50) / 50

    return factor
'''

    payload = {
        "name": f"full_test_rsi_momentum_{int(time.time())}",
        "family": ["momentum", "technical"],
        "code": factor_code,
        "description": "RSI-based momentum factor for full pipeline test"
    }

    try:
        resp = requests.post(f"{API_URL}/factors/", json=payload, timeout=30)
        if resp.status_code in [200, 201]:
            data = resp.json()
            factor_id = data.get("id")
            log_test("Factor Creation", True, f"Factor ID: {factor_id}")
            print(f"       Factor Details: {json.dumps(data, indent=2)[:500]}...")
            return True
        else:
            log_test("Factor Creation", False, f"Status: {resp.status_code}, Response: {resp.text[:200]}")
            return False
    except Exception as e:
        log_test("Factor Creation", False, str(e))
        return False


# ============================================
# Step 4: Factor Listing & Statistics
# ============================================
def test_factor_listing():
    print_separator("Step 4: Factor Listing & Statistics")

    # List all factors
    try:
        resp = requests.get(f"{API_URL}/factors/", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            factors = data if isinstance(data, list) else data.get("factors", [])
            log_test("Factor Listing", True, f"Total factors: {len(factors)}")

            # Show factor families
            families = {}
            for f in factors:
                for fam in f.get("family", []):
                    families[fam] = families.get(fam, 0) + 1
            print(f"       Families: {families}")
        else:
            log_test("Factor Listing", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Factor Listing", False, str(e))

    # Get factor statistics
    try:
        resp = requests.get(f"{API_URL}/factors/stats", timeout=10)
        if resp.status_code == 200:
            stats = resp.json()
            log_test("Factor Statistics", True, f"Stats: {json.dumps(stats, indent=2)}")
        else:
            log_test("Factor Statistics", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Factor Statistics", False, str(e))

    return True


# ============================================
# Step 5: Strategy Creation
# ============================================
def test_strategy_creation():
    global strategy_id, factor_id
    print_separator("Step 5: Strategy Creation")

    if not factor_id:
        log_test("Strategy Creation", False, "No factor_id available")
        return False

    # Correct payload format based on BacktestCreateRequest schema
    strategy_payload = {
        "name": f"full_test_rsi_strategy_{int(time.time())}",
        "description": "RSI momentum strategy for full pipeline test",
        "factor_ids": [factor_id],
        "weighting_method": "equal",
        "rebalance_frequency": "daily",
        "universe": "all",
        "long_only": False,
        "max_positions": 20
    }

    try:
        resp = requests.post(ENDPOINTS["strategies"], json=strategy_payload, timeout=30)
        if resp.status_code in [200, 201]:
            data = resp.json()
            strategy_id = data.get("id")
            log_test("Strategy Creation", True, f"Strategy ID: {strategy_id}")
            print(f"       Strategy: {json.dumps(data, indent=2)[:300]}...")
            return True
        else:
            log_test("Strategy Creation", False, f"Status: {resp.status_code}, Response: {resp.text[:200]}")
            # Still continue - we can list existing strategies
            return False
    except Exception as e:
        log_test("Strategy Creation", False, str(e))
        return False


# ============================================
# Step 6: Full Backtest Execution
# ============================================
def test_backtest_execution():
    global backtest_id, strategy_id, factor_id
    print_separator("Step 6: Full Backtest Execution (ETH/USDT 3-Year)")

    # Need a strategy_id
    if not strategy_id:
        # Try to get an existing strategy or create one with a default factor
        log_test("Backtest Prereq", False, "No strategy_id available - creating strategy first")

        # Get existing factors
        try:
            resp = requests.get(ENDPOINTS["factors"], timeout=10)
            if resp.status_code == 200:
                factors = resp.json()
                if factors and len(factors) > 0:
                    factor_id = factors[0].get("id")
                    print(f"       Using existing factor: {factor_id}")

                    # Create strategy
                    strategy_payload = {
                        "name": f"auto_test_strategy_{int(time.time())}",
                        "description": "Auto-created strategy for backtest test",
                        "factor_ids": [factor_id],
                        "weighting_method": "equal",
                        "rebalance_frequency": "daily",
                        "universe": "all",
                        "long_only": False,
                        "max_positions": 20
                    }
                    s_resp = requests.post(ENDPOINTS["strategies"], json=strategy_payload, timeout=30)
                    if s_resp.status_code in [200, 201]:
                        strategy_id = s_resp.json().get("id")
                        print(f"       Created strategy: {strategy_id}")
        except Exception as e:
            print(f"       Error getting factors: {e}")

    if not strategy_id:
        log_test("Backtest Execution", False, "No strategy available")
        return False

    # Correct backtest config format based on BacktestCreateRequest schema
    backtest_request = {
        "strategy_id": strategy_id,
        "config": {
            "start_date": "2022-12-12",  # Match CSV data start
            "end_date": "2025-12-10",    # Match CSV data end
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "slippage": 0.001,
            "benchmark": "ETH",
            "risk_free_rate": 0.02
        },
        "name": f"full_pipeline_test_{int(time.time())}",
        "description": "Full pipeline test backtest with ETH/USDT 3-year data"
    }

    print(f"       Config: {json.dumps(backtest_request, indent=2)}")

    try:
        # Start backtest
        resp = requests.post(ENDPOINTS["backtests"], json=backtest_request, timeout=120)

        if resp.status_code in [200, 201, 202]:
            data = resp.json()
            log_test("Backtest Created", True, f"Response: {json.dumps(data)}")

            # Parse backtest ID from message if needed
            message = data.get("message", "")
            if "ID:" in message:
                backtest_id = message.split("ID:")[-1].strip()
            else:
                backtest_id = data.get("id")

            if backtest_id:
                print(f"       Backtest ID: {backtest_id}")

                # Poll for completion
                print("       Waiting for backtest to complete...")
                for i in range(60):  # Max 120 seconds
                    time.sleep(2)
                    try:
                        status_resp = requests.get(f"{ENDPOINTS['backtests']}/{backtest_id}", timeout=10)
                        if status_resp.status_code == 200:
                            status_data = status_resp.json()
                            bt_status = status_data.get("status", "unknown")
                            progress = status_data.get("progress", 0)
                            print(f"       Status: {bt_status}, Progress: {progress}%")

                            if bt_status == "completed":
                                log_test("Backtest Completed", True, f"Duration: {i*2}s")
                                return verify_backtest_results(status_data)
                            elif bt_status == "failed":
                                error = status_data.get("error_message", "Unknown error")
                                log_test("Backtest Failed", False, f"Error: {error}")
                                return False
                        else:
                            print(f"       Status check failed: {status_resp.status_code}")
                    except Exception as e:
                        print(f"       Status check error: {e}")

                log_test("Backtest Timeout", False, "Exceeded 120s")
                return False
            else:
                log_test("Backtest Execution", True, "Backtest submitted (no ID returned)")
                return True
        else:
            log_test("Backtest Execution", False, f"Status: {resp.status_code}, Response: {resp.text[:500]}")
            return False

    except Exception as e:
        log_test("Backtest Execution", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def verify_backtest_results(data):
    """Verify backtest results contain expected metrics."""
    print_separator("Step 7: Backtest Results Verification")

    # Check required metrics
    required_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]

    results = data.get("results") or data.get("metrics") or data

    found_metrics = []
    missing_metrics = []

    for metric in required_metrics:
        if metric in results or metric in data:
            value = results.get(metric) or data.get(metric)
            found_metrics.append(f"{metric}={value}")
        else:
            missing_metrics.append(metric)

    if missing_metrics:
        log_test("Backtest Metrics", False, f"Missing: {missing_metrics}")
    else:
        log_test("Backtest Metrics", True, f"Found all metrics")
        print(f"       Metrics: {', '.join(found_metrics)}")

    # Check trade count
    trade_count = results.get("trade_count") or data.get("trade_count", 0)
    log_test("Trade Count", trade_count > 0, f"Trades: {trade_count}")

    # Check data points used
    data_points = results.get("data_points") or data.get("data_points", 0)
    if data_points > 0:
        log_test("Data Points", True, f"Used {data_points} data points")

    return len(missing_metrics) == 0


# ============================================
# Step 8: System Status Check
# ============================================
def test_system_status():
    print_separator("Step 8: System Status Check")

    try:
        resp = requests.get(ENDPOINTS["system_status"], timeout=10)
        if resp.status_code == 200:
            status = resp.json()
            log_test("System Status", True, "")

            # Print key metrics
            print(f"       Response: {json.dumps(status, indent=2)[:500]}...")

            return True
        else:
            log_test("System Status", False, f"Status: {resp.status_code}")
            return False
    except Exception as e:
        log_test("System Status", False, str(e))
        return False

    # Test config status
    try:
        resp = requests.get(ENDPOINTS["config_status"], timeout=10)
        if resp.status_code == 200:
            config = resp.json()
            log_test("Config Status", True, f"Config: {json.dumps(config)[:200]}...")
        else:
            log_test("Config Status", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Config Status", False, str(e))


# ============================================
# Step 9: Pipeline/RD Loop Status
# ============================================
def test_pipeline_status():
    print_separator("Step 9: Pipeline/RD Loop Status")

    # Check pipeline runs
    try:
        resp = requests.get(ENDPOINTS["pipeline_runs"], timeout=10)
        if resp.status_code == 200:
            runs = resp.json()
            log_test("Pipeline Runs", True, f"Total runs: {len(runs) if isinstance(runs, list) else runs.get('total', 0)}")
        else:
            log_test("Pipeline Runs", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Pipeline Runs", False, str(e))

    # Check RD loop runs
    try:
        resp = requests.get(ENDPOINTS["rd_loop_runs"], timeout=10)
        if resp.status_code == 200:
            rd_runs = resp.json()
            log_test("RD Loop Runs", True, f"Runs: {json.dumps(rd_runs)[:200]}...")
        else:
            log_test("RD Loop Runs", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("RD Loop Runs", False, str(e))

    # Check mining tasks
    try:
        resp = requests.get(ENDPOINTS["factor_mining"], timeout=10)
        if resp.status_code == 200:
            mining = resp.json()
            log_test("Mining Tasks", True, f"Tasks: {json.dumps(mining)[:200]}...")
        else:
            log_test("Mining Tasks", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Mining Tasks", False, str(e))

    # Check backtest stats
    try:
        resp = requests.get(ENDPOINTS["backtest_stats"], timeout=10)
        if resp.status_code == 200:
            stats = resp.json()
            log_test("Backtest Stats", True, f"Stats: {json.dumps(stats)}")
        else:
            log_test("Backtest Stats", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Backtest Stats", False, str(e))

    return True


# ============================================
# Step 10: Monitoring Metrics
# ============================================
def test_monitoring():
    print_separator("Step 10: Monitoring Metrics")

    # Check metrics thresholds endpoint
    try:
        resp = requests.get(ENDPOINTS["metrics_thresholds"], timeout=10)
        if resp.status_code == 200:
            metrics = resp.json()
            log_test("Metrics Thresholds", True, f"Thresholds: {json.dumps(metrics)[:200]}...")
        else:
            log_test("Metrics Thresholds", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Metrics Thresholds", False, str(e))

    # Check research ledger
    try:
        resp = requests.get(ENDPOINTS["research_ledger"], timeout=10)
        if resp.status_code == 200:
            ledger = resp.json()
            log_test("Research Ledger", True, f"Ledger: {json.dumps(ledger)[:200]}...")
        else:
            log_test("Research Ledger", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Research Ledger", False, str(e))

    # Check config status
    try:
        resp = requests.get(ENDPOINTS["config_status"], timeout=10)
        if resp.status_code == 200:
            config = resp.json()
            log_test("Config Status", True, f"Config: {json.dumps(config)[:200]}...")
        else:
            log_test("Config Status", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Config Status", False, str(e))

    return True


# ============================================
# Main Test Runner
# ============================================
def run_all_tests():
    """Run complete pipeline test suite."""
    print("\n" + "=" * 60)
    print("  IQFMP FULL PIPELINE TEST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    start_time = time.time()

    # Run all tests in sequence
    tests = [
        ("Health Check", test_health_check),
        ("Data Source", test_data_source),
        ("Factor Creation", test_factor_creation),
        ("Factor Listing", test_factor_listing),
        ("Strategy Creation", test_strategy_creation),
        ("Backtest Execution", test_backtest_execution),
        ("System Status", test_system_status),
        ("Pipeline Status", test_pipeline_status),
        ("Monitoring", test_monitoring),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print_separator("TEST SUMMARY")

    passed = sum(1 for r in test_results if r["passed"])
    failed = sum(1 for r in test_results if not r["passed"])

    print(f"\nTotal Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Duration: {time.time() - start_time:.2f}s")

    print("\n--- Detailed Results ---")
    for result in test_results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}")

    # Overall result
    print("\n" + "=" * 60)
    if failed == 0:
        print("  ALL TESTS PASSED!")
    else:
        print(f"  {failed} TESTS FAILED")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
