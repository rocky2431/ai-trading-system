"""
集成测试共享 Fixtures
提供测试所需的数据和模拟对象
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ==================== 基础 Fixtures ====================


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """模拟 LLM 生成的因子代码响应"""
    return {
        "factor_code": '''
def momentum_factor(df):
    """
    动量因子
    计算过去 N 天的收益率变化
    """
    import pandas as pd

    # 计算收益率
    returns = df["close"].pct_change(periods=20)

    # 标准化处理
    factor = (returns - returns.mean()) / returns.std()

    return factor
''',
        "factor_name": "momentum_20d",
        "hypothesis": "过去 20 天表现好的资产会继续表现好",
        "family": "momentum",
    }


@pytest.fixture
def sample_market_data() -> dict[str, Any]:
    """样本市场数据"""
    import random
    from datetime import datetime, timedelta

    random.seed(42)

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    base_price = 100.0

    prices = []
    for i in range(100):
        base_price = base_price * (1 + random.uniform(-0.02, 0.025))
        prices.append(base_price)

    return {
        "dates": dates,
        "symbols": ["BTC", "ETH", "SOL"],
        "data": {
            "BTC": {
                "open": [p * 0.99 for p in prices],
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": [random.uniform(1000, 5000) for _ in range(100)],
            },
            "ETH": {
                "open": [p * 0.05 * 0.99 for p in prices],
                "high": [p * 0.05 * 1.02 for p in prices],
                "low": [p * 0.05 * 0.98 for p in prices],
                "close": [p * 0.05 for p in prices],
                "volume": [random.uniform(5000, 20000) for _ in range(100)],
            },
            "SOL": {
                "open": [p * 0.002 * 0.99 for p in prices],
                "high": [p * 0.002 * 1.02 for p in prices],
                "low": [p * 0.002 * 0.98 for p in prices],
                "close": [p * 0.002 for p in prices],
                "volume": [random.uniform(10000, 50000) for _ in range(100)],
            },
        },
    }


@pytest.fixture
def sample_factor_result() -> dict[str, Any]:
    """样本因子评估结果"""
    import random

    random.seed(42)

    return {
        "factor_id": "test_momentum_20d",
        "factor_name": "momentum_20d",
        "family": "momentum",
        "metrics": {
            "ic_mean": random.uniform(0.02, 0.08),
            "ic_std": random.uniform(0.05, 0.15),
            "ir": random.uniform(0.3, 0.8),
            "sharpe_ratio": random.uniform(1.0, 2.5),
            "max_drawdown": random.uniform(-0.15, -0.05),
            "win_rate": random.uniform(0.5, 0.65),
            "turnover": random.uniform(0.1, 0.3),
        },
        "stability": {
            "time_stability": random.uniform(0.6, 0.9),
            "market_stability": random.uniform(0.5, 0.85),
            "regime_stability": random.uniform(0.55, 0.8),
        },
        "passed_threshold": True,
        "evaluation_date": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_strategy_config() -> dict[str, Any]:
    """样本策略配置"""
    return {
        "strategy_id": "test_multi_factor_strategy",
        "strategy_name": "多因子动量策略",
        "factors": [
            {"id": "momentum_20d", "weight": 0.4},
            {"id": "volatility_10d", "weight": 0.3},
            {"id": "volume_ratio", "weight": 0.3},
        ],
        "position_sizing": {
            "method": "equal_weight",
            "max_position_pct": 0.1,
            "max_leverage": 2.0,
        },
        "risk_management": {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "max_drawdown_pct": 0.15,
            "trailing_stop": True,
        },
        "trading": {
            "universe": ["BTC", "ETH", "SOL", "AVAX", "MATIC"],
            "rebalance_frequency": "daily",
            "execution_delay": 1,
        },
    }


@pytest.fixture
def sample_backtest_result() -> dict[str, Any]:
    """样本回测结果"""
    import random

    random.seed(42)

    return {
        "strategy_id": "test_multi_factor_strategy",
        "backtest_id": "bt_20240101_20240401",
        "period": {
            "start": "2024-01-01",
            "end": "2024-04-01",
        },
        "performance": {
            "total_return": random.uniform(0.1, 0.4),
            "annualized_return": random.uniform(0.4, 1.2),
            "sharpe_ratio": random.uniform(1.5, 3.0),
            "max_drawdown": random.uniform(-0.12, -0.05),
            "win_rate": random.uniform(0.52, 0.62),
            "profit_factor": random.uniform(1.3, 2.0),
            "avg_trade_return": random.uniform(0.005, 0.015),
            "trade_count": random.randint(50, 150),
        },
        "risk_metrics": {
            "volatility": random.uniform(0.15, 0.3),
            "var_95": random.uniform(-0.03, -0.015),
            "cvar_95": random.uniform(-0.04, -0.02),
            "beta": random.uniform(0.6, 1.2),
            "alpha": random.uniform(0.05, 0.2),
        },
        "execution_stats": {
            "total_trades": random.randint(50, 150),
            "long_trades": random.randint(25, 80),
            "short_trades": random.randint(20, 70),
            "avg_holding_period_hours": random.uniform(12, 72),
            "slippage_cost": random.uniform(0.001, 0.005),
        },
    }


# ==================== Mock Fixtures ====================


@pytest.fixture
def mock_llm_provider():
    """模拟 LLM Provider"""
    with patch("iqfmp.llm.provider.LLMProvider") as mock:
        instance = MagicMock()
        instance.generate.return_value = {
            "content": "def factor(df): return df['close'].pct_change(20)",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_qdrant_client():
    """模拟 Qdrant 客户端"""
    with patch("iqfmp.vector.client.QdrantClient") as mock:
        instance = MagicMock()
        instance.health_check.return_value = True
        instance.collection_exists.return_value = True
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_redis_client():
    """模拟 Redis 客户端"""
    with patch("redis.Redis") as mock:
        instance = MagicMock()
        instance.ping.return_value = True
        instance.get.return_value = None
        instance.set.return_value = True
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_exchange_adapter():
    """模拟交易所适配器"""

    class MockExchangeAdapter:
        def __init__(self):
            self.connected = True
            self.positions = {}
            self.orders = []

        async def get_balance(self) -> dict[str, float]:
            return {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0}

        async def get_positions(self) -> list[dict]:
            return list(self.positions.values())

        async def create_order(
            self,
            symbol: str,
            side: str,
            order_type: str,
            amount: float,
            price: float | None = None,
        ) -> dict:
            order = {
                "id": f"order_{len(self.orders)}",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": amount,
                "price": price,
                "status": "filled",
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.orders.append(order)
            return order

        async def close_position(self, symbol: str) -> bool:
            if symbol in self.positions:
                del self.positions[symbol]
            return True

    return MockExchangeAdapter()


# ==================== 数据 Fixtures ====================


@pytest.fixture
def factor_families() -> list[dict[str, Any]]:
    """因子家族定义"""
    return [
        {
            "id": "momentum",
            "name": "动量因子",
            "description": "基于价格动量的因子",
            "allowed_fields": ["open", "high", "low", "close", "volume", "returns"],
        },
        {
            "id": "volatility",
            "name": "波动率因子",
            "description": "基于价格波动的因子",
            "allowed_fields": ["high", "low", "close", "volume", "atr"],
        },
        {
            "id": "volume",
            "name": "成交量因子",
            "description": "基于成交量的因子",
            "allowed_fields": ["volume", "close", "vwap", "turnover"],
        },
        {
            "id": "funding",
            "name": "资金费率因子",
            "description": "加密货币特有因子",
            "allowed_fields": ["funding_rate", "open_interest", "basis"],
        },
        {
            "id": "sentiment",
            "name": "情绪因子",
            "description": "市场情绪指标",
            "allowed_fields": ["long_short_ratio", "taker_buy_ratio", "fear_greed"],
        },
        {
            "id": "technical",
            "name": "技术指标因子",
            "description": "传统技术分析指标",
            "allowed_fields": ["close", "high", "low", "volume", "ma", "rsi", "macd"],
        },
    ]


@pytest.fixture
def evaluation_thresholds() -> dict[str, float]:
    """评估阈值配置"""
    return {
        "min_ic": 0.02,
        "min_ir": 0.3,
        "min_sharpe": 1.0,
        "max_drawdown": -0.2,
        "min_stability": 0.6,
        "similarity_threshold": 0.85,
    }


# ==================== 测试环境 Fixtures ====================


@pytest.fixture(scope="session")
def test_env():
    """测试环境配置"""
    return {
        "env": "test",
        "debug": True,
        "log_level": "DEBUG",
        "db_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",
        "qdrant_url": "http://localhost:6333",
    }


@pytest.fixture
def clean_test_state():
    """清理测试状态"""
    yield
    # 测试后清理逻辑（如果需要）


@pytest.fixture
def capture_logs(caplog):
    """捕获日志"""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog
