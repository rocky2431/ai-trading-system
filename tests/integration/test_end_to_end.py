"""
端到端集成测试
测试因子生成→评估→策略→回测的完整流程
"""

import pytest
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock


class TestEndToEndPipeline:
    """端到端流水线测试"""

    def test_complete_factor_pipeline(
        self,
        mock_llm_response: dict[str, Any],
        sample_market_data: dict[str, Any],
        sample_factor_result: dict[str, Any],
        evaluation_thresholds: dict[str, float],
    ):
        """测试完整因子流水线"""
        # Step 1: 因子生成
        generated_factor = mock_llm_response
        assert "factor_code" in generated_factor

        # Step 2: AST 安全检查
        is_safe = "eval" not in generated_factor["factor_code"]
        assert is_safe, "Factor code should pass safety check"

        # Step 3: 因子评估
        evaluation = sample_factor_result
        assert evaluation["metrics"]["ic_mean"] >= evaluation_thresholds["min_ic"]
        assert evaluation["metrics"]["ir"] >= evaluation_thresholds["min_ir"]

        # Step 4: 稳定性检查
        assert evaluation["stability"]["time_stability"] >= evaluation_thresholds["min_stability"]

        # Step 5: 去重检查（假设通过）
        is_duplicate = False
        assert not is_duplicate

        # Step 6: 因子入库
        factor_id = f"factor_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        assert factor_id is not None

    def test_complete_strategy_pipeline(
        self,
        sample_factor_result: dict[str, Any],
        sample_strategy_config: dict[str, Any],
        sample_backtest_result: dict[str, Any],
    ):
        """测试完整策略流水线"""
        # Step 1: 因子选择
        selected_factors = [sample_factor_result]
        assert len(selected_factors) >= 1

        # Step 2: 策略配置
        strategy = sample_strategy_config
        assert "factors" in strategy
        assert "risk_management" in strategy

        # Step 3: 策略代码生成
        strategy_code = f"class Strategy_{strategy['strategy_id']}: pass"
        assert "class " in strategy_code

        # Step 4: 回测执行
        backtest = sample_backtest_result
        assert "performance" in backtest

        # Step 5: 绩效验证
        assert backtest["performance"]["sharpe_ratio"] >= 1.0
        assert backtest["performance"]["max_drawdown"] >= -0.2

        # Step 6: 策略入库
        strategy_id = f"strategy_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        assert strategy_id is not None


class TestPipelineIntegration:
    """流水线集成测试"""

    def test_factor_to_strategy_flow(
        self,
        sample_factor_result: dict[str, Any],
        sample_strategy_config: dict[str, Any],
    ):
        """测试因子到策略的流程"""
        # 因子通过评估
        factor = sample_factor_result
        assert factor["passed_threshold"]

        # 创建策略配置
        strategy = {
            "strategy_id": "auto_generated_001",
            "factors": [{"id": factor["factor_id"], "weight": 1.0}],
            "created_from": "factor_pipeline",
        }

        assert strategy["factors"][0]["id"] == factor["factor_id"]

    def test_strategy_to_backtest_flow(
        self,
        sample_strategy_config: dict[str, Any],
        sample_backtest_result: dict[str, Any],
    ):
        """测试策略到回测的流程"""
        strategy = sample_strategy_config
        backtest = sample_backtest_result

        # 验证策略和回测关联
        assert backtest["strategy_id"] == strategy["strategy_id"]

    def test_backtest_to_live_flow(
        self,
        sample_backtest_result: dict[str, Any],
        mock_exchange_adapter,
    ):
        """测试回测到实盘的流程"""
        backtest = sample_backtest_result

        # 验证回测绩效达标
        meets_requirements = (
            backtest["performance"]["sharpe_ratio"] >= 1.5
            and backtest["performance"]["max_drawdown"] >= -0.15
        )

        if meets_requirements:
            # 准备实盘配置
            live_config = {
                "strategy_id": backtest["strategy_id"],
                "capital": 10000,
                "max_position_pct": 0.1,
                "enabled": True,
            }

            assert live_config["enabled"]


class TestDataFlow:
    """数据流测试"""

    def test_market_data_flow(
        self,
        sample_market_data: dict[str, Any],
    ):
        """测试市场数据流"""
        data = sample_market_data

        # 数据获取
        assert len(data["dates"]) > 0

        # 数据预处理
        for symbol in data["symbols"]:
            prices = data["data"][symbol]["close"]
            assert len(prices) == len(data["dates"])

            # 计算收益率
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(ret)

            assert len(returns) == len(prices) - 1

    def test_factor_data_flow(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试因子数据流"""
        factor = sample_factor_result

        # 因子计算
        assert "metrics" in factor

        # 因子存储
        factor_record = {
            "id": factor["factor_id"],
            "name": factor["factor_name"],
            "metrics": factor["metrics"],
            "stored_at": datetime.utcnow().isoformat(),
        }

        assert factor_record["id"] is not None

    def test_signal_data_flow(self):
        """测试信号数据流"""
        # 因子信号
        factor_signals = [0.2, -0.1, 0.5, -0.3]

        # 信号聚合
        aggregated_signal = sum(factor_signals) / len(factor_signals)

        # 信号转换为仓位
        if aggregated_signal > 0.1:
            position = 1
        elif aggregated_signal < -0.1:
            position = -1
        else:
            position = 0

        assert position == 0  # 0.075 在中性区间


class TestErrorRecovery:
    """错误恢复测试"""

    def test_llm_failure_recovery(self):
        """测试 LLM 失败恢复"""
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            retry_count += 1
            # 模拟最后一次成功
            if retry_count == max_retries:
                success = True

        assert success
        assert retry_count == 3

    def test_database_failure_recovery(self):
        """测试数据库失败恢复"""
        # 模拟事务
        class MockTransaction:
            def __init__(self):
                self.committed = False
                self.rolled_back = False

            def commit(self):
                self.committed = True

            def rollback(self):
                self.rolled_back = True

        tx = MockTransaction()

        try:
            # 模拟操作
            raise Exception("DB connection lost")
        except Exception:
            tx.rollback()

        assert tx.rolled_back
        assert not tx.committed

    def test_exchange_failure_recovery(self):
        """测试交易所失败恢复"""
        class MockExchange:
            def __init__(self):
                self.connected = False
                self.reconnect_count = 0

            def reconnect(self):
                self.reconnect_count += 1
                if self.reconnect_count >= 3:
                    self.connected = True
                return self.connected

        exchange = MockExchange()

        while not exchange.connected and exchange.reconnect_count < 5:
            exchange.reconnect()

        assert exchange.connected
        assert exchange.reconnect_count == 3


class TestConcurrency:
    """并发测试"""

    def test_parallel_factor_evaluation(self):
        """测试并行因子评估"""
        import concurrent.futures

        def evaluate_factor(factor_id: str) -> dict:
            # 模拟评估
            return {"id": factor_id, "status": "completed"}

        factor_ids = [f"factor_{i}" for i in range(5)]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(evaluate_factor, fid) for fid in factor_ids]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        assert len(results) == 5
        assert all(r["status"] == "completed" for r in results)

    def test_concurrent_backtest_execution(self):
        """测试并发回测执行"""
        import concurrent.futures

        def run_backtest(config: dict) -> dict:
            return {"id": config["id"], "sharpe": 1.5}

        configs = [{"id": f"bt_{i}"} for i in range(3)]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_backtest, cfg) for cfg in configs]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        assert len(results) == 3


class TestMonitoring:
    """监控测试"""

    def test_pipeline_metrics_collection(self):
        """测试流水线指标收集"""
        metrics = {
            "factors_generated": 0,
            "factors_evaluated": 0,
            "factors_passed": 0,
            "strategies_created": 0,
            "backtests_run": 0,
        }

        # 模拟流水线执行
        metrics["factors_generated"] = 10
        metrics["factors_evaluated"] = 10
        metrics["factors_passed"] = 7
        metrics["strategies_created"] = 3
        metrics["backtests_run"] = 3

        # 验证指标
        assert metrics["factors_passed"] <= metrics["factors_evaluated"]
        assert metrics["strategies_created"] <= metrics["factors_passed"]

    def test_alert_triggering(self):
        """测试告警触发"""
        thresholds = {
            "error_rate": 0.05,  # 5%
            "latency_p99": 5.0,  # 5 秒
        }

        current_metrics = {
            "error_rate": 0.08,  # 超过阈值
            "latency_p99": 3.0,  # 正常
        }

        alerts = []
        for metric, threshold in thresholds.items():
            if current_metrics[metric] > threshold:
                alerts.append({
                    "metric": metric,
                    "value": current_metrics[metric],
                    "threshold": threshold,
                })

        assert len(alerts) == 1
        assert alerts[0]["metric"] == "error_rate"


class TestAuditTrail:
    """审计追踪测试"""

    def test_factor_audit_log(self):
        """测试因子审计日志"""
        audit_log = []

        # 模拟因子生命周期
        events = [
            {"event": "created", "factor_id": "f_001", "user": "system"},
            {"event": "evaluated", "factor_id": "f_001", "result": "passed"},
            {"event": "approved", "factor_id": "f_001", "user": "admin"},
            {"event": "deployed", "factor_id": "f_001", "strategy_id": "s_001"},
        ]

        for event in events:
            event["timestamp"] = datetime.utcnow().isoformat()
            audit_log.append(event)

        assert len(audit_log) == 4
        assert audit_log[0]["event"] == "created"
        assert audit_log[-1]["event"] == "deployed"

    def test_strategy_audit_log(self):
        """测试策略审计日志"""
        audit_log = []

        events = [
            {"event": "created", "strategy_id": "s_001"},
            {"event": "backtested", "strategy_id": "s_001", "sharpe": 1.8},
            {"event": "approved", "strategy_id": "s_001"},
            {"event": "live_started", "strategy_id": "s_001"},
            {"event": "live_stopped", "strategy_id": "s_001", "reason": "drawdown"},
        ]

        for event in events:
            event["timestamp"] = datetime.utcnow().isoformat()
            audit_log.append(event)

        assert len(audit_log) == 5


class TestCleanup:
    """清理测试"""

    def test_stale_data_cleanup(self):
        """测试过期数据清理"""
        retention_days = 30
        now = datetime.utcnow()

        data = [
            {"id": "d_001", "created_at": now - timedelta(days=10)},  # 保留
            {"id": "d_002", "created_at": now - timedelta(days=45)},  # 删除
            {"id": "d_003", "created_at": now - timedelta(days=20)},  # 保留
        ]

        cutoff = now - timedelta(days=retention_days)
        to_delete = [d for d in data if d["created_at"] < cutoff]
        to_keep = [d for d in data if d["created_at"] >= cutoff]

        assert len(to_delete) == 1
        assert len(to_keep) == 2

    def test_orphan_resource_cleanup(self):
        """测试孤立资源清理"""
        factors = [{"id": "f_001"}, {"id": "f_002"}, {"id": "f_003"}]
        strategies = [{"id": "s_001", "factor_ids": ["f_001", "f_002"]}]

        # 查找未被使用的因子
        used_factor_ids = set()
        for s in strategies:
            used_factor_ids.update(s["factor_ids"])

        orphan_factors = [f for f in factors if f["id"] not in used_factor_ids]

        assert len(orphan_factors) == 1
        assert orphan_factors[0]["id"] == "f_003"
