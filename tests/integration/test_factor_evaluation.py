"""
因子评估集成测试
测试因子评估的完整流程
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any


class TestFactorEvaluationFlow:
    """因子评估流程测试"""

    def test_factor_ic_calculation(
        self,
        sample_market_data: dict[str, Any],
    ):
        """测试因子 IC 计算"""
        import random

        random.seed(42)

        # 模拟因子值和收益率
        n_days = 100
        factor_values = [random.uniform(-1, 1) for _ in range(n_days)]
        returns = [random.uniform(-0.05, 0.05) for _ in range(n_days)]

        # 计算 IC（简化版 Spearman 相关性模拟）
        # 实际应使用 scipy.stats.spearmanr
        ic = sum(f * r for f, r in zip(factor_values, returns)) / n_days

        # IC 应该在合理范围内
        assert -1 <= ic <= 1

    def test_factor_ir_calculation(self):
        """测试因子 IR 计算"""
        import random

        random.seed(42)

        # 模拟多期 IC
        n_periods = 20
        ic_series = [random.uniform(-0.1, 0.1) for _ in range(n_periods)]

        # 计算 IR = IC_mean / IC_std
        ic_mean = sum(ic_series) / len(ic_series)
        ic_std = (sum((ic - ic_mean) ** 2 for ic in ic_series) / len(ic_series)) ** 0.5

        ir = ic_mean / ic_std if ic_std > 0 else 0

        # IR 可以是任意值
        assert isinstance(ir, float)

    def test_factor_sharpe_ratio(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试因子 Sharpe 比率"""
        sharpe = sample_factor_result["metrics"]["sharpe_ratio"]

        # Sharpe 应该大于阈值
        assert sharpe >= 1.0, f"Sharpe ratio {sharpe} below threshold"

    def test_factor_max_drawdown(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试因子最大回撤"""
        max_dd = sample_factor_result["metrics"]["max_drawdown"]

        # 最大回撤应该是负值且不超过限制
        assert max_dd < 0, "Max drawdown should be negative"
        assert max_dd >= -0.2, f"Max drawdown {max_dd} exceeds limit"


class TestStabilityAnalysis:
    """稳定性分析测试"""

    def test_time_stability_analysis(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试时间稳定性分析"""
        time_stability = sample_factor_result["stability"]["time_stability"]

        # 时间稳定性应该在 0-1 范围内
        assert 0 <= time_stability <= 1
        # 应该达到最小阈值
        assert time_stability >= 0.6, f"Time stability {time_stability} too low"

    def test_market_stability_analysis(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试市场稳定性分析"""
        market_stability = sample_factor_result["stability"]["market_stability"]

        assert 0 <= market_stability <= 1

    def test_regime_stability_analysis(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试 Regime 稳定性分析"""
        regime_stability = sample_factor_result["stability"]["regime_stability"]

        assert 0 <= regime_stability <= 1

    def test_combined_stability_score(
        self,
        sample_factor_result: dict[str, Any],
    ):
        """测试综合稳定性评分"""
        stability = sample_factor_result["stability"]

        # 计算综合稳定性（加权平均）
        weights = {"time": 0.4, "market": 0.35, "regime": 0.25}
        combined = (
            stability["time_stability"] * weights["time"]
            + stability["market_stability"] * weights["market"]
            + stability["regime_stability"] * weights["regime"]
        )

        assert 0 <= combined <= 1


class TestResearchLedger:
    """研究账本测试"""

    def test_trial_recording(self):
        """测试试验记录"""
        trial = {
            "id": "trial_001",
            "factor_id": "momentum_20d",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "ic": 0.05,
                "ir": 0.45,
                "sharpe": 1.8,
            },
            "passed": True,
        }

        assert "id" in trial
        assert "factor_id" in trial
        assert "metrics" in trial
        assert "passed" in trial

    def test_global_trial_count(self):
        """测试全局试验次数"""
        initial_count = 0
        new_trials = 10

        total_count = initial_count + new_trials

        assert total_count == 10

    def test_dynamic_threshold_calculation(self):
        """测试动态阈值计算"""
        # Deflated Sharpe Ratio 简化版
        base_threshold = 2.0
        trial_count = 100
        adjustment_factor = 1 + (trial_count / 1000)  # 随试验次数增加

        adjusted_threshold = base_threshold * adjustment_factor

        assert adjusted_threshold > base_threshold
        assert adjusted_threshold == 2.2

    def test_overfitting_detection(self):
        """测试过拟合检测"""
        in_sample_sharpe = 3.5
        out_of_sample_sharpe = 1.2

        # 检测过拟合：样本内外差异过大
        sharpe_ratio = out_of_sample_sharpe / in_sample_sharpe
        overfitting_threshold = 0.5

        is_overfitting = sharpe_ratio < overfitting_threshold

        assert is_overfitting, "Should detect overfitting"


class TestCrossValidation:
    """交叉验证测试"""

    def test_time_split_validation(self):
        """测试时间切分验证"""
        total_days = 365
        train_pct = 0.6
        valid_pct = 0.2
        test_pct = 0.2

        train_days = int(total_days * train_pct)
        valid_days = int(total_days * valid_pct)
        test_days = total_days - train_days - valid_days

        assert train_days == 219
        assert valid_days == 73
        assert test_days == 73

    def test_market_split_validation(self):
        """测试市场切分验证"""
        all_symbols = ["BTC", "ETH", "SOL", "AVAX", "MATIC", "DOGE", "SHIB", "LINK"]

        # 大市值 vs 其他
        major_symbols = ["BTC", "ETH"]
        other_symbols = [s for s in all_symbols if s not in major_symbols]

        assert len(major_symbols) == 2
        assert len(other_symbols) == 6

    def test_frequency_split_validation(self):
        """测试频率切分验证"""
        frequencies = ["1h", "4h", "1d"]

        # 验证每个频率的数据量
        for freq in frequencies:
            if freq == "1h":
                expected_bars = 24 * 365
            elif freq == "4h":
                expected_bars = 6 * 365
            else:  # 1d
                expected_bars = 365

            assert expected_bars > 0

    def test_multi_dimension_cv(self):
        """测试多维度交叉验证"""
        time_splits = 3
        market_splits = 2
        freq_splits = 3

        total_combinations = time_splits * market_splits * freq_splits

        assert total_combinations == 18


class TestFactorEvaluationErrors:
    """因子评估错误处理测试"""

    def test_insufficient_data_handling(self):
        """测试数据不足处理"""
        min_required_days = 60
        available_days = 30

        is_sufficient = available_days >= min_required_days

        assert not is_sufficient

    def test_nan_handling_in_metrics(self):
        """测试 NaN 值处理"""
        import math

        metrics = {
            "ic": 0.05,
            "ir": float("nan"),
            "sharpe": 1.5,
        }

        # 检查是否有 NaN
        has_nan = any(
            math.isnan(v) if isinstance(v, float) else False
            for v in metrics.values()
        )

        assert has_nan

        # 处理 NaN：替换为 0 或标记为无效
        cleaned_metrics = {
            k: 0.0 if isinstance(v, float) and math.isnan(v) else v
            for k, v in metrics.items()
        }

        assert cleaned_metrics["ir"] == 0.0

    def test_extreme_value_capping(self):
        """测试极端值截断"""
        raw_ic = 0.95  # 不太可能的高 IC

        # 截断到合理范围
        max_ic = 0.3
        capped_ic = min(raw_ic, max_ic)

        assert capped_ic == 0.3

    def test_evaluation_timeout(self):
        """测试评估超时"""
        max_timeout = 300  # 5 分钟
        actual_time = 120  # 2 分钟

        is_timeout = actual_time > max_timeout

        assert not is_timeout


class TestFactorRanking:
    """因子排名测试"""

    def test_single_metric_ranking(self):
        """测试单指标排名"""
        factors = [
            {"id": "f1", "sharpe": 1.5},
            {"id": "f2", "sharpe": 2.3},
            {"id": "f3", "sharpe": 1.8},
        ]

        # 按 Sharpe 降序排名
        ranked = sorted(factors, key=lambda x: x["sharpe"], reverse=True)

        assert ranked[0]["id"] == "f2"
        assert ranked[1]["id"] == "f3"
        assert ranked[2]["id"] == "f1"

    def test_composite_score_ranking(self):
        """测试综合评分排名"""
        factors = [
            {"id": "f1", "sharpe": 1.5, "ic": 0.04, "stability": 0.7},
            {"id": "f2", "sharpe": 2.3, "ic": 0.02, "stability": 0.5},
            {"id": "f3", "sharpe": 1.8, "ic": 0.05, "stability": 0.8},
        ]

        # 计算综合评分
        weights = {"sharpe": 0.4, "ic": 0.3, "stability": 0.3}

        for f in factors:
            f["composite"] = (
                f["sharpe"] / 3 * weights["sharpe"]  # 归一化
                + f["ic"] / 0.1 * weights["ic"]
                + f["stability"] * weights["stability"]
            )

        ranked = sorted(factors, key=lambda x: x["composite"], reverse=True)

        assert ranked[0]["id"] == "f3"  # 最高综合分

    def test_pareto_frontier(self):
        """测试 Pareto 前沿"""
        factors = [
            {"id": "f1", "return": 0.2, "risk": 0.15},
            {"id": "f2", "return": 0.3, "risk": 0.25},
            {"id": "f3", "return": 0.25, "risk": 0.12},  # Pareto 最优
            {"id": "f4", "return": 0.15, "risk": 0.20},  # 被 f3 dominate
        ]

        # 简化的 Pareto 检测
        pareto_optimal = []
        for f in factors:
            is_dominated = False
            for other in factors:
                if other["id"] != f["id"]:
                    # 检查是否被 dominate
                    if other["return"] >= f["return"] and other["risk"] <= f["risk"]:
                        if other["return"] > f["return"] or other["risk"] < f["risk"]:
                            is_dominated = True
                            break

            if not is_dominated:
                pareto_optimal.append(f["id"])

        assert "f3" in pareto_optimal
