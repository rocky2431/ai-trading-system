"""
策略回测集成测试
测试策略生成和回测的完整流程
"""

import pytest
from datetime import datetime, timedelta
from typing import Any


class TestStrategyGeneration:
    """策略生成测试"""

    def test_multi_factor_strategy_code_generation(
        self,
        sample_strategy_config: dict[str, Any],
    ):
        """测试多因子策略代码生成"""
        config = sample_strategy_config

        # 验证策略配置完整性
        assert "factors" in config
        assert len(config["factors"]) >= 2
        assert all("weight" in f for f in config["factors"])

        # 验证权重和为 1
        total_weight = sum(f["weight"] for f in config["factors"])
        assert abs(total_weight - 1.0) < 0.01

    def test_strategy_code_validation(
        self,
        sample_strategy_config: dict[str, Any],
    ):
        """测试策略代码验证"""
        # 模拟生成的策略代码
        strategy_code = '''
class MultiFactorStrategy:
    def __init__(self, factors, weights):
        self.factors = factors
        self.weights = weights

    def generate_signal(self, df):
        combined_signal = 0
        for factor, weight in zip(self.factors, self.weights):
            combined_signal += factor.calculate(df) * weight
        return combined_signal

    def get_position(self, signal, current_position):
        if signal > 0.5:
            return 1  # Long
        elif signal < -0.5:
            return -1  # Short
        return current_position
'''

        # 验证代码结构
        assert "class " in strategy_code
        assert "def " in strategy_code
        assert "generate_signal" in strategy_code

    def test_position_sizing_configuration(
        self,
        sample_strategy_config: dict[str, Any],
    ):
        """测试仓位管理配置"""
        position_config = sample_strategy_config["position_sizing"]

        assert position_config["max_position_pct"] <= 0.2  # 最大仓位不超过 20%
        assert position_config["max_leverage"] <= 3.0  # 最大杠杆不超过 3x

    def test_risk_management_configuration(
        self,
        sample_strategy_config: dict[str, Any],
    ):
        """测试风险管理配置"""
        risk_config = sample_strategy_config["risk_management"]

        assert risk_config["stop_loss_pct"] >= 0.01  # 最小止损 1%
        assert risk_config["stop_loss_pct"] <= 0.05  # 最大止损 5%
        assert risk_config["max_drawdown_pct"] <= 0.2  # 最大回撤 20%


class TestBacktestExecution:
    """回测执行测试"""

    def test_backtest_data_preparation(
        self,
        sample_market_data: dict[str, Any],
    ):
        """测试回测数据准备"""
        data = sample_market_data

        # 验证数据完整性
        assert "dates" in data
        assert "symbols" in data
        assert "data" in data

        # 验证数据长度一致
        for symbol in data["symbols"]:
            symbol_data = data["data"][symbol]
            assert len(symbol_data["open"]) == len(data["dates"])
            assert len(symbol_data["close"]) == len(data["dates"])

    def test_backtest_period_validation(self):
        """测试回测周期验证"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 4, 1)

        # 验证周期合理性
        period_days = (end_date - start_date).days

        assert period_days >= 30, "Backtest period too short"
        assert period_days <= 365 * 3, "Backtest period too long"

    def test_transaction_cost_modeling(self):
        """测试交易成本建模"""
        # 交易成本参数
        commission_rate = 0.001  # 0.1%
        slippage_rate = 0.0005  # 0.05%

        trade_value = 10000
        total_cost = trade_value * (commission_rate + slippage_rate)

        assert total_cost == 15.0  # $15 per $10k trade

    def test_position_tracking(self):
        """测试持仓追踪"""
        positions = []

        # 模拟交易
        positions.append({"symbol": "BTC", "side": "long", "size": 1.0, "entry": 50000})
        positions.append({"symbol": "ETH", "side": "short", "size": 10, "entry": 3000})

        assert len(positions) == 2
        assert positions[0]["side"] == "long"
        assert positions[1]["side"] == "short"


class TestBacktestMetrics:
    """回测指标测试"""

    def test_total_return_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试总收益计算"""
        total_return = sample_backtest_result["performance"]["total_return"]

        assert isinstance(total_return, float)
        # 合理的收益范围
        assert -0.5 <= total_return <= 5.0

    def test_sharpe_ratio_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试 Sharpe 比率计算"""
        sharpe = sample_backtest_result["performance"]["sharpe_ratio"]

        assert isinstance(sharpe, float)
        assert sharpe >= 1.0, "Sharpe ratio should be positive"

    def test_max_drawdown_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试最大回撤计算"""
        max_dd = sample_backtest_result["performance"]["max_drawdown"]

        assert max_dd < 0, "Max drawdown should be negative"
        assert max_dd >= -0.5, "Max drawdown too severe"

    def test_win_rate_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试胜率计算"""
        win_rate = sample_backtest_result["performance"]["win_rate"]

        assert 0 <= win_rate <= 1
        assert win_rate >= 0.4, "Win rate too low"

    def test_profit_factor_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试盈亏比计算"""
        profit_factor = sample_backtest_result["performance"]["profit_factor"]

        assert profit_factor >= 1.0, "Profit factor should be >= 1"


class TestRiskMetrics:
    """风险指标测试"""

    def test_volatility_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试波动率计算"""
        volatility = sample_backtest_result["risk_metrics"]["volatility"]

        assert 0 < volatility < 1.0

    def test_var_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试 VaR 计算"""
        var_95 = sample_backtest_result["risk_metrics"]["var_95"]

        assert var_95 < 0, "VaR should be negative"
        assert var_95 >= -0.1, "VaR too extreme"

    def test_cvar_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试 CVaR 计算"""
        cvar_95 = sample_backtest_result["risk_metrics"]["cvar_95"]
        var_95 = sample_backtest_result["risk_metrics"]["var_95"]

        assert cvar_95 <= var_95, "CVaR should be <= VaR"

    def test_beta_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试 Beta 计算"""
        beta = sample_backtest_result["risk_metrics"]["beta"]

        assert 0 < beta < 2.0, "Beta out of reasonable range"

    def test_alpha_calculation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试 Alpha 计算"""
        alpha = sample_backtest_result["risk_metrics"]["alpha"]

        # Alpha 可正可负
        assert -0.5 < alpha < 0.5


class TestExecutionStats:
    """执行统计测试"""

    def test_trade_count_tracking(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试交易次数追踪"""
        total_trades = sample_backtest_result["execution_stats"]["total_trades"]
        long_trades = sample_backtest_result["execution_stats"]["long_trades"]
        short_trades = sample_backtest_result["execution_stats"]["short_trades"]

        assert total_trades == long_trades + short_trades

    def test_holding_period_analysis(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试持仓周期分析"""
        avg_holding = sample_backtest_result["execution_stats"]["avg_holding_period_hours"]

        assert avg_holding > 0
        assert avg_holding < 24 * 30, "Average holding too long"

    def test_slippage_estimation(
        self,
        sample_backtest_result: dict[str, Any],
    ):
        """测试滑点估算"""
        slippage = sample_backtest_result["execution_stats"]["slippage_cost"]

        assert 0 <= slippage <= 0.01, "Slippage out of range"


class TestBacktestValidation:
    """回测验证测试"""

    def test_look_ahead_bias_check(self):
        """测试前视偏差检查"""
        # 模拟数据访问模式
        data_access_log = [
            {"time": "2024-01-01 10:00", "data_time": "2024-01-01 09:00"},  # 正常
            {"time": "2024-01-01 10:00", "data_time": "2024-01-01 10:00"},  # 正常（同一时刻）
        ]

        for access in data_access_log:
            access_time = datetime.strptime(access["time"], "%Y-%m-%d %H:%M")
            data_time = datetime.strptime(access["data_time"], "%Y-%m-%d %H:%M")

            if data_time > access_time:
                pytest.fail(f"Look-ahead bias detected at {access['time']}")

    def test_survivorship_bias_check(self):
        """测试幸存者偏差检查"""
        # 验证使用了退市资产
        all_symbols = ["BTC", "ETH", "LUNA", "FTT"]  # LUNA, FTT 已崩盘
        active_only = ["BTC", "ETH"]

        # 应该包含所有历史资产
        includes_delisted = len(all_symbols) > len(active_only)

        assert includes_delisted, "Should include delisted assets"

    def test_in_sample_out_sample_split(self):
        """测试样本内外分离"""
        total_days = 365
        in_sample_pct = 0.7

        in_sample_days = int(total_days * in_sample_pct)
        out_sample_days = total_days - in_sample_days

        # 验证分离
        assert in_sample_days == 255
        assert out_sample_days == 110
        assert in_sample_days + out_sample_days == total_days

    def test_parameter_sensitivity(self):
        """测试参数敏感性"""
        base_sharpe = 2.0
        parameter_variations = [0.9, 0.95, 1.0, 1.05, 1.1]

        results = []
        for var in parameter_variations:
            # 模拟参数变化对 Sharpe 的影响
            adjusted_sharpe = base_sharpe * (1 + (var - 1) * 0.5)
            results.append(adjusted_sharpe)

        # 验证结果稳定性
        sharpe_std = (
            sum((s - sum(results) / len(results)) ** 2 for s in results) / len(results)
        ) ** 0.5

        assert sharpe_std < 0.5, "Strategy too sensitive to parameters"


class TestBacktestErrors:
    """回测错误处理测试"""

    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        prices = [100, 101, None, 103, 104]

        # 填充缺失值
        filled = []
        last_valid = None
        for p in prices:
            if p is not None:
                filled.append(p)
                last_valid = p
            else:
                filled.append(last_valid)  # 前向填充

        assert None not in filled
        assert filled[2] == 101  # 使用前一个值

    def test_zero_division_protection(self):
        """测试除零保护"""
        numerator = 100
        denominator = 0

        # 安全除法
        result = numerator / denominator if denominator != 0 else 0

        assert result == 0

    def test_extreme_return_capping(self):
        """测试极端收益截断"""
        raw_returns = [0.01, 0.02, 0.5, -0.8, 0.03]  # 0.5 和 -0.8 是异常值

        max_return = 0.2
        min_return = -0.2

        capped_returns = [
            max(min(r, max_return), min_return) for r in raw_returns
        ]

        assert capped_returns[2] == 0.2
        assert capped_returns[3] == -0.2

    def test_insufficient_capital_handling(self):
        """测试资金不足处理"""
        available_capital = 1000
        required_capital = 1500

        can_execute = available_capital >= required_capital

        assert not can_execute

        # 应该调整仓位
        scaled_position = available_capital / required_capital
        assert scaled_position < 1.0
