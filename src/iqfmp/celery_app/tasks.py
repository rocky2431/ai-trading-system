"""
Celery 任务定义
包含回测、因子评估、因子生成等异步任务
支持重试机制和状态追踪
"""

import logging
from datetime import datetime
from typing import Any
from celery import shared_task
from celery.exceptions import MaxRetriesExceededError

from .app import celery_app

logger = logging.getLogger(__name__)


class TaskError(Exception):
    """任务执行错误"""
    pass


class RetryableError(Exception):
    """可重试的错误"""
    pass


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.backtest_task",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(RetryableError,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    acks_late=True,
    track_started=True,
    priority=3,
)
def backtest_task(
    self,
    strategy_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    策略回测任务

    Args:
        strategy_id: 策略 ID
        config: 回测配置
            - start_date: 开始日期
            - end_date: 结束日期
            - initial_capital: 初始资金
            - symbols: 交易标的列表
            - frequency: 数据频率

    Returns:
        回测结果
            - task_id: 任务 ID
            - strategy_id: 策略 ID
            - metrics: 绩效指标
            - trades: 交易记录
    """
    task_id = self.request.id
    logger.info(f"Starting backtest task {task_id} for strategy {strategy_id}")

    try:
        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": 100,
                "status": "Initializing backtest...",
                "started_at": datetime.utcnow().isoformat(),
            },
        )

        # 模拟回测执行过程
        # 实际实现中会调用 strategy.backtest 模块
        result = _execute_backtest(self, strategy_id, config)

        logger.info(f"Backtest task {task_id} completed successfully")
        return {
            "task_id": task_id,
            "strategy_id": strategy_id,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": result,
        }

    except RetryableError as e:
        logger.warning(f"Backtest task {task_id} failed with retryable error: {e}")
        raise self.retry(exc=e)

    except MaxRetriesExceededError:
        logger.error(f"Backtest task {task_id} exceeded max retries")
        return {
            "task_id": task_id,
            "strategy_id": strategy_id,
            "status": "failed",
            "error": "Max retries exceeded",
        }

    except Exception as e:
        logger.error(f"Backtest task {task_id} failed: {e}")
        return {
            "task_id": task_id,
            "strategy_id": strategy_id,
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.evaluate_factor_task",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(RetryableError,),
    retry_backoff=True,
    acks_late=True,
    track_started=True,
    priority=5,
)
def evaluate_factor_task(
    self,
    factor_id: str,
    factor_code: str,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """
    因子评估任务

    Args:
        factor_id: 因子 ID
        factor_code: 因子代码
        evaluation_config: 评估配置
            - cv_splits: 交叉验证切分数
            - metrics: 评估指标列表

    Returns:
        评估结果
            - factor_id: 因子 ID
            - metrics: IC/IR/Sharpe 等指标
            - stability: 稳定性分析
    """
    task_id = self.request.id
    logger.info(f"Starting factor evaluation task {task_id} for factor {factor_id}")

    try:
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": 100,
                "status": "Loading factor data...",
                "started_at": datetime.utcnow().isoformat(),
            },
        )

        # 模拟因子评估过程
        result = _execute_factor_evaluation(self, factor_id, factor_code, evaluation_config)

        logger.info(f"Factor evaluation task {task_id} completed")
        return {
            "task_id": task_id,
            "factor_id": factor_id,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": result,
        }

    except RetryableError as e:
        logger.warning(f"Factor evaluation task {task_id} failed with retryable error: {e}")
        raise self.retry(exc=e)

    except Exception as e:
        logger.error(f"Factor evaluation task {task_id} failed: {e}")
        return {
            "task_id": task_id,
            "factor_id": factor_id,
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.generate_factor_task",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    track_started=True,
    priority=5,
)
def generate_factor_task(
    self,
    hypothesis: str,
    factor_family: str,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """
    因子生成任务

    Args:
        hypothesis: 因子假设描述
        factor_family: 因子家族
        generation_config: 生成配置

    Returns:
        生成结果
            - factor_id: 新生成的因子 ID
            - code: 因子代码
            - metadata: 因子元数据
    """
    task_id = self.request.id
    logger.info(f"Starting factor generation task {task_id}")

    try:
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": 100,
                "status": "Calling LLM to generate factor...",
                "started_at": datetime.utcnow().isoformat(),
            },
        )

        # 模拟因子生成过程
        result = _execute_factor_generation(self, hypothesis, factor_family, generation_config)

        logger.info(f"Factor generation task {task_id} completed")
        return {
            "task_id": task_id,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": result,
        }

    except Exception as e:
        logger.error(f"Factor generation task {task_id} failed: {e}")
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.batch_backtest_task",
    max_retries=1,
    acks_late=True,
    track_started=True,
    priority=2,
)
def batch_backtest_task(
    self,
    strategies: list[dict[str, Any]],
    common_config: dict[str, Any],
) -> dict[str, Any]:
    """
    批量回测任务

    Args:
        strategies: 策略列表
        common_config: 通用配置

    Returns:
        批量回测结果
    """
    task_id = self.request.id
    logger.info(f"Starting batch backtest task {task_id} for {len(strategies)} strategies")

    results = []
    total = len(strategies)

    for i, strategy in enumerate(strategies):
        self.update_state(
            state="PROGRESS",
            meta={
                "current": i,
                "total": total,
                "status": f"Running backtest {i + 1}/{total}...",
            },
        )

        try:
            result = _execute_backtest(self, strategy["id"], {**common_config, **strategy.get("config", {})})
            results.append({
                "strategy_id": strategy["id"],
                "status": "completed",
                "result": result,
            })
        except Exception as e:
            results.append({
                "strategy_id": strategy["id"],
                "status": "failed",
                "error": str(e),
            })

    return {
        "task_id": task_id,
        "status": "completed",
        "total": total,
        "successful": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.risk_check_task",
    max_retries=0,
    acks_late=False,
    track_started=True,
    priority=9,
)
def risk_check_task(
    self,
    account_id: str,
    check_config: dict[str, Any],
) -> dict[str, Any]:
    """
    风控检查任务（高优先级）

    Args:
        account_id: 账户 ID
        check_config: 检查配置

    Returns:
        风控检查结果
    """
    task_id = self.request.id
    logger.info(f"Running risk check task {task_id} for account {account_id}")

    # 模拟风控检查
    return {
        "task_id": task_id,
        "account_id": account_id,
        "status": "completed",
        "risk_level": "normal",
        "checks": {
            "margin_usage": {"status": "ok", "value": 45.2},
            "max_drawdown": {"status": "ok", "value": 3.5},
            "position_concentration": {"status": "ok", "value": 42.5},
        },
    }


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.emergency_close_task",
    max_retries=0,
    acks_late=False,
    track_started=True,
    priority=10,
)
def emergency_close_task(
    self,
    account_id: str,
    positions: list[str],
    reason: str,
) -> dict[str, Any]:
    """
    紧急平仓任务（最高优先级）

    Args:
        account_id: 账户 ID
        positions: 持仓 ID 列表
        reason: 平仓原因

    Returns:
        平仓结果
    """
    task_id = self.request.id
    logger.warning(f"EMERGENCY: Running emergency close task {task_id} for account {account_id}")
    logger.warning(f"Reason: {reason}")
    logger.warning(f"Positions to close: {positions}")

    # 模拟紧急平仓
    return {
        "task_id": task_id,
        "account_id": account_id,
        "status": "completed",
        "closed_positions": positions,
        "reason": reason,
        "completed_at": datetime.utcnow().isoformat(),
    }


# 辅助函数

def _execute_backtest(task, strategy_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """执行回测的内部实现 - 使用真实回测引擎"""
    from iqfmp.core.backtest_engine import BacktestEngine
    from iqfmp.core.factor_engine import BUILTIN_FACTORS

    # 阶段1: 初始化
    task.update_state(
        state="PROGRESS",
        meta={"current": 10, "total": 100, "status": "Loading market data..."},
    )

    # 获取因子代码 (从配置或使用默认)
    factor_code = config.get("factor_code")
    if not factor_code:
        # 根据策略类型选择内置因子
        factor_name = config.get("factor_name", "momentum_20d")
        factor_code = BUILTIN_FACTORS.get(factor_name, BUILTIN_FACTORS["momentum_20d"])

    # 阶段2: 加载数据
    task.update_state(
        state="PROGRESS",
        meta={"current": 30, "total": 100, "status": "Initializing backtest engine..."},
    )

    try:
        engine = BacktestEngine()

        # 阶段3: 执行回测
        task.update_state(
            state="PROGRESS",
            meta={"current": 50, "total": 100, "status": "Running backtest simulation..."},
        )

        result = engine.run_factor_backtest(
            factor_code=factor_code,
            initial_capital=config.get("initial_capital", 100000.0),
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            rebalance_frequency=config.get("rebalance_frequency", "daily"),
            position_size=config.get("position_size", 1.0),
            long_only=config.get("long_only", False),
        )

        # 阶段4: 处理结果
        task.update_state(
            state="PROGRESS",
            meta={"current": 90, "total": 100, "status": "Processing results..."},
        )

        return {
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "win_rate": result.win_rate,
            "total_trades": result.trade_count,
            "profit_factor": result.profit_factor,
            "calmar_ratio": result.calmar_ratio,
            "volatility": result.volatility,
            "avg_trade_return": result.avg_trade_return,
            "avg_holding_period": result.avg_holding_period,
            "monthly_returns": result.monthly_returns,
        }

    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        raise TaskError(f"Backtest failed: {e}")


def _execute_factor_evaluation(
    task,
    factor_id: str,
    factor_code: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """执行因子评估的内部实现 - 使用真实因子引擎"""
    from iqfmp.core.factor_engine import FactorEngine, FactorEvaluator, create_engine_with_sample_data
    from iqfmp.evaluation.research_ledger import DynamicThreshold, ThresholdConfig

    # 阶段1: 加载数据
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Loading market data..."},
    )

    try:
        # 创建引擎并加载样本数据
        engine = create_engine_with_sample_data()

        # 阶段2: 计算因子值
        task.update_state(
            state="PROGRESS",
            meta={"current": 40, "total": 100, "status": "Computing factor values..."},
        )

        factor_values = engine.compute_factor(factor_code, factor_name=factor_id)

        # 阶段3: 评估因子
        task.update_state(
            state="PROGRESS",
            meta={"current": 60, "total": 100, "status": "Evaluating factor metrics..."},
        )

        evaluator = FactorEvaluator(engine)
        eval_result = evaluator.evaluate(
            factor_values=factor_values,
            forward_periods=config.get("forward_periods", [1, 5, 10]),
        )

        # 阶段4: 计算动态阈值
        task.update_state(
            state="PROGRESS",
            meta={"current": 80, "total": 100, "status": "Checking threshold..."},
        )

        # 获取当前试验数计算阈值
        n_trials = config.get("n_trials", 1)
        threshold_calc = DynamicThreshold(ThresholdConfig())
        current_threshold = threshold_calc.calculate(n_trials)

        metrics = eval_result["metrics"]
        sharpe = metrics.get("sharpe", 0.0)
        passed_threshold = sharpe >= current_threshold

        # 阶段5: 汇总结果
        task.update_state(
            state="PROGRESS",
            meta={"current": 95, "total": 100, "status": "Finalizing results..."},
        )

        # 计算稳定性评分
        stability = eval_result.get("stability", {})
        time_stability = stability.get("time_stability", {})
        market_stability = stability.get("market_stability", {})
        stability_score = (
            abs(time_stability.get("monthly_ic_stability", 0)) +
            abs(market_stability.get("overall", 0))
        ) / 2

        return {
            "ic": metrics.get("ic_mean", 0.0),
            "ic_std": metrics.get("ic_std", 0.0),
            "ic_ir": metrics.get("ir", 0.0),
            "sharpe": sharpe,
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "turnover": metrics.get("turnover", 0.0),
            "ic_by_split": metrics.get("ic_by_split", {}),
            "sharpe_by_split": metrics.get("sharpe_by_split", {}),
            "stability_score": round(stability_score, 4),
            "threshold_used": round(current_threshold, 4),
            "passed_threshold": passed_threshold,
        }

    except Exception as e:
        logger.error(f"Factor evaluation failed: {e}")
        raise TaskError(f"Factor evaluation failed: {e}")


def _execute_factor_generation(
    task,
    hypothesis: str,
    factor_family: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """执行因子生成的内部实现"""
    import time
    import uuid

    task.update_state(
        state="PROGRESS",
        meta={"current": 30, "total": 100, "status": "Generating factor code..."},
    )
    time.sleep(0.2)

    task.update_state(
        state="PROGRESS",
        meta={"current": 60, "total": 100, "status": "Running safety checks..."},
    )
    time.sleep(0.1)

    task.update_state(
        state="PROGRESS",
        meta={"current": 90, "total": 100, "status": "Saving factor..."},
    )
    time.sleep(0.1)

    return {
        "factor_id": str(uuid.uuid4()),
        "hypothesis": hypothesis,
        "family": factor_family,
        "code": f"# Generated factor for: {hypothesis}\ndef calculate(data):\n    return data['close'].pct_change()",
    }
