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
    风控检查任务（高优先级）- 使用真实风控引擎

    使用 RiskController 执行四层风险检查：
    1. 回撤检查 - MAX_DRAWDOWN_THRESHOLD (15%)
    2. 持仓集中度检查 - MAX_POSITION_RATIO (30%)
    3. 杠杆检查 - MAX_LEVERAGE (3x)
    4. 单日亏损检查 - EMERGENCY_LOSS_THRESHOLD (5%)

    Args:
        account_id: 账户 ID
        check_config: 检查配置
            - equity: 当前权益
            - peak_equity: 峰值权益
            - total_position_value: 总仓位价值
            - daily_pnl: 当日盈亏
            - positions: 持仓列表 [{"symbol": str, "value": float}]

    Returns:
        风控检查结果
    """
    from decimal import Decimal
    from iqfmp.exchange.risk import (
        Account,
        Position,
        RiskConfig,
        RiskController,
        RiskLevel,
    )

    task_id = self.request.id
    logger.info(f"Running risk check task {task_id} for account {account_id}")

    # 从 check_config 解析账户数据
    equity = Decimal(str(check_config.get("equity", 100000)))
    peak_equity = Decimal(str(check_config.get("peak_equity", equity)))
    total_position_value = Decimal(str(check_config.get("total_position_value", 0)))
    daily_pnl = Decimal(str(check_config.get("daily_pnl", 0)))
    positions_data = check_config.get("positions", [])

    # 构建 Account 对象
    account = Account(
        equity=equity,
        total_position_value=total_position_value,
        daily_pnl=daily_pnl,
        peak_equity=peak_equity,
    )

    # 初始化风控控制器
    risk_config = RiskConfig()
    controller = RiskController(config=risk_config, initial_equity=peak_equity)
    controller.update_equity(equity)

    # 添加持仓到控制器
    for pos in positions_data:
        symbol = pos.get("symbol", "UNKNOWN")
        value = Decimal(str(pos.get("value", 0)))
        controller.add_position(symbol, value)

    # 执行风控检查 - 逐个持仓检查
    all_violations = []
    checks_result = {}

    # 检查每个持仓
    for pos in positions_data:
        symbol = pos.get("symbol", "UNKNOWN")
        value = Decimal(str(pos.get("value", 0)))
        position = Position(symbol=symbol, value=value)
        result = controller.check_risk_sync(position, account)
        all_violations.extend(result.violations)

    # 汇总检查结果
    # 1. 回撤检查
    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else Decimal("0")
    drawdown_threshold = RiskController.MAX_DRAWDOWN_THRESHOLD
    drawdown_ok = drawdown <= drawdown_threshold
    checks_result["max_drawdown"] = {
        "status": "ok" if drawdown_ok else "breach",
        "value": float(drawdown * 100),
        "threshold": float(drawdown_threshold * 100),
        "message": f"回撤 {drawdown * 100:.2f}% {'正常' if drawdown_ok else '超限'}",
    }

    # 2. 杠杆检查
    leverage = total_position_value / equity if equity > 0 else Decimal("0")
    leverage_threshold = RiskController.MAX_LEVERAGE
    leverage_ok = leverage <= leverage_threshold
    checks_result["leverage"] = {
        "status": "ok" if leverage_ok else "breach",
        "value": float(leverage),
        "threshold": float(leverage_threshold),
        "message": f"杠杆 {leverage:.2f}x {'正常' if leverage_ok else '超限'}",
    }

    # 3. 单日亏损检查
    if daily_pnl < 0 and equity > 0:
        daily_loss_ratio = -daily_pnl / equity
    else:
        daily_loss_ratio = Decimal("0")
    daily_loss_threshold = RiskController.EMERGENCY_LOSS_THRESHOLD
    daily_loss_ok = daily_loss_ratio <= daily_loss_threshold
    checks_result["daily_loss"] = {
        "status": "ok" if daily_loss_ok else "breach",
        "value": float(daily_loss_ratio * 100),
        "threshold": float(daily_loss_threshold * 100),
        "message": f"单日亏损 {daily_loss_ratio * 100:.2f}% {'正常' if daily_loss_ok else '超限'}",
    }

    # 4. 持仓集中度检查
    max_concentration = Decimal("0")
    concentration_threshold = RiskController.MAX_POSITION_RATIO
    for pos in positions_data:
        value = Decimal(str(pos.get("value", 0)))
        if equity > 0:
            conc = value / equity
            if conc > max_concentration:
                max_concentration = conc
    concentration_ok = max_concentration <= concentration_threshold
    checks_result["position_concentration"] = {
        "status": "ok" if concentration_ok else "breach",
        "value": float(max_concentration * 100),
        "threshold": float(concentration_threshold * 100),
        "message": f"最大持仓集中度 {max_concentration * 100:.2f}% {'正常' if concentration_ok else '超限'}",
    }

    # 确定整体风险等级
    is_safe = all(c["status"] == "ok" for c in checks_result.values())
    has_critical = any(v.severity == "critical" for v in all_violations)
    has_high = any(v.severity == "high" for v in all_violations)

    if has_critical:
        risk_level = "critical"
        recommended_action = "emergency_close_all"
    elif has_high:
        risk_level = "danger"
        recommended_action = "reduce_position"
    elif not is_safe:
        risk_level = "warning"
        recommended_action = "monitor"
    else:
        risk_level = "normal"
        recommended_action = "none"

    logger.info(f"Risk check completed: level={risk_level}, violations={len(all_violations)}")

    return {
        "task_id": task_id,
        "account_id": account_id,
        "status": "completed",
        "risk_level": risk_level,
        "is_safe": is_safe,
        "recommended_action": recommended_action,
        "checks": checks_result,
        "violations": [
            {
                "type": v.type,
                "severity": v.severity,
                "action": v.action,
                "message": v.message,
                "current_value": float(v.current_value),
                "threshold": float(v.threshold),
            }
            for v in all_violations
        ],
        "thresholds": {
            k: float(v) for k, v in RiskController.get_hard_thresholds().items()
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
    紧急平仓任务（最高优先级）- 调用真实交易所 API

    通过 ccxt 连接交易所执行紧急平仓操作。
    如果交易所凭证未配置，将返回错误并发送告警通知。

    Args:
        account_id: 账户 ID
        positions: 持仓数据列表，每个元素可以是:
            - 字符串: "BTCUSDT" (仅符号)
            - 字典: {"symbol": "BTCUSDT", "side": "long", "quantity": 0.1}
        reason: 平仓原因

    Returns:
        平仓结果
    """
    import asyncio
    import os
    from decimal import Decimal

    task_id = self.request.id
    logger.warning(f"EMERGENCY: Running emergency close task {task_id} for account {account_id}")
    logger.warning(f"Reason: {reason}")
    logger.warning(f"Positions to close: {positions}")

    close_results = []
    telegram_sent = False

    # 尝试发送 Telegram 告警
    async def _send_telegram_alert() -> bool:
        """发送 Telegram 紧急告警"""
        try:
            from iqfmp.exchange.emergency import TelegramNotifier

            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")

            if not bot_token or not chat_id:
                logger.warning("Telegram credentials not configured")
                return False

            notifier = TelegramNotifier(
                bot_token=bot_token,
                chat_id=chat_id,
                enabled=True,
            )

            # 发送紧急告警
            message = (
                f"🚨 <b>紧急平仓执行中</b>\n\n"
                f"<b>账户</b>: {account_id}\n"
                f"<b>原因</b>: {reason}\n"
                f"<b>持仓数量</b>: {len(positions)}\n"
                f"<b>任务ID</b>: {task_id}"
            )
            return await notifier._send_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    try:
        telegram_sent = asyncio.run(_send_telegram_alert())
    except Exception as e:
        logger.error(f"Telegram alert error: {e}")

    # 检查交易所凭证
    exchange_type = os.getenv("EXCHANGE_TYPE", "binance")
    api_key = os.getenv("EXCHANGE_API_KEY", os.getenv("BINANCE_API_KEY"))
    api_secret = os.getenv("EXCHANGE_API_SECRET", os.getenv("BINANCE_API_SECRET"))

    if not api_key or not api_secret:
        logger.error("Exchange credentials not configured - cannot execute real close orders")
        return {
            "task_id": task_id,
            "account_id": account_id,
            "status": "failed",
            "error": "Exchange credentials not configured. Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET.",
            "positions_to_close": positions,
            "reason": reason,
            "telegram_sent": telegram_sent,
            "completed_at": datetime.utcnow().isoformat(),
        }

    # 执行真实平仓
    async def _execute_emergency_close() -> list[dict]:
        """异步执行紧急平仓"""
        results = []

        try:
            # 动态导入 ccxt
            import ccxt.async_support as ccxt_async

            # 创建交易所连接
            exchange_class = getattr(ccxt_async, exchange_type, ccxt_async.binance)
            exchange = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "sandbox": os.getenv("EXCHANGE_SANDBOX", "false").lower() == "true",
                "options": {"defaultType": "future"},
            })

            try:
                # 加载市场信息
                await exchange.load_markets()

                for pos_data in positions:
                    # 解析持仓数据
                    if isinstance(pos_data, str):
                        symbol = pos_data
                        quantity = None
                        side = None
                    else:
                        symbol = pos_data.get("symbol", "")
                        quantity = pos_data.get("quantity")
                        side = pos_data.get("side")

                    if not symbol:
                        results.append({
                            "symbol": "UNKNOWN",
                            "status": "failed",
                            "error": "Invalid position data",
                        })
                        continue

                    try:
                        # 获取当前持仓
                        positions_info = await exchange.fetch_positions([symbol])
                        position = next((p for p in positions_info if p["symbol"] == symbol and float(p.get("contracts", 0)) != 0), None)

                        if not position:
                            results.append({
                                "symbol": symbol,
                                "status": "skipped",
                                "message": "No open position found",
                            })
                            continue

                        # 确定平仓方向和数量
                        pos_side = position.get("side", "long")
                        pos_contracts = abs(float(position.get("contracts", 0)))

                        if quantity:
                            close_quantity = min(float(quantity), pos_contracts)
                        else:
                            close_quantity = pos_contracts

                        # 执行市价平仓
                        order_side = "sell" if pos_side == "long" else "buy"
                        order = await exchange.create_market_order(
                            symbol=symbol,
                            side=order_side,
                            amount=close_quantity,
                            params={"reduceOnly": True},
                        )

                        results.append({
                            "symbol": symbol,
                            "status": "success",
                            "order_id": order.get("id"),
                            "side": order_side,
                            "quantity": close_quantity,
                            "average_price": order.get("average"),
                            "filled": order.get("filled"),
                        })
                        logger.info(f"Successfully closed {symbol}: {order}")

                    except Exception as e:
                        logger.error(f"Failed to close {symbol}: {e}")
                        results.append({
                            "symbol": symbol,
                            "status": "failed",
                            "error": str(e),
                        })

            finally:
                await exchange.close()

        except ImportError:
            logger.error("ccxt not installed - cannot execute exchange orders")
            return [{"status": "failed", "error": "ccxt not installed"}]
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
            return [{"status": "failed", "error": str(e)}]

        return results

    try:
        close_results = asyncio.run(_execute_emergency_close())
    except Exception as e:
        logger.error(f"Emergency close execution failed: {e}")
        close_results = [{"status": "failed", "error": str(e)}]

    # 统计结果
    success_count = sum(1 for r in close_results if r.get("status") == "success")
    failed_count = sum(1 for r in close_results if r.get("status") == "failed")
    skipped_count = sum(1 for r in close_results if r.get("status") == "skipped")

    # 发送结果通知
    async def _send_result_notification() -> bool:
        try:
            from iqfmp.exchange.emergency import TelegramNotifier

            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")

            if not bot_token or not chat_id:
                return False

            notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id, enabled=True)

            status_icon = "✅" if failed_count == 0 else "⚠️"
            message = (
                f"{status_icon} <b>紧急平仓完成</b>\n\n"
                f"<b>成功</b>: {success_count}\n"
                f"<b>失败</b>: {failed_count}\n"
                f"<b>跳过</b>: {skipped_count}\n"
                f"<b>任务ID</b>: {task_id}"
            )
            return await notifier._send_message(message)
        except Exception:
            return False

    try:
        asyncio.run(_send_result_notification())
    except Exception:
        pass

    logger.warning(f"Emergency close completed: success={success_count}, failed={failed_count}, skipped={skipped_count}")

    return {
        "task_id": task_id,
        "account_id": account_id,
        "status": "completed" if failed_count == 0 else "partial",
        "reason": reason,
        "results": close_results,
        "summary": {
            "total": len(positions),
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
        },
        "telegram_sent": telegram_sent,
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
    from datetime import datetime as _dt
    from iqfmp.core.factor_engine import FactorEngine, FactorEvaluator, create_engine_with_sample_data
    from iqfmp.core.data_provider import load_ohlcv_sync
    from iqfmp.evaluation.research_ledger import DynamicThreshold, ThresholdConfig

    # 阶段1: 加载数据
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Loading market data..."},
    )

    try:
        # 创建引擎并加载真实数据（优先 data_config）
        data_cfg = config.get("data_config") or {}
        symbol = (data_cfg.get("symbols") or ["ETHUSDT"])[0]
        timeframe = (data_cfg.get("timeframes") or ["1d"])[0]
        start_date = data_cfg.get("start_date")
        end_date = data_cfg.get("end_date")

        df = None
        try:
            df = load_ohlcv_sync(symbol=symbol, timeframe=timeframe, use_db=True)
            if start_date and end_date:
                try:
                    start_dt = _dt.fromisoformat(start_date)
                    end_dt = _dt.fromisoformat(end_date)
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=None)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=None)
                    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
                except Exception as sub_e:  # noqa: BLE001
                    logger.warning(f"Date filter failed, using full dataset: {sub_e}")

            # 如果提供时间区间，尝试加载含衍生品的统一数据集
            if start_date and end_date:
                try:
                    import asyncio
                    from iqfmp.data.provider import UnifiedMarketDataProvider, DataLoadConfig
                    from iqfmp.db.database import get_async_session

                    start_dt = _dt.fromisoformat(start_date)
                    end_dt = _dt.fromisoformat(end_date)

                    async def _load_unified():
                        async with get_async_session() as async_session:
                            provider = UnifiedMarketDataProvider(
                                async_session,
                                exchange=data_cfg.get("exchange", "binance"),
                            )
                            result = await provider.load_market_data(
                                symbol=symbol,
                                start_date=start_dt,
                                end_date=end_dt,
                                timeframe=timeframe,
                                config=DataLoadConfig(
                                    include_derivatives=True,
                                    market_type=data_cfg.get("market_type", "futures"),
                                ),
                            )
                            return result.df

                    unified_df = asyncio.run(_load_unified())
                    if unified_df is not None and len(unified_df) > 0:
                        df = unified_df
                        logger.info("Loaded unified market data with derivatives for evaluation")
                except Exception as sub_e:  # noqa: BLE001
                    logger.warning(f"Unified derivative data load failed, fallback to OHLCV only: {sub_e}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Real data load failed, fallback to sample data: {e}")
            df = None

        if df is not None and not df.empty:
            engine = FactorEngine(df=df)
        else:
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
    """执行因子生成的内部实现 - 调用真实 LLM

    使用 FactorGenerationAgent 通过 OpenRouter API 生成真实的因子代码。
    包含 AST 安全检查、字段约束验证和**错误反馈循环**。

    错误反馈循环机制：
    1. 初次生成因子
    2. 如果生成或计算失败，将错误信息反馈给 LLM
    3. LLM 根据错误信息生成改进版本
    4. 最多重试 max_retries 次
    """
    import asyncio
    import uuid

    from iqfmp.agents.factor_generation import (
        FactorFamily,
        FactorGenerationAgent,
        FactorGenerationConfig,
        FactorGenerationError,
        InvalidFactorError,
    )
    from iqfmp.llm.provider import LLMConfig, LLMProvider

    task.update_state(
        state="PROGRESS",
        meta={"current": 10, "total": 100, "status": "Initializing LLM provider..."},
    )

    # 优先从 ConfigService 配置文件读取配置（用户在前端配置的）
    # Celery worker 是独立进程，无法继承 FastAPI 进程的环境变量
    # 因此需要直接从配置文件 ~/.iqfmp/config.json 读取
    # 注意：每次任务执行都重新读取，确保前端修改能及时生效
    import json
    from pathlib import Path

    config_file = Path.home() / ".iqfmp" / "config.json"
    api_key_from_config = None
    if config_file.exists():
        try:
            with open(config_file) as f:
                saved_config = json.load(f)
                api_key_from_config = saved_config.get("api_key")
        except Exception as e:
            logger.warning(f"Failed to read config file: {e}")

    # 如果从配置文件获取到 API key，设置到环境变量供 LLMConfig.from_env() 使用
    if api_key_from_config:
        import os
        os.environ["OPENROUTER_API_KEY"] = api_key_from_config
        logger.info("Loaded API key from config file ~/.iqfmp/config.json")

    # 强制重新加载 AgentModelRegistry，确保获取前端最新的模型配置
    # 这样用户在前端切换模型后，下次任务会使用新模型
    from iqfmp.agents.model_config import reload_model_registry
    reload_model_registry()
    logger.info("Reloaded agent model registry from config file")

    # 尝试从环境变量加载 LLM 配置
    try:
        llm_config = LLMConfig.from_env()
    except ValueError as e:
        # 如果没有配置 API key，记录警告并返回降级结果
        logger.warning(f"LLM config error: {e}. Factor generation will use fallback.")
        return {
            "factor_id": str(uuid.uuid4()),
            "hypothesis": hypothesis,
            "family": factor_family,
            "code": f"# LLM not configured - placeholder factor\n# Hypothesis: {hypothesis}\ndef calculate(data):\n    return data['close'].pct_change()",
            "warning": "LLM not configured. Please configure OpenRouter API key in Settings page.",
        }

    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Connecting to LLM..."},
    )

    # 映射 factor_family 字符串到枚举
    family_map = {f.value: f for f in FactorFamily}
    factor_family_enum = family_map.get(factor_family.lower(), FactorFamily.MOMENTUM)

    # 创建 agent 配置
    max_retries = config.get("max_retries", 5)
    agent_config = FactorGenerationConfig(
        name="celery_factor_generator",
        security_check_enabled=True,
        field_constraint_enabled=True,
        max_retries=max_retries,
        include_examples=True,
    )

    async def _generate_with_feedback_loop() -> dict[str, Any]:
        """异步生成因子，包含智能指标反馈循环

        Intelligent Feedback Loop:
        1. Parse user hypothesis to extract requested indicators
        2. Generate factor expression
        3. Check if all indicators are implemented
        4. If missing, provide specific feedback and retry
        5. Continue until complete or max retries reached
        """
        # Import indicator intelligence module
        from iqfmp.agents.indicator_intelligence import (
            check_factor_completeness,
            get_indicator_summary,
            analyze_indicator_coverage,
        )

        async with LLMProvider(llm_config) as llm:
            agent = FactorGenerationAgent(config=agent_config, llm_provider=llm)

            last_code = None
            last_error = None

            # Log detected indicators from hypothesis
            indicator_summary = get_indicator_summary(hypothesis)
            logger.info(f"Factor generation - {indicator_summary}")

            for attempt in range(max_retries):
                try:
                    task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": 30 + attempt * 15,
                            "total": 100,
                            "status": f"Generating factor (attempt {attempt + 1}/{max_retries})...",
                            "attempt": attempt + 1,
                            "indicator_summary": indicator_summary,
                        },
                    )

                    if attempt == 0:
                        # 首次尝试：正常生成
                        generated = await agent.generate(
                            user_request=hypothesis,
                            factor_family=factor_family_enum,
                        )
                    else:
                        # 后续尝试：使用错误反馈进行改进
                        logger.info(f"Refining factor (attempt {attempt + 1}): {last_error[:200] if last_error else 'No error'}...")
                        task.update_state(
                            state="PROGRESS",
                            meta={
                                "current": 30 + attempt * 15,
                                "total": 100,
                                "status": f"Refining factor based on feedback (attempt {attempt + 1}/{max_retries})...",
                                "attempt": attempt + 1,
                                "is_refining": True,
                            },
                        )
                        generated = await agent.refine(
                            original_code=last_code,
                            error_message=last_error,
                            user_request=hypothesis,
                            factor_family=factor_family_enum,
                        )

                    # 验证生成的因子可以计算（快速检查）
                    try:
                        _validate_factor_computes(generated.code)
                    except Exception as compute_error:
                        # 因子生成成功但计算失败，保存错误信息用于下次改进
                        last_code = generated.code
                        last_error = f"Factor computation failed: {str(compute_error)}"
                        logger.warning(f"Factor computation validation failed: {compute_error}")
                        if attempt < max_retries - 1:
                            continue
                        raise

                    # ===== 智能指标完整性检查 =====
                    # Check if all requested indicators are implemented
                    is_complete, missing_feedback = check_factor_completeness(
                        hypothesis=hypothesis,
                        expression=generated.code,
                    )

                    if not is_complete and attempt < max_retries - 1:
                        # 指标不完整，提供具体反馈让LLM改进
                        analysis = analyze_indicator_coverage(hypothesis, generated.code)
                        logger.warning(
                            f"Indicator coverage incomplete: "
                            f"requested={analysis.requested}, "
                            f"found={analysis.found}, "
                            f"missing={analysis.missing}"
                        )
                        task.update_state(
                            state="PROGRESS",
                            meta={
                                "current": 30 + attempt * 15,
                                "total": 100,
                                "status": f"Missing indicators: {', '.join(analysis.missing)}. Refining...",
                                "attempt": attempt + 1,
                                "missing_indicators": list(analysis.missing),
                            },
                        )
                        last_code = generated.code
                        last_error = missing_feedback
                        continue  # Retry with specific feedback

                    # Log indicator analysis for successful generation
                    analysis = analyze_indicator_coverage(hypothesis, generated.code)
                    logger.info(
                        f"Factor generated with indicators: {analysis.found}, "
                        f"completion rate: {analysis.completion_rate:.0%}"
                    )

                    task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": 85,
                            "total": 100,
                            "status": "Factor validated successfully, saving...",
                        },
                    )

                    return {
                        "factor_id": str(uuid.uuid4()),
                        "hypothesis": hypothesis,
                        "family": generated.family.value,
                        "name": generated.name,
                        "description": generated.description,
                        "code": generated.code,
                        "metadata": {
                            **generated.metadata,
                            "attempts": attempt + 1,
                            "refined": attempt > 0,
                            "indicators_requested": list(analysis.requested),
                            "indicators_found": list(analysis.found),
                            "indicator_completion_rate": analysis.completion_rate,
                        },
                    }

                except (FactorGenerationError, InvalidFactorError) as e:
                    # 生成过程中的错误，保存用于下次改进
                    last_error = str(e)
                    last_code = getattr(e, 'code', last_code) or "# Previous code unavailable"
                    logger.warning(f"Factor generation attempt {attempt + 1} failed: {e}")
                    if attempt >= max_retries - 1:
                        raise

            # 不应该到达这里，但作为安全保障
            raise FactorGenerationError(f"Max retries ({max_retries}) exceeded. Last error: {last_error}")

    try:
        # 在 Celery 同步任务中运行异步代码
        result = asyncio.run(_generate_with_feedback_loop())
        logger.info(f"Factor generated successfully: {result.get('name')} (attempts: {result.get('metadata', {}).get('attempts', 1)})")
        return result

    except FactorGenerationError as e:
        logger.error(f"Factor generation failed after all retries: {e}")
        raise TaskError(f"Factor generation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in factor generation: {e}")
        raise TaskError(f"Factor generation error: {e}")


def _validate_factor_computes(code: str) -> bool:
    """Quick validation that factor expression can be computed.

    Args:
        code: Qlib expression or Python code

    Returns:
        True if valid

    Raises:
        Exception if validation fails
    """
    # For Qlib expressions, do a quick syntax check
    if "$" in code or any(op in code for op in ["Mean(", "Std(", "Ref(", "RSI(", "MACD("]):
        # Check for invalid fields
        invalid_fields = ["$returns", "$quote_volume", "$funding_rate", "$open_interest"]
        for invalid in invalid_fields:
            if invalid in code:
                raise ValueError(f"Invalid field used: {invalid}. Only $open, $high, $low, $close, $volume are allowed.")

        # Check for balanced parentheses
        paren_count = 0
        for char in code:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            if paren_count < 0:
                raise ValueError("Unbalanced parentheses: extra ')'")
        if paren_count != 0:
            raise ValueError("Unbalanced parentheses: missing ')'")

        return True

    # For Python code, try to parse it
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        raise ValueError(f"Python syntax error: {e}")


# ==================== Mining Task (C2 Fix: Persistent) ====================


@celery_app.task(
    bind=True,
    name="iqfmp.celery_app.tasks.mining_task",
    max_retries=1,
    default_retry_delay=60,
    acks_late=True,
    track_started=True,
    priority=5,
    queue="default",
)
def mining_task(
    self,
    task_id: str,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    """
    因子挖掘任务 - 持久化到 Redis (C2 Fix)

    此任务通过 Celery 执行，确保：
    1. 任务持久化到 Redis，服务重启后可恢复
    2. 支持任务取消检查
    3. 进度实时更新

    Args:
        task_id: 挖掘任务 ID
        task_config: 任务配置
            - name: 任务名称
            - target_count: 目标因子数量
            - factor_families: 因子家族列表
            - auto_evaluate: 是否自动评估

    Returns:
        挖掘结果
            - task_id: 任务 ID
            - status: 完成状态
            - generated_count: 生成的因子数
            - passed_count: 通过评估的因子数
            - failed_count: 失败的因子数
    """
    celery_task_id = self.request.id
    logger.info(f"Starting mining task {task_id} (Celery ID: {celery_task_id})")

    generated_count = 0
    passed_count = 0
    failed_count = 0

    try:
        # Update initial state
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": 100,
                "status": "Initializing mining task...",
                "started_at": datetime.utcnow().isoformat(),
                "task_id": task_id,
            },
        )

        target_count = task_config.get("target_count", 10)
        families = task_config.get("factor_families") or ["momentum", "volatility"]
        auto_evaluate = task_config.get("auto_evaluate", True)
        task_name = task_config.get("name", "Mining Task")
        # Use description as the primary hypothesis for factor generation
        # If no description, fall back to task_name
        task_description = task_config.get("description") or task_name

        logger.info(f"Mining task config: name='{task_name}', description='{task_description}'")

        # Run mining loop
        result = _execute_mining_task(
            celery_task=self,
            task_id=task_id,
            task_name=task_name,
            task_description=task_description,  # Pass description for factor generation
            target_count=target_count,
            families=families,
            auto_evaluate=auto_evaluate,
            data_config=task_config.get("data_config"),
        )

        logger.info(f"Mining task {task_id} completed: {result}")
        return {
            "task_id": task_id,
            "celery_task_id": celery_task_id,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            **result,
        }

    except Exception as e:
        logger.error(f"Mining task {task_id} failed: {e}")
        return {
            "task_id": task_id,
            "celery_task_id": celery_task_id,
            "status": "failed",
            "error": str(e),
            "generated_count": generated_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
        }


def _execute_mining_task(
    celery_task,
    task_id: str,
    task_name: str,
    task_description: str,
    target_count: int,
    families: list[str],
    auto_evaluate: bool,
    data_config: dict | None = None,
) -> dict[str, Any]:
    """执行因子挖掘的内部实现.

    此函数在 Celery Worker 中同步执行。
    使用数据库连接进行因子生成和评估。

    Args:
        task_description: 用户输入的因子描述，用作 LLM 的主要指令
    """
    import asyncio
    import time

    generated_count = 0
    passed_count = 0
    failed_count = 0

    # Cache redis client for cancellation checks (avoid reconnect per-iteration)
    redis_client = None
    try:
        import os
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)
    except Exception:
        redis_client = None

    for i in range(target_count):
        # Check for cancellation via Redis
        if redis_client is not None:
            try:
                if redis_client.sismember("mining_tasks:cancelled", task_id):
                    logger.info(f"Mining task {task_id} cancelled at iteration {i}")
                    return {
                        "generated_count": generated_count,
                        "passed_count": passed_count,
                        "failed_count": failed_count,
                        "cancelled": True,
                    }
            except Exception:
                pass  # Redis check failed, continue execution

        # Update progress
        progress = ((i + 1) / target_count) * 100
        celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": int(progress),
                "total": 100,
                "status": f"Generating factor {i + 1}/{target_count}...",
                "generated_count": generated_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
            },
        )

        # Select family for this iteration
        family = families[i % len(families)]

        # Use task_description as the primary factor generation instruction
        # If description contains specific indicators, use it directly
        indicator_keywords = ["macd", "rsi", "ema", "sma", "wr", "ssl", "zigzag", "ziggy",
                              "bollinger", "atr", "adx", "cci", "momentum", "结合", "combine",
                              "策略", "指标", "williams", "stochastic", "ichimoku", "趋势",
                              "均线", "突破", "反转", "超买", "超卖"]
        is_specific_request = any(keyword in task_description.lower() for keyword in indicator_keywords)

        logger.info(f"DEBUG task_description='{task_description}', is_specific={is_specific_request}")

        if is_specific_request:
            # User provided specific indicator description - use it directly as LLM instruction
            description = f"{task_description} (variant #{i+1}, family: {family})"
        else:
            # Generic description - use auto-generated prompt
            description = f"Auto-generated {family} factor #{i+1}: {task_description}"

        logger.info(f"DEBUG hypothesis='{description}'")

        try:
            # Generate factor using sync implementation
            factor_result = _execute_factor_generation(
                celery_task,
                hypothesis=description,
                factor_family=family,
                config={},
            )
            generated_count += 1

            # Auto-evaluate if enabled
            if auto_evaluate:
                try:
                    eval_result = _execute_factor_evaluation(
                        celery_task,
                        factor_id=factor_result.get("factor_id", "unknown"),
                        factor_code=factor_result.get("code", ""),
                        config={
                            "n_trials": generated_count,
                            "data_config": data_config or {},
                        },
                    )
                    sharpe = eval_result.get("sharpe", 0.0)
                    threshold = eval_result.get("threshold_used", 0.0)
                    passed = eval_result.get("passed_threshold", False)
                    logger.info(f"Factor evaluation: sharpe={sharpe:.4f}, threshold={threshold:.4f}, passed={passed}")

                    if passed:
                        passed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.warning(f"Factor evaluation failed with exception: {e}")
                    import traceback
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                    failed_count += 1

        except Exception as e:
            logger.error(f"Factor generation failed: {e}")
            failed_count += 1

        # Small delay to avoid overwhelming resources
        time.sleep(0.3)

    return {
        "generated_count": generated_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "cancelled": False,
    }
