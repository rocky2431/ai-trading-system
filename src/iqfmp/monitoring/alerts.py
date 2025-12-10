"""
告警管理器
提供告警规则配置和告警触发功能
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: str  # 条件表达式
    severity: AlertSeverity
    description: str
    threshold: float
    duration: int = 60  # 持续时间（秒）
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if "alert_name" not in self.labels:
            self.labels["alert_name"] = self.name


@dataclass
class Alert:
    """告警实例"""
    rule: AlertRule
    status: AlertStatus
    value: float
    started_at: datetime
    resolved_at: Optional[datetime] = None
    notified: bool = False

    @property
    def duration(self) -> float:
        """告警持续时间"""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.rule.name,
            "severity": self.rule.severity.value,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.rule.threshold,
            "description": self.rule.description,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration": self.duration,
            "labels": self.rule.labels,
            "annotations": self.rule.annotations,
        }


class AlertManager:
    """
    告警管理器
    处理告警规则评估和告警触发
    """

    def __init__(self):
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.notifiers: list[Callable[[Alert], None]] = []
        self._pending_alerts: dict[str, tuple[float, datetime]] = {}

        # 注册默认规则
        self._register_default_rules()

    def _register_default_rules(self):
        """注册默认告警规则"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_usage_percent > threshold",
                severity=AlertSeverity.WARNING,
                description="CPU 使用率过高",
                threshold=80.0,
                duration=300,
                labels={"component": "system"},
            ),
            AlertRule(
                name="critical_cpu_usage",
                condition="cpu_usage_percent > threshold",
                severity=AlertSeverity.CRITICAL,
                description="CPU 使用率严重过高",
                threshold=95.0,
                duration=60,
                labels={"component": "system"},
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage_percent > threshold",
                severity=AlertSeverity.WARNING,
                description="内存使用率过高",
                threshold=85.0,
                duration=300,
                labels={"component": "system"},
            ),
            AlertRule(
                name="high_error_rate",
                condition="error_rate > threshold",
                severity=AlertSeverity.ERROR,
                description="错误率过高",
                threshold=5.0,  # 5%
                duration=120,
                labels={"component": "application"},
            ),
            AlertRule(
                name="high_latency",
                condition="request_latency_p99 > threshold",
                severity=AlertSeverity.WARNING,
                description="请求延迟过高 (P99)",
                threshold=2.0,  # 2秒
                duration=180,
                labels={"component": "application"},
            ),
            AlertRule(
                name="critical_latency",
                condition="request_latency_p99 > threshold",
                severity=AlertSeverity.CRITICAL,
                description="请求延迟严重过高 (P99)",
                threshold=5.0,  # 5秒
                duration=60,
                labels={"component": "application"},
            ),
            AlertRule(
                name="task_queue_backlog",
                condition="task_queue_length > threshold",
                severity=AlertSeverity.WARNING,
                description="任务队列积压",
                threshold=100,
                duration=300,
                labels={"component": "celery"},
            ),
            AlertRule(
                name="high_position_risk",
                condition="position_risk_score > threshold",
                severity=AlertSeverity.ERROR,
                description="持仓风险过高",
                threshold=0.8,
                duration=60,
                labels={"component": "trading"},
            ),
            AlertRule(
                name="daily_loss_limit",
                condition="daily_pnl_loss > threshold",
                severity=AlertSeverity.CRITICAL,
                description="日亏损超过限制",
                threshold=10000.0,
                duration=0,  # 立即触发
                labels={"component": "trading"},
            ),
        ]

        for rule in default_rules:
            self.register_rule(rule)

    def register_rule(self, rule: AlertRule):
        """注册告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    def unregister_rule(self, name: str):
        """取消注册告警规则"""
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Unregistered alert rule: {name}")

    def add_notifier(self, notifier: Callable[[Alert], None]):
        """添加通知器"""
        self.notifiers.append(notifier)

    def evaluate_rule(
        self,
        rule_name: str,
        current_value: float,
    ) -> Optional[Alert]:
        """
        评估告警规则

        Args:
            rule_name: 规则名称
            current_value: 当前值

        Returns:
            如果触发告警则返回 Alert，否则返回 None
        """
        rule = self.rules.get(rule_name)
        if not rule:
            logger.warning(f"Alert rule not found: {rule_name}")
            return None

        now = datetime.utcnow()

        # 检查是否超过阈值
        exceeds_threshold = current_value > rule.threshold

        if exceeds_threshold:
            # 检查是否已经在待处理状态
            if rule_name in self._pending_alerts:
                pending_value, pending_start = self._pending_alerts[rule_name]
                elapsed = (now - pending_start).total_seconds()

                # 检查是否超过持续时间
                if elapsed >= rule.duration:
                    # 触发告警
                    alert = self._create_or_update_alert(rule, current_value, now)
                    del self._pending_alerts[rule_name]
                    return alert
            else:
                # 开始计时
                self._pending_alerts[rule_name] = (current_value, now)

                # 如果持续时间为 0，立即触发
                if rule.duration == 0:
                    alert = self._create_or_update_alert(rule, current_value, now)
                    del self._pending_alerts[rule_name]
                    return alert
        else:
            # 值恢复正常
            if rule_name in self._pending_alerts:
                del self._pending_alerts[rule_name]

            # 解决活跃告警
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = now
                del self.active_alerts[rule_name]
                self.alert_history.append(alert)
                logger.info(f"Alert resolved: {rule_name}")

                # 发送恢复通知
                self._notify(alert)

        return None

    def _create_or_update_alert(
        self,
        rule: AlertRule,
        value: float,
        timestamp: datetime,
    ) -> Alert:
        """创建或更新告警"""
        if rule.name in self.active_alerts:
            # 更新现有告警
            alert = self.active_alerts[rule.name]
            alert.value = value
            return alert

        # 创建新告警
        alert = Alert(
            rule=rule,
            status=AlertStatus.FIRING,
            value=value,
            started_at=timestamp,
        )

        self.active_alerts[rule.name] = alert
        logger.warning(
            f"Alert triggered: {rule.name} "
            f"(value={value:.2f}, threshold={rule.threshold:.2f})"
        )

        # 发送通知
        self._notify(alert)

        return alert

    def _notify(self, alert: Alert):
        """发送告警通知"""
        for notifier in self.notifiers:
            try:
                notifier(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")

        alert.notified = True

    def evaluate_all(self, metrics: dict[str, float]) -> list[Alert]:
        """
        评估所有规则

        Args:
            metrics: 指标字典

        Returns:
            触发的告警列表
        """
        triggered_alerts = []

        metric_mapping = {
            "high_cpu_usage": "cpu_usage_percent",
            "critical_cpu_usage": "cpu_usage_percent",
            "high_memory_usage": "memory_usage_percent",
            "high_error_rate": "error_rate",
            "high_latency": "request_latency_p99",
            "critical_latency": "request_latency_p99",
            "task_queue_backlog": "task_queue_length",
            "high_position_risk": "position_risk_score",
            "daily_loss_limit": "daily_pnl_loss",
        }

        for rule_name, metric_name in metric_mapping.items():
            if metric_name in metrics:
                alert = self.evaluate_rule(rule_name, metrics[metric_name])
                if alert:
                    triggered_alerts.append(alert)

        return triggered_alerts

    def get_active_alerts(self) -> list[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """获取告警历史"""
        alerts = self.alert_history

        if severity:
            alerts = [a for a in alerts if a.rule.severity == severity]

        return alerts[-limit:]

    def get_alert_stats(self) -> dict[str, Any]:
        """获取告警统计"""
        total_alerts = len(self.alert_history) + len(self.active_alerts)

        by_severity = {}
        for severity in AlertSeverity:
            count = sum(
                1 for a in self.alert_history + list(self.active_alerts.values())
                if a.rule.severity == severity
            )
            by_severity[severity.value] = count

        return {
            "total_alerts": total_alerts,
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len(self.alert_history),
            "by_severity": by_severity,
        }


# 单例实例
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """获取告警管理器单例"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
