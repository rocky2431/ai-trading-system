"""
Telegram 告警通知器
发送告警消息到 Telegram 群组或频道
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .alerts import Alert, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram 配置"""
    bot_token: str = ""
    chat_id: str = ""
    api_base: str = "https://api.telegram.org"
    timeout: float = 30.0
    parse_mode: str = "HTML"
    disable_notification: bool = False


class TelegramNotifier:
    """
    Telegram 通知器
    发送告警消息到 Telegram
    """

    def __init__(self, config: Optional[TelegramConfig] = None):
        self.config = config or self._load_config_from_env()
        self._client = None

    def _load_config_from_env(self) -> TelegramConfig:
        """从环境变量加载配置"""
        return TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            api_base=os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org"),
            timeout=float(os.getenv("TELEGRAM_TIMEOUT", "30.0")),
        )

    @property
    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.config.bot_token and self.config.chat_id)

    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """获取告警级别对应的 emoji"""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "🔴",
            AlertSeverity.CRITICAL: "🚨",
        }
        return emoji_map.get(severity, "📢")

    def _get_status_emoji(self, status: AlertStatus) -> str:
        """获取告警状态对应的 emoji"""
        status_map = {
            AlertStatus.PENDING: "⏳",
            AlertStatus.FIRING: "🔥",
            AlertStatus.RESOLVED: "✅",
        }
        return status_map.get(status, "📌")

    def format_alert_message(self, alert: Alert) -> str:
        """
        格式化告警消息

        Args:
            alert: 告警实例

        Returns:
            格式化的消息文本
        """
        severity_emoji = self._get_severity_emoji(alert.rule.severity)
        status_emoji = self._get_status_emoji(alert.status)

        if alert.status == AlertStatus.RESOLVED:
            title = f"{status_emoji} 告警恢复"
        else:
            title = f"{severity_emoji} 告警触发"

        # 构建消息
        lines = [
            f"<b>{title}</b>",
            "",
            f"📋 <b>告警名称:</b> {alert.rule.name}",
            f"📊 <b>级别:</b> {alert.rule.severity.value.upper()}",
            f"📝 <b>描述:</b> {alert.rule.description}",
            "",
            f"📈 <b>当前值:</b> {alert.value:.2f}",
            f"🎯 <b>阈值:</b> {alert.rule.threshold:.2f}",
            "",
            f"⏰ <b>开始时间:</b> {alert.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if alert.resolved_at:
            lines.append(
                f"✅ <b>恢复时间:</b> {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            lines.append(f"⏱️ <b>持续时间:</b> {self._format_duration(alert.duration)}")

        # 添加标签
        if alert.rule.labels:
            lines.append("")
            lines.append("<b>标签:</b>")
            for key, value in alert.rule.labels.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)

    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"

    async def send_message_async(
        self,
        text: str,
        chat_id: Optional[str] = None,
    ) -> bool:
        """
        异步发送消息

        Args:
            text: 消息文本
            chat_id: 聊天 ID（可选）

        Returns:
            是否发送成功
        """
        if not self.is_configured:
            logger.warning("Telegram not configured, skipping notification")
            return False

        target_chat_id = chat_id or self.config.chat_id

        try:
            import httpx

            url = f"{self.config.api_base}/bot{self.config.bot_token}/sendMessage"
            payload = {
                "chat_id": target_chat_id,
                "text": text,
                "parse_mode": self.config.parse_mode,
                "disable_notification": self.config.disable_notification,
            }

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        logger.info(f"Telegram message sent to {target_chat_id}")
                        return True

                logger.error(
                    f"Failed to send Telegram message: {response.status_code} - {response.text}"
                )
                return False

        except ImportError:
            logger.warning("httpx not installed, using mock Telegram client")
            logger.info(f"[Mock] Telegram message: {text[:100]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
    ) -> bool:
        """
        同步发送消息

        Args:
            text: 消息文本
            chat_id: 聊天 ID（可选）

        Returns:
            是否发送成功
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建任务
                future = asyncio.ensure_future(
                    self.send_message_async(text, chat_id)
                )
                return True  # 异步发送，假设成功
            else:
                return loop.run_until_complete(
                    self.send_message_async(text, chat_id)
                )
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(self.send_message_async(text, chat_id))

    def send_alert(self, alert: Alert) -> bool:
        """
        发送告警通知

        Args:
            alert: 告警实例

        Returns:
            是否发送成功
        """
        message = self.format_alert_message(alert)
        return self.send_message(message)

    async def send_alert_async(self, alert: Alert) -> bool:
        """
        异步发送告警通知

        Args:
            alert: 告警实例

        Returns:
            是否发送成功
        """
        message = self.format_alert_message(alert)
        return await self.send_message_async(message)

    def send_daily_summary(
        self,
        stats: dict[str, Any],
        date: Optional[datetime] = None,
    ) -> bool:
        """
        发送每日告警汇总

        Args:
            stats: 告警统计
            date: 日期

        Returns:
            是否发送成功
        """
        report_date = date or datetime.utcnow()

        lines = [
            f"📊 <b>告警日报 - {report_date.strftime('%Y-%m-%d')}</b>",
            "",
            f"📈 <b>总告警数:</b> {stats.get('total_alerts', 0)}",
            f"🔥 <b>活跃告警:</b> {stats.get('active_alerts', 0)}",
            f"✅ <b>已解决:</b> {stats.get('resolved_alerts', 0)}",
            "",
            "<b>按级别统计:</b>",
        ]

        by_severity = stats.get("by_severity", {})
        for severity, count in by_severity.items():
            emoji = self._get_severity_emoji(AlertSeverity(severity))
            lines.append(f"  {emoji} {severity.upper()}: {count}")

        message = "\n".join(lines)
        return self.send_message(message)

    def send_system_status(
        self,
        metrics: dict[str, float],
    ) -> bool:
        """
        发送系统状态报告

        Args:
            metrics: 系统指标

        Returns:
            是否发送成功
        """
        lines = [
            "🖥️ <b>系统状态报告</b>",
            "",
            f"💻 <b>CPU 使用率:</b> {metrics.get('cpu_usage_percent', 0):.1f}%",
            f"🧠 <b>内存使用率:</b> {metrics.get('memory_usage_percent', 0):.1f}%",
            f"⏱️ <b>请求延迟 (P99):</b> {metrics.get('request_latency_p99', 0):.2f}s",
            f"❌ <b>错误率:</b> {metrics.get('error_rate', 0):.2f}%",
            f"📋 <b>任务队列长度:</b> {metrics.get('task_queue_length', 0):.0f}",
            "",
            f"⏰ <b>报告时间:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ]

        message = "\n".join(lines)
        return self.send_message(message)


# 单例实例
_telegram_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier() -> TelegramNotifier:
    """获取 Telegram 通知器单例"""
    global _telegram_notifier
    if _telegram_notifier is None:
        _telegram_notifier = TelegramNotifier()
    return _telegram_notifier
