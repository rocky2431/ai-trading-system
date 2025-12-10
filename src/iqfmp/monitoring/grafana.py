"""
Grafana 仪表板配置生成器
生成 IQFMP 监控仪表板的 JSON 配置
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GrafanaPanel:
    """Grafana 面板配置"""
    id: int
    title: str
    type: str  # graph, gauge, stat, table, etc.
    gridPos: dict[str, int]
    targets: list[dict[str, Any]] = field(default_factory=list)
    fieldConfig: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GrafanaDashboard:
    """Grafana 仪表板配置"""
    uid: str
    title: str
    description: str
    panels: list[GrafanaPanel] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"


class GrafanaDashboardGenerator:
    """
    Grafana 仪表板生成器
    生成 IQFMP 监控仪表板配置
    """

    def __init__(
        self,
        datasource: str = "Prometheus",
        namespace: str = "iqfmp",
        subsystem: str = "app",
    ):
        self.datasource = datasource
        self.namespace = namespace
        self.subsystem = subsystem
        self._panel_id = 0

    def _next_panel_id(self) -> int:
        """生成下一个面板 ID"""
        self._panel_id += 1
        return self._panel_id

    def _metric_name(self, name: str) -> str:
        """生成完整指标名称"""
        return f"{self.namespace}_{self.subsystem}_{name}"

    def _create_prometheus_target(
        self,
        expr: str,
        legend_format: str = "",
        ref_id: str = "A",
    ) -> dict[str, Any]:
        """创建 Prometheus 查询目标"""
        return {
            "datasource": {"type": "prometheus", "uid": self.datasource},
            "expr": expr,
            "legendFormat": legend_format,
            "refId": ref_id,
        }

    def generate_system_overview_dashboard(self) -> dict[str, Any]:
        """生成系统概览仪表板"""
        panels = []

        # Row: 系统概览
        panels.append(self._create_row_panel("系统概览", 0))

        # CPU 使用率 Gauge
        panels.append(
            self._create_gauge_panel(
                title="CPU 使用率",
                expr=f"{self._metric_name('cpu_usage_percent')}",
                unit="percent",
                thresholds=[
                    {"value": 0, "color": "green"},
                    {"value": 70, "color": "yellow"},
                    {"value": 85, "color": "red"},
                ],
                grid_pos={"x": 0, "y": 1, "w": 6, "h": 6},
            )
        )

        # 内存使用率 Gauge
        panels.append(
            self._create_gauge_panel(
                title="内存使用率",
                expr=f"{self._metric_name('memory_usage_bytes')} / 1024 / 1024 / 1024",
                unit="decgbytes",
                thresholds=[
                    {"value": 0, "color": "green"},
                    {"value": 4, "color": "yellow"},
                    {"value": 8, "color": "red"},
                ],
                grid_pos={"x": 6, "y": 1, "w": 6, "h": 6},
            )
        )

        # 活跃请求数 Stat
        panels.append(
            self._create_stat_panel(
                title="活跃请求",
                expr=f"sum({self._metric_name('active_requests')})",
                unit="short",
                grid_pos={"x": 12, "y": 1, "w": 6, "h": 6},
            )
        )

        # 错误计数 Stat
        panels.append(
            self._create_stat_panel(
                title="错误总数",
                expr=f"sum(increase({self._metric_name('errors_total')}[1h]))",
                unit="short",
                color_mode="background",
                grid_pos={"x": 18, "y": 1, "w": 6, "h": 6},
            )
        )

        # Row: 请求性能
        panels.append(self._create_row_panel("请求性能", 7))

        # 请求延迟 Graph
        panels.append(
            self._create_timeseries_panel(
                title="请求延迟分布",
                targets=[
                    self._create_prometheus_target(
                        f"histogram_quantile(0.50, rate({self._metric_name('request_latency_seconds_bucket')}[5m]))",
                        "P50",
                        "A",
                    ),
                    self._create_prometheus_target(
                        f"histogram_quantile(0.90, rate({self._metric_name('request_latency_seconds_bucket')}[5m]))",
                        "P90",
                        "B",
                    ),
                    self._create_prometheus_target(
                        f"histogram_quantile(0.99, rate({self._metric_name('request_latency_seconds_bucket')}[5m]))",
                        "P99",
                        "C",
                    ),
                ],
                unit="s",
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
            )
        )

        # 请求速率 Graph
        panels.append(
            self._create_timeseries_panel(
                title="请求速率 (RPS)",
                targets=[
                    self._create_prometheus_target(
                        f"sum(rate({self._metric_name('requests_total')}[5m]))",
                        "Total RPS",
                    ),
                ],
                unit="reqps",
                grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
            )
        )

        # Row: 任务执行
        panels.append(self._create_row_panel("任务执行", 16))

        # 任务执行时间 Graph
        panels.append(
            self._create_timeseries_panel(
                title="任务执行时间",
                targets=[
                    self._create_prometheus_target(
                        f"histogram_quantile(0.95, rate({self._metric_name('task_duration_seconds_bucket')}[5m]))",
                        "{{task_name}}",
                    ),
                ],
                unit="s",
                grid_pos={"x": 0, "y": 17, "w": 12, "h": 8},
            )
        )

        # 任务成功/失败率 Graph
        panels.append(
            self._create_timeseries_panel(
                title="任务状态",
                targets=[
                    self._create_prometheus_target(
                        f"sum(rate({self._metric_name('tasks_total')}{{status='success'}}[5m])) by (task_name)",
                        "{{task_name}} - Success",
                        "A",
                    ),
                    self._create_prometheus_target(
                        f"sum(rate({self._metric_name('tasks_total')}{{status='failure'}}[5m])) by (task_name)",
                        "{{task_name}} - Failure",
                        "B",
                    ),
                ],
                unit="short",
                grid_pos={"x": 12, "y": 17, "w": 12, "h": 8},
            )
        )

        return self._build_dashboard(
            uid="iqfmp-system-overview",
            title="IQFMP - 系统概览",
            description="IQFMP 系统监控概览仪表板",
            panels=panels,
            tags=["iqfmp", "system", "overview"],
        )

    def generate_trading_dashboard(self) -> dict[str, Any]:
        """生成交易监控仪表板"""
        panels = []

        # Row: 交易概览
        panels.append(self._create_row_panel("交易概览", 0))

        # 持仓数量 Stat
        panels.append(
            self._create_stat_panel(
                title="总持仓数",
                expr=f"sum({self._metric_name('position_count')})",
                unit="short",
                grid_pos={"x": 0, "y": 1, "w": 6, "h": 4},
            )
        )

        # 已实现 PnL Stat
        panels.append(
            self._create_stat_panel(
                title="已实现 PnL",
                expr=f"{self._metric_name('pnl_total')}{{type='realized'}}",
                unit="currencyUSD",
                color_mode="value",
                grid_pos={"x": 6, "y": 1, "w": 6, "h": 4},
            )
        )

        # 未实现 PnL Stat
        panels.append(
            self._create_stat_panel(
                title="未实现 PnL",
                expr=f"{self._metric_name('pnl_total')}{{type='unrealized'}}",
                unit="currencyUSD",
                color_mode="value",
                grid_pos={"x": 12, "y": 1, "w": 6, "h": 4},
            )
        )

        # 回测计数 Stat
        panels.append(
            self._create_stat_panel(
                title="今日回测",
                expr=f"sum(increase({self._metric_name('backtests_total')}[24h]))",
                unit="short",
                grid_pos={"x": 18, "y": 1, "w": 6, "h": 4},
            )
        )

        # Row: 持仓详情
        panels.append(self._create_row_panel("持仓详情", 5))

        # 持仓分布 Graph
        panels.append(
            self._create_timeseries_panel(
                title="持仓分布",
                targets=[
                    self._create_prometheus_target(
                        f"{self._metric_name('position_count')}",
                        "{{symbol}} - {{side}}",
                    ),
                ],
                unit="short",
                grid_pos={"x": 0, "y": 6, "w": 12, "h": 8},
            )
        )

        # PnL 趋势 Graph
        panels.append(
            self._create_timeseries_panel(
                title="PnL 趋势",
                targets=[
                    self._create_prometheus_target(
                        f"{self._metric_name('pnl_total')}{{type='realized'}}",
                        "已实现",
                        "A",
                    ),
                    self._create_prometheus_target(
                        f"{self._metric_name('pnl_total')}{{type='unrealized'}}",
                        "未实现",
                        "B",
                    ),
                ],
                unit="currencyUSD",
                grid_pos={"x": 12, "y": 6, "w": 12, "h": 8},
            )
        )

        # Row: 因子分析
        panels.append(self._create_row_panel("因子分析", 14))

        # 因子数量 Graph
        panels.append(
            self._create_timeseries_panel(
                title="因子数量趋势",
                targets=[
                    self._create_prometheus_target(
                        f"{self._metric_name('factor_count')}",
                        "{{family}}",
                    ),
                ],
                unit="short",
                grid_pos={"x": 0, "y": 15, "w": 24, "h": 8},
            )
        )

        return self._build_dashboard(
            uid="iqfmp-trading",
            title="IQFMP - 交易监控",
            description="IQFMP 交易和因子监控仪表板",
            panels=panels,
            tags=["iqfmp", "trading", "factors"],
        )

    def _create_row_panel(self, title: str, y: int) -> dict[str, Any]:
        """创建行面板"""
        return {
            "id": self._next_panel_id(),
            "type": "row",
            "title": title,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "collapsed": False,
        }

    def _create_stat_panel(
        self,
        title: str,
        expr: str,
        unit: str = "short",
        color_mode: str = "value",
        grid_pos: Optional[dict] = None,
    ) -> dict[str, Any]:
        """创建 Stat 面板"""
        return {
            "id": self._next_panel_id(),
            "type": "stat",
            "title": title,
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 6, "h": 4},
            "targets": [self._create_prometheus_target(expr)],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "color": {"mode": color_mode},
                },
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": "",
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": color_mode,
                "graphMode": "area",
            },
        }

    def _create_gauge_panel(
        self,
        title: str,
        expr: str,
        unit: str = "percent",
        thresholds: Optional[list[dict]] = None,
        grid_pos: Optional[dict] = None,
    ) -> dict[str, Any]:
        """创建 Gauge 面板"""
        default_thresholds = [
            {"value": 0, "color": "green"},
            {"value": 70, "color": "yellow"},
            {"value": 90, "color": "red"},
        ]

        return {
            "id": self._next_panel_id(),
            "type": "gauge",
            "title": title,
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 6, "h": 6},
            "targets": [self._create_prometheus_target(expr)],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or default_thresholds,
                    },
                },
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": "",
                },
                "orientation": "auto",
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
            },
        }

    def _create_timeseries_panel(
        self,
        title: str,
        targets: list[dict],
        unit: str = "short",
        grid_pos: Optional[dict] = None,
    ) -> dict[str, Any]:
        """创建时序图面板"""
        return {
            "id": self._next_panel_id(),
            "type": "timeseries",
            "title": title,
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
            "targets": targets,
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "showPoints": "auto",
                        "pointSize": 5,
                        "stacking": {"mode": "none", "group": "A"},
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "scaleDistribution": {"type": "linear"},
                    },
                },
            },
            "options": {
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom",
                    "calcs": [],
                },
                "tooltip": {"mode": "single", "sort": "none"},
            },
        }

    def _build_dashboard(
        self,
        uid: str,
        title: str,
        description: str,
        panels: list[dict],
        tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """构建完整的仪表板配置"""
        return {
            "uid": uid,
            "title": title,
            "description": description,
            "tags": tags or [],
            "style": "dark",
            "timezone": "browser",
            "editable": True,
            "graphTooltip": 0,
            "panels": panels,
            "schemaVersion": 38,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {
                "refresh_intervals": ["5s", "10s", "30s", "1m", "5m"],
                "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "7d"],
            },
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": {"type": "datasource", "uid": "grafana"},
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "target": {
                            "limit": 100,
                            "matchAny": False,
                            "tags": [],
                            "type": "dashboard",
                        },
                        "type": "dashboard",
                    }
                ]
            },
        }

    def export_dashboard(
        self,
        dashboard: dict[str, Any],
        filepath: str,
    ):
        """导出仪表板配置到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)

        logger.info(f"Dashboard exported to: {filepath}")

    def export_all_dashboards(self, output_dir: str):
        """导出所有仪表板"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # 系统概览仪表板
        system_dashboard = self.generate_system_overview_dashboard()
        self.export_dashboard(
            system_dashboard,
            os.path.join(output_dir, "system-overview.json"),
        )

        # 交易监控仪表板
        trading_dashboard = self.generate_trading_dashboard()
        self.export_dashboard(
            trading_dashboard,
            os.path.join(output_dir, "trading.json"),
        )

        logger.info(f"All dashboards exported to: {output_dir}")
