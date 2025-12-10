"""
Prometheus 指标收集器
提供应用级别的指标收集和导出功能
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """指标配置"""
    namespace: str = "iqfmp"
    subsystem: str = "app"
    port: int = 9090
    path: str = "/metrics"
    enable_default_metrics: bool = True


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PrometheusMetrics:
    """
    Prometheus 指标封装
    使用 prometheus_client 库进行指标收集
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._summaries: dict[str, Any] = {}
        self._initialized = False

    def _init_prometheus(self):
        """初始化 Prometheus 客户端"""
        if self._initialized:
            return

        try:
            from prometheus_client import (
                Counter,
                Gauge,
                Histogram,
                Summary,
                REGISTRY,
                generate_latest,
                start_http_server,
            )

            self._Counter = Counter
            self._Gauge = Gauge
            self._Histogram = Histogram
            self._Summary = Summary
            self._REGISTRY = REGISTRY
            self._generate_latest = generate_latest
            self._start_http_server = start_http_server
            self._initialized = True

            logger.info("Prometheus client initialized")

        except ImportError:
            logger.warning("prometheus_client not installed, using mock metrics")
            self._initialized = False

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ) -> Any:
        """创建或获取 Counter 指标"""
        self._init_prometheus()

        full_name = f"{self.config.namespace}_{self.config.subsystem}_{name}"

        if full_name not in self._counters:
            if self._initialized:
                self._counters[full_name] = self._Counter(
                    full_name,
                    description,
                    labels or [],
                )
            else:
                self._counters[full_name] = MockCounter(full_name)

        return self._counters[full_name]

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ) -> Any:
        """创建或获取 Gauge 指标"""
        self._init_prometheus()

        full_name = f"{self.config.namespace}_{self.config.subsystem}_{name}"

        if full_name not in self._gauges:
            if self._initialized:
                self._gauges[full_name] = self._Gauge(
                    full_name,
                    description,
                    labels or [],
                )
            else:
                self._gauges[full_name] = MockGauge(full_name)

        return self._gauges[full_name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
        buckets: Optional[tuple] = None,
    ) -> Any:
        """创建或获取 Histogram 指标"""
        self._init_prometheus()

        full_name = f"{self.config.namespace}_{self.config.subsystem}_{name}"

        if full_name not in self._histograms:
            if self._initialized:
                kwargs = {"labelnames": labels or []}
                if buckets:
                    kwargs["buckets"] = buckets
                self._histograms[full_name] = self._Histogram(
                    full_name,
                    description,
                    **kwargs,
                )
            else:
                self._histograms[full_name] = MockHistogram(full_name)

        return self._histograms[full_name]

    def summary(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ) -> Any:
        """创建或获取 Summary 指标"""
        self._init_prometheus()

        full_name = f"{self.config.namespace}_{self.config.subsystem}_{name}"

        if full_name not in self._summaries:
            if self._initialized:
                self._summaries[full_name] = self._Summary(
                    full_name,
                    description,
                    labels or [],
                )
            else:
                self._summaries[full_name] = MockSummary(full_name)

        return self._summaries[full_name]

    def start_server(self, port: Optional[int] = None):
        """启动 Prometheus HTTP 服务器"""
        self._init_prometheus()

        if not self._initialized:
            logger.warning("Cannot start Prometheus server - client not initialized")
            return

        actual_port = port or self.config.port
        self._start_http_server(actual_port)
        logger.info(f"Prometheus metrics server started on port {actual_port}")

    def generate_metrics(self) -> bytes:
        """生成指标输出"""
        self._init_prometheus()

        if not self._initialized:
            return b""

        return self._generate_latest(self._REGISTRY)


class MetricsCollector:
    """
    应用指标收集器
    提供高级别的指标收集 API
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.prometheus = PrometheusMetrics(self.config)
        self._setup_default_metrics()

    def _setup_default_metrics(self):
        """设置默认指标"""
        # 请求计数器
        self.request_total = self.prometheus.counter(
            "requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        # 请求延迟直方图
        self.request_latency = self.prometheus.histogram(
            "request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # 活跃请求数
        self.active_requests = self.prometheus.gauge(
            "active_requests",
            "Number of active requests",
            ["endpoint"],
        )

        # 任务执行指标
        self.task_total = self.prometheus.counter(
            "tasks_total",
            "Total number of tasks executed",
            ["task_name", "status"],
        )

        self.task_duration = self.prometheus.histogram(
            "task_duration_seconds",
            "Task execution duration in seconds",
            ["task_name"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
        )

        # 错误计数
        self.errors_total = self.prometheus.counter(
            "errors_total",
            "Total number of errors",
            ["error_type", "component"],
        )

        # 系统资源指标
        self.cpu_usage = self.prometheus.gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
        )

        self.memory_usage = self.prometheus.gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
        )

        # 业务指标
        self.factor_count = self.prometheus.gauge(
            "factor_count",
            "Number of factors in the system",
            ["family"],
        )

        self.backtest_count = self.prometheus.counter(
            "backtests_total",
            "Total number of backtests executed",
            ["strategy", "result"],
        )

        self.position_count = self.prometheus.gauge(
            "position_count",
            "Number of open positions",
            ["symbol", "side"],
        )

        self.pnl_total = self.prometheus.gauge(
            "pnl_total",
            "Total PnL",
            ["type"],
        )

    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        latency: float,
    ):
        """记录请求指标"""
        self.request_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status),
        ).inc()

        self.request_latency.labels(
            method=method,
            endpoint=endpoint,
        ).observe(latency)

    def record_task(
        self,
        task_name: str,
        status: str,
        duration: float,
    ):
        """记录任务执行指标"""
        self.task_total.labels(
            task_name=task_name,
            status=status,
        ).inc()

        self.task_duration.labels(
            task_name=task_name,
        ).observe(duration)

    def record_error(self, error_type: str, component: str):
        """记录错误"""
        self.errors_total.labels(
            error_type=error_type,
            component=component,
        ).inc()

    def update_system_metrics(self):
        """更新系统资源指标"""
        try:
            import psutil

            self.cpu_usage.set(psutil.cpu_percent())
            self.memory_usage.set(psutil.Process().memory_info().rss)

        except ImportError:
            logger.debug("psutil not installed, skipping system metrics")

    def update_factor_count(self, family: str, count: int):
        """更新因子计数"""
        self.factor_count.labels(family=family).set(count)

    def record_backtest(self, strategy: str, result: str):
        """记录回测"""
        self.backtest_count.labels(
            strategy=strategy,
            result=result,
        ).inc()

    def update_position(self, symbol: str, side: str, count: int):
        """更新持仓数量"""
        self.position_count.labels(
            symbol=symbol,
            side=side,
        ).set(count)

    def update_pnl(self, pnl_type: str, value: float):
        """更新 PnL"""
        self.pnl_total.labels(type=pnl_type).set(value)

    def start_server(self, port: Optional[int] = None):
        """启动指标服务器"""
        self.prometheus.start_server(port)

    def get_metrics(self) -> bytes:
        """获取指标输出"""
        return self.prometheus.generate_metrics()


# Mock 实现
class MockMetric:
    """Mock 指标基类"""

    def __init__(self, name: str):
        self.name = name
        self._values: dict[tuple, float] = {}

    def labels(self, **kwargs) -> "MockMetric":
        return self


class MockCounter(MockMetric):
    """Mock Counter"""

    def inc(self, amount: float = 1):
        pass


class MockGauge(MockMetric):
    """Mock Gauge"""

    def set(self, value: float):
        pass

    def inc(self, amount: float = 1):
        pass

    def dec(self, amount: float = 1):
        pass


class MockHistogram(MockMetric):
    """Mock Histogram"""

    def observe(self, value: float):
        pass


class MockSummary(MockMetric):
    """Mock Summary"""

    def observe(self, value: float):
        pass


# 单例实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取指标收集器单例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# 装饰器
def track_request_latency(endpoint: str):
    """请求延迟追踪装饰器"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            collector.active_requests.labels(endpoint=endpoint).inc()

            start_time = time.time()
            status = 200

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                collector.record_error(type(e).__name__, endpoint)
                raise
            finally:
                latency = time.time() - start_time
                collector.record_request("GET", endpoint, status, latency)
                collector.active_requests.labels(endpoint=endpoint).dec()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            collector.active_requests.labels(endpoint=endpoint).inc()

            start_time = time.time()
            status = 200

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                collector.record_error(type(e).__name__, endpoint)
                raise
            finally:
                latency = time.time() - start_time
                collector.record_request("GET", endpoint, status, latency)
                collector.active_requests.labels(endpoint=endpoint).dec()

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_task_execution(task_name: str):
    """任务执行追踪装饰器"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "failure"
                collector.record_error(type(e).__name__, task_name)
                raise
            finally:
                duration = time.time() - start_time
                collector.record_task(task_name, status, duration)

        return wrapper

    return decorator


@contextmanager
def track_latency(collector: MetricsCollector, method: str, endpoint: str):
    """延迟追踪上下文管理器"""
    start_time = time.time()
    status = 200

    try:
        yield
    except Exception:
        status = 500
        raise
    finally:
        latency = time.time() - start_time
        collector.record_request(method, endpoint, status, latency)
