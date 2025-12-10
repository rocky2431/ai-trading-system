"""
Celery 应用配置
配置 Redis 作为 broker 和 result backend
支持多优先级队列和任务重试
"""

import os
from celery import Celery
from kombu import Queue, Exchange

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# 构建 Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# 创建 Celery 应用
celery_app = Celery(
    "iqfmp",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "iqfmp.celery_app.tasks",
    ],
)

# 定义交换机
default_exchange = Exchange("default", type="direct")
priority_exchange = Exchange("priority", type="direct")

# 定义队列配置
# - high: 紧急任务（风控、止损）
# - default: 普通任务（因子评估）
# - low: 低优先级任务（批量回测）
celery_app.conf.task_queues = (
    Queue("high", priority_exchange, routing_key="high", queue_arguments={"x-max-priority": 10}),
    Queue("default", default_exchange, routing_key="default", queue_arguments={"x-max-priority": 5}),
    Queue("low", default_exchange, routing_key="low", queue_arguments={"x-max-priority": 3}),
)

# 默认队列
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_routing_key = "default"

# 任务路由配置
celery_app.conf.task_routes = {
    # 高优先级任务
    "iqfmp.celery_app.tasks.emergency_close_task": {"queue": "high"},
    "iqfmp.celery_app.tasks.risk_check_task": {"queue": "high"},
    # 默认优先级任务
    "iqfmp.celery_app.tasks.evaluate_factor_task": {"queue": "default"},
    "iqfmp.celery_app.tasks.generate_factor_task": {"queue": "default"},
    # 低优先级任务
    "iqfmp.celery_app.tasks.backtest_task": {"queue": "low"},
    "iqfmp.celery_app.tasks.batch_backtest_task": {"queue": "low"},
}

# Celery 配置
celery_app.conf.update(
    # 序列化配置
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # 时区配置
    timezone="UTC",
    enable_utc=True,

    # 任务结果配置
    result_expires=3600 * 24,  # 结果保留 24 小时
    result_extended=True,  # 扩展结果信息

    # Worker 配置
    worker_prefetch_multiplier=1,  # 每次预取 1 个任务
    worker_concurrency=4,  # 并发 worker 数

    # 任务配置
    task_acks_late=True,  # 任务完成后才确认
    task_reject_on_worker_lost=True,  # worker 丢失时拒绝任务
    task_time_limit=3600,  # 任务超时 1 小时
    task_soft_time_limit=3300,  # 软超时 55 分钟

    # 重试配置
    task_default_retry_delay=60,  # 默认重试延迟 60 秒
    task_max_retries=3,  # 默认最大重试次数

    # 任务追踪
    task_track_started=True,
    task_send_sent_event=True,

    # 优先级支持
    broker_transport_options={
        "priority_steps": list(range(10)),
        "sep": ":",
        "queue_order_strategy": "priority",
    },
)


def get_celery_app() -> Celery:
    """获取 Celery 应用实例"""
    return celery_app
