"""Celery 应用配置。

为什么单独一个文件
------------------
Celery 的 worker 进程要用同一份配置（`celery -A celery_app worker`），
而 API 进程也要用它派发任务。配置放这里，两边都 import 它，
避免出现"API 连 0 号库、worker 连 1 号库"这种查半天的问题。

环境变量
--------
CELERY_BROKER_URL     默认 redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND 默认 redis://127.0.0.1:6379/1
CELERY_EAGER=1        同步执行（测试用，不需要 broker）
"""

import os

from celery import Celery

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1")

celery_app = Celery(
    "shop_agent",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,

    # 任务一启动就标记为 STARTED，前端才能显示"处理中"而不是一直 PENDING
    task_track_started=True,

    # 单个索引任务的硬上限，防止卡死占住 worker
    task_time_limit=300,
    task_soft_time_limit=240,

    # 结果只留 1 小时，避免 Redis 无限增长
    result_expires=3600,

    # worker 启动时 broker 还没就绪也不要直接退出
    broker_connection_retry_on_startup=True,

    # 连接超时压到 1 秒：Redis 不可用时要快速失败，
    # 否则每次上传请求都要等好几秒才走到降级逻辑
    broker_transport_options={
        "socket_connect_timeout": 1,
        "socket_timeout": 1,
    },
    redis_socket_connect_timeout=1,
    redis_socket_timeout=1,
)

# 测试时用 CELERY_EAGER=1 同步执行，不需要真实 broker
if os.getenv("CELERY_EAGER") == "1":
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
