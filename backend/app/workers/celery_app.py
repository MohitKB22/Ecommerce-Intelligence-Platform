"""Celery application - used only when CELERY_BROKER_URL is configured."""
from __future__ import annotations

from app.core.config import settings
from app.workers.tasks import SCHEDULE, run_task

celery_app = None

if settings.CELERY_BROKER_URL:  # pragma: no cover - requires a live broker
    from celery import Celery
    from celery.schedules import schedule

    celery_app = Celery(
        "ecommerce_intelligence",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND or settings.CELERY_BROKER_URL,
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        worker_max_tasks_per_child=200,
        task_time_limit=600,
        task_soft_time_limit=540,
    )

    @celery_app.task(name="eci.run_task")
    def run_named_task(name: str):
        return run_task(name)

    celery_app.conf.beat_schedule = {
        f"periodic-{name}": {
            "task": "eci.run_task",
            "schedule": schedule(run_every=interval),
            "args": (name,),
        }
        for name, (_, interval) in SCHEDULE.items()
    }
