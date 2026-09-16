"""Liveness, readiness and dependency health."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Response, status

from app.api.deps import DbSession
from app.core.cache import get_cache
from app.core.config import settings
from app.core.db import dialect_name, ping
from app.ml.model_store import MODEL_NAMES, get_model_store
from app.schemas.ml import HealthResponse

router = APIRouter(tags=["health"])
_STARTED_AT = time.time()
APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Overall service health")
def health(db: DbSession, response: Response) -> HealthResponse:
    """Aggregate health of the database, cache and ML models.

    Returns `degraded` (still HTTP 200) when optional dependencies are down, and
    503 only when the database - which the API cannot function without - fails.
    """
    cache = get_cache()
    store = get_model_store()

    db_ok = ping()
    cache_ok = cache.ping()
    model_status = {name: store.is_available(name) for name in MODEL_NAMES}
    models_available = sum(1 for ok in model_status.values() if ok)

    if not db_ok:
        overall = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif not cache_ok or models_available < len(MODEL_NAMES):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        environment=settings.ENVIRONMENT,
        version=APP_VERSION,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
        checks={
            "database": {
                "status": "ok" if db_ok else "error",
                "dialect": dialect_name(),
                "required": True,
            },
            "cache": {
                "status": "ok" if cache_ok else "unavailable",
                "backend": cache.backend_name,
                "required": False,
                "stats": cache.stats(),
            },
            "models": {
                "status": "ok" if models_available == len(MODEL_NAMES) else "partial",
                "available": models_available,
                "expected": len(MODEL_NAMES),
                "detail": model_status,
                "required": False,
            },
        },
    )


@router.get("/health/live", summary="Liveness probe")
def liveness() -> dict[str, Any]:
    """Cheap check that the process is running - used by container orchestration."""
    return {"status": "alive", "uptime_seconds": round(time.time() - _STARTED_AT, 2)}


@router.get("/health/ready", summary="Readiness probe")
def readiness(db: DbSession, response: Response) -> dict[str, Any]:
    """Ready only when the database is reachable; the cache may be absent."""
    db_ok = ping()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if db_ok else "not_ready",
        "database": "ok" if db_ok else "error",
        "cache": "ok" if get_cache().ping() else "unavailable",
    }
