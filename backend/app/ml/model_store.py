"""Runtime model loading, caching and inference instrumentation.

Artifacts are loaded lazily from the registry, cached in-process, and hot-reloaded
when a newer version is registered. Every inference call is timed and (sampled)
recorded to `model_predictions` so the monitoring dashboard reflects real traffic.
"""
from __future__ import annotations

import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.errors import ModelNotAvailableError
from app.core.logging_config import get_logger

logger = get_logger("model_store")

MODEL_NAMES = ("recommendation", "segmentation", "sentiment", "forecasting", "price_prediction")
PREDICTION_SAMPLE_RATE = 0.15


@dataclass
class LoadedModel:
    name: str
    artifact: Any
    version: str
    card: Any
    loaded_at: datetime
    # rolling counters used by /health and the ML dashboard
    calls: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    recent_latencies: list[float] = field(default_factory=list)
    recent_predictions: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / self.calls, 4) if self.calls else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        ordered = sorted(self.recent_latencies)
        return round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 4)

    @property
    def error_rate(self) -> float:
        return round(self.failures / self.calls, 5) if self.calls else 0.0

    def record(self, latency_ms: float, prediction: float | None, ok: bool) -> None:
        self.calls += 1
        self.total_latency_ms += latency_ms
        if not ok:
            self.failures += 1
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > 500:
            del self.recent_latencies[:-500]
        if prediction is not None:
            self.recent_predictions.append(float(prediction))
            if len(self.recent_predictions) > 1000:
                del self.recent_predictions[:-1000]

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "loaded_at": self.loaded_at.isoformat(),
            "calls": self.calls,
            "failures": self.failures,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


class ModelStore:
    """Process-wide cache of loaded model artifacts."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.RLock()
        self._missing: dict[str, str] = {}
        self._pending_predictions: list[dict[str, Any]] = []

    # ---- loading -----------------------------------------------------
    def _registry(self):
        from ml.registry import get_registry

        return get_registry(settings.MODEL_PATH)

    def load(self, name: str, force: bool = False) -> LoadedModel:
        with self._lock:
            existing = self._models.get(name)
            if existing is not None and not force:
                return existing
            try:
                registry = self._registry()
                artifact, card = registry.load(name)
            except FileNotFoundError as exc:
                self._missing[name] = str(exc)
                raise ModelNotAvailableError(
                    f"Model '{name}' has not been trained yet. Run: python -m ml.training.train_{name}"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - corrupt artifact, bad pickle, etc.
                self._missing[name] = str(exc)
                logger.error("model_load_failed", model=name, error=str(exc))
                raise ModelNotAvailableError(f"Model '{name}' could not be loaded.") from exc

            loaded = LoadedModel(
                name=name, artifact=artifact, version=card.version, card=card,
                loaded_at=datetime.now(timezone.utc),
            )
            self._models[name] = loaded
            self._missing.pop(name, None)
            logger.info("model_loaded", model=name, version=card.version, algorithm=card.algorithm)
            return loaded

    def get(self, name: str) -> LoadedModel:
        model = self._models.get(name)
        return model if model is not None else self.load(name)

    def try_get(self, name: str) -> LoadedModel | None:
        try:
            return self.get(name)
        except ModelNotAvailableError:
            return None

    def artifact(self, name: str) -> Any:
        return self.get(name).artifact

    def reload_all(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for name in MODEL_NAMES:
            try:
                results[name] = self.load(name, force=True).version
            except ModelNotAvailableError as exc:
                results[name] = f"unavailable: {exc.message}"
        return results

    def warm_up(self) -> dict[str, str]:
        """Best-effort preload at startup; missing models must not block boot."""
        results: dict[str, str] = {}
        for name in MODEL_NAMES:
            try:
                results[name] = self.load(name).version
            except ModelNotAvailableError:
                results[name] = "unavailable"
        return results

    def check_for_updates(self) -> list[str]:
        """Hot-reload any model whose registry version has moved on."""
        updated: list[str] = []
        try:
            registry = self._registry()
        except Exception:  # noqa: BLE001
            return updated
        for name, loaded in list(self._models.items()):
            card = registry.latest_card(name)
            if card and card.version != loaded.version:
                try:
                    self.load(name, force=True)
                    updated.append(name)
                except ModelNotAvailableError:
                    continue
        return updated

    # ---- instrumentation ---------------------------------------------
    @contextmanager
    def timed(self, name: str, entity_type: str = "product", entity_id: int | None = None) -> Iterator[dict]:
        """Time an inference call and sample it into the prediction log."""
        started = time.perf_counter()
        box: dict[str, Any] = {"prediction": None, "error": None}
        ok = True
        try:
            yield box
        except Exception as exc:  # noqa: BLE001 - re-raised after recording
            ok = False
            box["error"] = f"{type(exc).__name__}"
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            model = self._models.get(name)
            if model is not None:
                model.record(latency_ms, box.get("prediction"), ok)
            if random.random() < PREDICTION_SAMPLE_RATE or not ok:
                self._pending_predictions.append({
                    "model_name": name,
                    "model_version": model.version if model else "unknown",
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "prediction": float(box.get("prediction") or 0.0),
                    "latency_ms": round(latency_ms, 4),
                    "succeeded": ok,
                    "error_code": box.get("error"),
                    "features": {},
                    "predicted_at": datetime.now(timezone.utc),
                })

    def drain_predictions(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            batch = self._pending_predictions[:limit]
            del self._pending_predictions[:limit]
            return batch

    def flush_predictions(self) -> int:
        """Persist sampled predictions; never raises."""
        batch = self.drain_predictions()
        if not batch:
            return 0
        try:
            from sqlalchemy.orm import Session

            from app.core.db import engine
            from app.models import ModelPrediction

            with Session(engine) as session:
                session.bulk_insert_mappings(ModelPrediction, batch)
                session.commit()
            return len(batch)
        except Exception as exc:  # noqa: BLE001 - telemetry must never break serving
            logger.warning("prediction_flush_failed", error=str(exc), dropped=len(batch))
            return 0

    # ---- introspection -----------------------------------------------
    def status(self) -> dict[str, Any]:
        loaded = {name: model.stats() for name, model in self._models.items()}
        return {
            "loaded": loaded,
            "missing": {name: self._missing.get(name, "not loaded") for name in MODEL_NAMES
                        if name not in self._models},
            "available": list(loaded),
            "pending_prediction_logs": len(self._pending_predictions),
        }

    def is_available(self, name: str) -> bool:
        if name in self._models:
            return True
        try:
            return self._registry().exists(name)
        except Exception:  # noqa: BLE001
            return False


_store: ModelStore | None = None
_store_lock = threading.Lock()


def get_model_store() -> ModelStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ModelStore()
    return _store


def reset_model_store() -> None:
    global _store
    with _store_lock:
        _store = None
