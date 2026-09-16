"""ML model monitoring: registry view, live inference stats and drift detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml.model_store import MODEL_NAMES, get_model_store
from app.models import ModelPrediction, ModelRegistryEntry, Product, Review, UserEvent

logger = get_logger("monitoring")


class MonitoringService:
    def __init__(self, db: Session):
        self.db = db
        self.store = get_model_store()

    def registry(self) -> list[dict[str, Any]]:
        """Every registered model version, newest first."""
        rows = self.db.execute(
            select(ModelRegistryEntry).order_by(ModelRegistryEntry.trained_at.desc())
        ).scalars().all()
        live = self.store.status()["loaded"]
        out = []
        for row in rows:
            stats = live.get(row.name, {})
            out.append({
                "name": row.name,
                "version": row.version,
                "stage": row.stage,
                "algorithm": row.algorithm,
                "dataset_version": row.dataset_version,
                "training_rows": row.training_rows,
                "trained_at": row.trained_at.isoformat() if row.trained_at else None,
                "training_duration_s": row.training_duration_s,
                "metrics": row.metrics,
                "params": row.params,
                "feature_names": row.feature_names,
                "is_active": row.is_active,
                "notes": row.notes,
                "is_loaded": row.name in live and stats.get("version") == row.version,
                "live_stats": stats if stats.get("version") == row.version else None,
            })
        return out

    def model_health(self, days: int = 7) -> list[dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(
                ModelPrediction.model_name,
                ModelPrediction.model_version,
                func.count(ModelPrediction.id),
                func.avg(ModelPrediction.latency_ms),
                func.max(ModelPrediction.latency_ms),
                func.sum(case((ModelPrediction.succeeded.is_(False), 1), else_=0)),
            )
            .where(ModelPrediction.predicted_at >= since)
            .group_by(ModelPrediction.model_name, ModelPrediction.model_version)
        ).all()

        logged = {
            (name, version): {
                "logged_predictions": int(count),
                "avg_latency_ms": round(float(avg or 0), 4),
                "max_latency_ms": round(float(mx or 0), 4),
                "failures": int(fails or 0),
            }
            for name, version, count, avg, mx, fails in rows
        }

        live = self.store.status()
        out = []
        for name in MODEL_NAMES:
            stats = live["loaded"].get(name)
            available = self.store.is_available(name)
            entry: dict[str, Any] = {
                "model": name,
                "available": available,
                "loaded": stats is not None,
                "status": "healthy" if stats else ("trained" if available else "missing"),
            }
            if stats:
                entry.update({
                    "version": stats["version"],
                    "loaded_at": stats["loaded_at"],
                    "calls": stats["calls"],
                    "failures": stats["failures"],
                    "error_rate": stats["error_rate"],
                    "avg_latency_ms": stats["avg_latency_ms"],
                    "p95_latency_ms": stats["p95_latency_ms"],
                })
                entry.update(logged.get((name, stats["version"]), {}))
                if stats["error_rate"] > 0.05:
                    entry["status"] = "degraded"
            else:
                entry["hint"] = f"Run: python -m ml.training.train_{name}"
            out.append(entry)
        return out

    def drift_report(self) -> list[dict[str, Any]]:
        """Compare live feature/prediction distributions against training baselines."""
        reports: list[dict[str, Any]] = []
        for name in MODEL_NAMES:
            loaded = self.store.try_get(name)
            if loaded is None:
                continue
            baseline = loaded.card.baseline_stats or {}
            report: dict[str, Any] = {
                "model": name,
                "version": loaded.version,
                "trained_at": loaded.card.trained_at,
                "checks": [],
                "status": "ok",
            }
            for check in self._checks_for(name, baseline):
                report["checks"].append(check)
                if check.get("drift_detected"):
                    report["status"] = "drift_detected"
            # prediction drift from the live inference buffer
            predictions = loaded.recent_predictions
            if len(predictions) >= 30:
                mean_now = float(np.mean(predictions))
                report["prediction_stats"] = {
                    "samples": len(predictions),
                    "mean": round(mean_now, 5),
                    "std": round(float(np.std(predictions)), 5),
                    "p05": round(float(np.percentile(predictions, 5)), 5),
                    "p95": round(float(np.percentile(predictions, 95)), 5),
                }
            reports.append(report)
        return reports

    def _checks_for(self, name: str, baseline: dict[str, Any]) -> list[dict[str, Any]]:
        """Recompute the training-time statistics on today's data and compare."""
        checks: list[dict[str, Any]] = []
        threshold = settings.ML_DRIFT_THRESHOLD

        def compare(label: str, baseline_value: float | None, current_value: float | None,
                    tolerance: float = threshold) -> dict[str, Any]:
            if baseline_value is None or current_value is None:
                return {"feature": label, "status": "unavailable"}
            denominator = abs(baseline_value) if abs(baseline_value) > 1e-9 else 1.0
            relative = abs(current_value - baseline_value) / denominator
            return {
                "feature": label,
                "baseline": round(float(baseline_value), 5),
                "current": round(float(current_value), 5),
                "relative_change": round(float(relative), 5),
                "threshold": tolerance,
                "drift_detected": bool(relative > tolerance),
            }

        if name == "sentiment":
            rows = self.db.execute(
                select(Review.sentiment_label, func.count(Review.id))
                .where(Review.sentiment_label.isnot(None)).group_by(Review.sentiment_label)
            ).all()
            total = sum(int(c) for _, c in rows) or 1
            current = {label: int(count) / total for label, count in rows}
            for label, base_share in (baseline.get("label_distribution") or {}).items():
                checks.append(compare(f"label_share:{label}", base_share, current.get(label, 0.0)))

        elif name == "recommendation":
            in_stock = self.db.execute(
                select(func.avg(case((Product.inventory > 0, 1.0), else_=0.0)))
            ).scalar_one_or_none()
            checks.append(compare("in_stock_ratio", baseline.get("in_stock_ratio"),
                                  float(in_stock) if in_stock is not None else None))

        elif name == "price_prediction":
            median_price = self.db.execute(
                select(func.avg(Product.price)).where(Product.is_active.is_(True))
            ).scalar_one_or_none()
            checks.append(compare("price_level", baseline.get("price_median"),
                                  float(median_price) if median_price is not None else None,
                                  tolerance=threshold * 1.5))

        elif name == "forecasting":
            since = datetime.now(timezone.utc) - timedelta(days=30)
            recent = self.db.execute(
                select(func.count(UserEvent.id)).where(
                    UserEvent.event_type == "purchase", UserEvent.occurred_at >= since)
            ).scalar_one() or 0
            checks.append({
                "feature": "recent_purchase_volume_30d",
                "current": int(recent),
                "baseline": baseline.get("units_mean"),
                "status": "informational",
            })

        elif name == "segmentation":
            from app.models import CustomerSegment

            rows = self.db.execute(
                select(func.count(func.distinct(CustomerSegment.user_id)))
            ).scalar_one() or 0
            checks.append({
                "feature": "segmented_customers",
                "current": int(rows),
                "status": "informational",
            })
        return checks

    def summary(self) -> dict[str, Any]:
        health = self.model_health()
        drift = self.drift_report()
        return {
            "models_total": len(MODEL_NAMES),
            "models_loaded": sum(1 for h in health if h["loaded"]),
            "models_missing": [h["model"] for h in health if not h["available"]],
            "models_degraded": [h["model"] for h in health if h.get("status") == "degraded"],
            "drift_detected": [d["model"] for d in drift if d["status"] == "drift_detected"],
            "registry_size": int(self.db.execute(
                select(func.count(ModelRegistryEntry.id))
            ).scalar_one() or 0),
            "sampled_predictions": int(self.db.execute(
                select(func.count(ModelPrediction.id))
            ).scalar_one() or 0),
        }
