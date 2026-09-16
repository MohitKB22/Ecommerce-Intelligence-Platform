"""Customer segmentation serving."""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.errors import NotFoundError
from app.ml.model_store import get_model_store
from app.models import CustomerSegment, User

SEGMENT_TTL = 600

SEGMENT_DESCRIPTIONS = {
    "high_value": "Top spenders with high order values - protect with loyalty perks and early access.",
    "loyal_customer": "Frequent, long-tenured buyers - strong candidates for referral programmes.",
    "frequent_buyer": "High purchase frequency across many categories - ideal for cross-sell.",
    "discount_seeker": "Predominantly buys on promotion - responds to markdowns, thin margin.",
    "occasional_buyer": "Infrequent, low-value purchases - target with reactivation campaigns.",
    "at_risk": "Previously active but has gone quiet - prioritise win-back messaging.",
    "new_customer": "Recently joined with little history - focus on onboarding and first repeat purchase.",
}


class SegmentationService:
    def __init__(self, db: Session):
        self.db = db
        self.store = get_model_store()
        self.cache = get_cache()

    def latest_version(self) -> str | None:
        return self.db.execute(
            select(CustomerSegment.model_version).order_by(CustomerSegment.created_at.desc()).limit(1)
        ).scalar_one_or_none()

    def overview(self) -> dict[str, Any]:
        version = self.latest_version()
        if version is None:
            return {"available": False, "segments": [], "total_customers": 0,
                    "note": "Segmentation has not been run yet. Run: python -m ml.training.train_segmentation"}

        key = self.cache.key("segments", version)
        cached = self.cache.get_json(key)
        if cached is not None:
            return cached

        rows = self.db.execute(
            select(
                CustomerSegment.segment_label,
                func.count(CustomerSegment.id),
                func.avg(CustomerSegment.monetary),
                func.avg(CustomerSegment.frequency),
                func.avg(CustomerSegment.recency_days),
                func.avg(CustomerSegment.avg_order_value),
                func.avg(CustomerSegment.rfm_score),
                func.avg(CustomerSegment.confidence),
            )
            .where(CustomerSegment.model_version == version)
            .group_by(CustomerSegment.segment_label)
            .order_by(func.count(CustomerSegment.id).desc())
        ).all()

        total = sum(int(r[1]) for r in rows) or 1
        segments = [
            {
                "label": label,
                "name": label.replace("_", " ").title(),
                "description": SEGMENT_DESCRIPTIONS.get(label, ""),
                "customer_count": int(count),
                "share": round(int(count) / total, 4),
                "avg_monetary": round(float(monetary or 0), 4),
                "avg_frequency": round(float(frequency or 0), 4),
                "avg_recency_days": round(float(recency or 0), 2),
                "avg_order_value": round(float(aov or 0), 4),
                "avg_rfm_score": round(float(rfm or 0), 4),
                "avg_confidence": round(float(confidence or 0), 4),
            }
            for label, count, monetary, frequency, recency, aov, rfm, confidence in rows
        ]

        loaded = self.store.try_get("segmentation")
        payload = {
            "available": True,
            "model_version": version,
            "total_customers": total,
            "segments": segments,
            "metrics": (loaded.card.metrics if loaded else {}),
            # JSON object keys must be strings; the artifact keys clusters by int.
            "cluster_profiles": ({str(k): v for k, v in loaded.artifact.cluster_profiles.items()}
                                 if loaded else {}),
            "feature_names": (loaded.artifact.feature_names if loaded else []),
        }
        self.cache.set_json(key, payload, SEGMENT_TTL)
        return payload

    def for_user(self, user_id: int) -> dict[str, Any]:
        if self.db.get(User, user_id) is None:
            raise NotFoundError(f"User {user_id} was not found.")
        row = self.db.execute(
            select(CustomerSegment).where(CustomerSegment.user_id == user_id)
            .order_by(CustomerSegment.created_at.desc()).limit(1)
        ).scalars().first()
        if row is None:
            return {"user_id": user_id, "segment": None,
                    "note": "This customer has not been assigned a segment yet."}
        return {
            "user_id": user_id,
            "segment": row.segment_label,
            "segment_name": row.segment_label.replace("_", " ").title(),
            "description": SEGMENT_DESCRIPTIONS.get(row.segment_label, ""),
            "cluster_id": row.cluster_id,
            "model_version": row.model_version,
            "confidence": round(float(row.confidence), 4),
            "rfm_score": round(float(row.rfm_score), 4),
            "features": {
                "recency_days": round(float(row.recency_days), 2),
                "frequency": round(float(row.frequency), 4),
                "monetary": round(float(row.monetary), 4),
                "avg_order_value": round(float(row.avg_order_value), 4),
                "product_diversity": round(float(row.product_diversity), 4),
                "session_frequency": round(float(row.session_frequency), 4),
            },
        }

    def members(self, label: str, limit: int = 50) -> list[dict[str, Any]]:
        version = self.latest_version()
        if version is None:
            return []
        rows = self.db.execute(
            select(CustomerSegment, User)
            .join(User, User.id == CustomerSegment.user_id)
            .where(CustomerSegment.segment_label == label, CustomerSegment.model_version == version)
            .order_by(CustomerSegment.rfm_score.desc())
            .limit(limit)
        ).all()
        return [
            {
                "user_id": segment.user_id, "email": user.email, "full_name": user.full_name,
                "rfm_score": round(float(segment.rfm_score), 4),
                "monetary": round(float(segment.monetary), 4),
                "recency_days": round(float(segment.recency_days), 1),
                "confidence": round(float(segment.confidence), 4),
            }
            for segment, user in rows
        ]
