"""Feature store: versioned, timestamped features with a Redis-backed read path.

Offline (PostgreSQL/SQLite) is the source of truth via `user_features` and
`product_features`. Online reads go through the cache so serving stays fast, and
fall back to the database on a cache miss.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.logging_config import get_logger
from app.models import ProductFeature, UserFeature

logger = get_logger("feature_store")

USER_FEATURES = (
    "user_purchase_frequency",
    "user_average_order_value",
    "user_total_spend",
    "user_recency_days",
    "user_session_count",
    "user_distinct_products",
    "user_discount_affinity",
    "user_review_count",
)
PRODUCT_FEATURES = (
    "product_popularity",
    "product_conversion_rate",
    "product_rating",
    "product_demand_30d",
    "product_price_trend",
    "product_view_count",
    "product_cart_rate",
    "product_return_rate",
)
DEFAULT_VERSION = "v1"
ONLINE_TTL = 600


@dataclass(slots=True)
class FeatureVector:
    entity_id: int
    values: dict[str, float]
    version: str
    computed_at: str | None = None

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.values.get(name, default))

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "version": self.version,
            "computed_at": self.computed_at,
            "features": self.values,
        }


class FeatureStore:
    """Thin abstraction over the two feature tables."""

    def __init__(self, db: Session, version: str = DEFAULT_VERSION):
        self.db = db
        self.version = version
        self.cache = get_cache()

    # ---- write path ---------------------------------------------------
    def write_user_features(self, rows: Iterable[dict[str, Any]], version: str | None = None) -> int:
        return self._write(UserFeature, "user_id", rows, version)

    def write_product_features(self, rows: Iterable[dict[str, Any]], version: str | None = None) -> int:
        return self._write(ProductFeature, "product_id", rows, version)

    def _write(self, model, id_field: str, rows: Iterable[dict[str, Any]], version: str | None) -> int:
        version = version or self.version
        now = datetime.now(timezone.utc)
        payload = []
        entity_ids: set[int] = set()
        for row in rows:
            entity_id = int(row[id_field])
            entity_ids.add(entity_id)
            for name, value in row.get("features", {}).items():
                payload.append({
                    id_field: entity_id,
                    "feature_name": name,
                    "feature_value": float(value),
                    "feature_version": version,
                    "computed_at": now,
                })
        if not payload:
            return 0
        # Replace this version's rows for the affected entities (idempotent rebuild).
        self.db.execute(
            delete(model).where(
                getattr(model, id_field).in_(entity_ids), model.feature_version == version
            )
        )
        for start in range(0, len(payload), 1000):
            self.db.bulk_insert_mappings(model, payload[start:start + 1000])
        self.db.commit()
        namespace = "features:user" if model is UserFeature else "features:product"
        for entity_id in entity_ids:
            self.cache.invalidate(self.cache.key(namespace, version, entity_id))
        logger.info("features_written", table=model.__tablename__, rows=len(payload), version=version)
        return len(payload)

    # ---- read path ----------------------------------------------------
    def get_user_features(self, user_id: int, version: str | None = None) -> FeatureVector:
        return self._get(UserFeature, "user_id", "features:user", user_id, version)

    def get_product_features(self, product_id: int, version: str | None = None) -> FeatureVector:
        return self._get(ProductFeature, "product_id", "features:product", product_id, version)

    def _get(self, model, id_field: str, namespace: str, entity_id: int, version: str | None) -> FeatureVector:
        version = version or self.version
        key = self.cache.key(namespace, version, entity_id)
        cached = self.cache.get_json(key)
        if cached is not None:
            return FeatureVector(entity_id, cached.get("features", {}), version, cached.get("computed_at"))

        rows = self.db.execute(
            select(model.feature_name, model.feature_value, model.computed_at).where(
                getattr(model, id_field) == entity_id, model.feature_version == version
            )
        ).all()
        values = {name: float(value) for name, value, _ in rows}
        computed_at = max((r[2] for r in rows), default=None)
        payload = {"features": values, "computed_at": computed_at.isoformat() if computed_at else None}
        self.cache.set_json(key, payload, ONLINE_TTL)
        return FeatureVector(entity_id, values, version, payload["computed_at"])

    def get_many_product_features(self, product_ids: Sequence[int],
                                  version: str | None = None) -> dict[int, FeatureVector]:
        version = version or self.version
        if not product_ids:
            return {}
        rows = self.db.execute(
            select(ProductFeature.product_id, ProductFeature.feature_name, ProductFeature.feature_value)
            .where(ProductFeature.product_id.in_(list(product_ids)),
                   ProductFeature.feature_version == version)
        ).all()
        grouped: dict[int, dict[str, float]] = {int(pid): {} for pid in product_ids}
        for pid, name, value in rows:
            grouped.setdefault(int(pid), {})[name] = float(value)
        return {pid: FeatureVector(pid, values, version) for pid, values in grouped.items()}

    # ---- metadata -----------------------------------------------------
    def versions(self) -> dict[str, list[str]]:
        user_versions = [
            v for (v,) in self.db.execute(select(UserFeature.feature_version).distinct()).all()
        ]
        product_versions = [
            v for (v,) in self.db.execute(select(ProductFeature.feature_version).distinct()).all()
        ]
        return {"user": sorted(user_versions), "product": sorted(product_versions)}

    def coverage(self, version: str | None = None) -> dict[str, Any]:
        from sqlalchemy import func

        version = version or self.version
        user_rows = self.db.execute(
            select(func.count(func.distinct(UserFeature.user_id))).where(UserFeature.feature_version == version)
        ).scalar_one()
        product_rows = self.db.execute(
            select(func.count(func.distinct(ProductFeature.product_id)))
            .where(ProductFeature.feature_version == version)
        ).scalar_one()
        return {
            "version": version,
            "users_with_features": int(user_rows or 0),
            "products_with_features": int(product_rows or 0),
            "user_feature_names": list(USER_FEATURES),
            "product_feature_names": list(PRODUCT_FEATURES),
        }
