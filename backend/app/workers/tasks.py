"""Background jobs.

Runs under Celery when a broker is configured, and under an in-process thread
scheduler otherwise, so the same job code works in both deployment modes.
Every task is idempotent and safe to re-run.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import func, select

from app.core.db import session_scope
from app.core.logging_config import get_logger
from app.ml.model_store import get_model_store

logger = get_logger("worker")


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite does not persist timezone info, so timestamps come back naive.

    Normalise to UTC-aware before any arithmetic, otherwise subtracting a naive
    column value from `datetime.now(timezone.utc)` raises.
    """
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def flush_prediction_logs() -> dict[str, Any]:
    """Persist sampled inference telemetry buffered in memory."""
    written = get_model_store().flush_predictions()
    return {"task": "flush_prediction_logs", "rows_written": written}


def refresh_search_index() -> dict[str, Any]:
    """Rebuild the BM25 + vector index so catalogue edits become searchable."""
    from app.search.index import get_index_manager

    with session_scope() as db:
        index = get_index_manager().rebuild(db)
        return {"task": "refresh_search_index", **index.stats()}


def reload_models() -> dict[str, Any]:
    """Hot-reload any model whose registry version has advanced."""
    updated = get_model_store().check_for_updates()
    return {"task": "reload_models", "updated": updated}


def compute_feature_store() -> dict[str, Any]:
    """Recompute the offline feature store from current operational data."""
    from app.models import Order, OrderItem, Product, Review, UserEvent
    from app.services.feature_store import FeatureStore

    now = datetime.now(timezone.utc)
    with session_scope() as db:
        store = FeatureStore(db)

        order_rows = db.execute(
            select(Order.user_id, func.count(Order.id), func.coalesce(func.sum(Order.total_amount), 0.0),
                   func.coalesce(func.avg(Order.total_amount), 0.0), func.max(Order.placed_at),
                   func.coalesce(func.sum(Order.discount_amount), 0.0))
            .where(Order.status.in_(("paid", "shipped", "delivered")))
            .group_by(Order.user_id)
        ).all()
        sessions = dict(db.execute(
            select(UserEvent.user_id, func.count(func.distinct(UserEvent.session_id)))
            .group_by(UserEvent.user_id)
        ).all())
        distinct_products = dict(db.execute(
            select(Order.user_id, func.count(func.distinct(OrderItem.product_id)))
            .join(OrderItem, OrderItem.order_id == Order.id).group_by(Order.user_id)
        ).all())
        review_counts = dict(db.execute(
            select(Review.user_id, func.count(Review.id)).group_by(Review.user_id)
        ).all())

        user_rows = []
        for user_id, orders, spend, aov, last_order, discount in order_rows:
            if user_id is None:
                continue
            last_order_utc = _as_utc(last_order)
            recency = (now - last_order_utc).days if last_order_utc else 999
            user_rows.append({"user_id": int(user_id), "features": {
                "user_purchase_frequency": float(orders or 0),
                "user_average_order_value": float(aov or 0.0),
                "user_total_spend": float(spend or 0.0),
                "user_recency_days": float(recency),
                "user_session_count": float(sessions.get(user_id, 0) or 0),
                "user_distinct_products": float(distinct_products.get(user_id, 0) or 0),
                "user_discount_affinity": float(discount or 0.0) / float(spend) if spend else 0.0,
                "user_review_count": float(review_counts.get(user_id, 0) or 0),
            }})
        user_written = store.write_user_features(user_rows)

        views = dict(db.execute(
            select(UserEvent.product_id, func.count(UserEvent.id))
            .where(UserEvent.event_type.in_(("product_view", "click")))
            .group_by(UserEvent.product_id)
        ).all())
        purchases = dict(db.execute(
            select(UserEvent.product_id, func.count(UserEvent.id))
            .where(UserEvent.event_type == "purchase").group_by(UserEvent.product_id)
        ).all())
        carts = dict(db.execute(
            select(UserEvent.product_id, func.count(UserEvent.id))
            .where(UserEvent.event_type == "add_to_cart").group_by(UserEvent.product_id)
        ).all())
        since = now - timedelta(days=30)
        demand_30d = dict(db.execute(
            select(OrderItem.product_id, func.coalesce(func.sum(OrderItem.quantity), 0))
            .join(Order, Order.id == OrderItem.order_id)
            .where(Order.placed_at >= since, Order.status.in_(("paid", "shipped", "delivered")))
            .group_by(OrderItem.product_id)
        ).all())

        product_rows = []
        for product in db.execute(select(Product)).scalars().all():
            view_count = float(views.get(product.id, 0) or 0)
            purchase_count = float(purchases.get(product.id, 0) or 0)
            cart_count = float(carts.get(product.id, 0) or 0)
            product_rows.append({"product_id": int(product.id), "features": {
                "product_popularity": float(np.log1p(product.rating_count)),
                "product_conversion_rate": (purchase_count / view_count) if view_count >= 3 else 0.0,
                "product_rating": float(product.rating_avg),
                "product_demand_30d": float(demand_30d.get(product.id, 0) or 0),
                "product_price_trend": float(product.discount_pct),
                "product_view_count": view_count,
                "product_cart_rate": (cart_count / view_count) if view_count >= 3 else 0.0,
                "product_return_rate": 0.0,
            }})
        product_written = store.write_product_features(product_rows)

    return {"task": "compute_feature_store", "user_features": user_written,
            "product_features": product_written}


def score_pending_reviews(limit: int = 1000) -> dict[str, Any]:
    """Classify reviews created since the last sentiment batch run."""
    from app.models import Review

    loaded = get_model_store().try_get("sentiment")
    if loaded is None:
        return {"task": "score_pending_reviews", "skipped": "sentiment model unavailable"}

    artifact = loaded.artifact
    with session_scope() as db:
        pending = db.execute(
            select(Review).where(Review.sentiment_label.is_(None)).limit(limit)
        ).scalars().all()
        if not pending:
            return {"task": "score_pending_reviews", "scored": 0}
        texts = [f"{r.title}. {r.body}" for r in pending]
        labels, scores = artifact.predict(texts)
        for review, label, score, text in zip(pending, labels, scores, texts):
            review.sentiment_label = label
            review.sentiment_score = float(score)
            review.sentiment_model_version = loaded.version
            review.aspects = artifact.extract_aspects(text)
    return {"task": "score_pending_reviews", "scored": len(pending)}


def refresh_trending_cache() -> dict[str, Any]:
    """Warm the trending/deals caches so the homepage never pays a cold miss."""
    from app.core.cache import get_cache
    from app.recommendation.service import RecommendationService

    get_cache().invalidate_namespace("trending")
    with session_scope() as db:
        service = RecommendationService(db)
        trending = service.trending(20)
        deals = service.deals(20)
    return {"task": "refresh_trending_cache", "trending": trending["count"], "deals": deals["count"]}


#: name -> (callable, interval in seconds)
SCHEDULE: dict[str, tuple[Any, int]] = {
    "flush_prediction_logs": (flush_prediction_logs, 60),
    "refresh_trending_cache": (refresh_trending_cache, 300),
    "score_pending_reviews": (score_pending_reviews, 600),
    "reload_models": (reload_models, 900),
    "compute_feature_store": (compute_feature_store, 1800),
    "refresh_search_index": (refresh_search_index, 3600),
}

ALL_TASKS = {name: fn for name, (fn, _) in SCHEDULE.items()}


def run_task(name: str) -> dict[str, Any]:
    task = ALL_TASKS.get(name)
    if task is None:
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(sorted(ALL_TASKS))}")
    logger.info("task_started", task=name)
    try:
        result = task()
        logger.info("task_completed", **result)
        return result
    except Exception as exc:  # noqa: BLE001 - a failing job must not kill the worker
        logger.error("task_failed", task=name, error=str(exc), exc_info=True)
        return {"task": name, "error": f"{type(exc).__name__}: {exc}"}
