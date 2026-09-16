"""Business, AI, customer and product analytics for the admin dashboard.

Every number here is computed from the operational tables - there are no
hard-coded or illustrative figures.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.models import (
    Category,
    CustomerSegment,
    Order,
    OrderItem,
    Product,
    RecommendationLog,
    SearchEvent,
    User,
    UserEvent,
)
from app.repositories.product_repository import ProductRepository

ANALYTICS_TTL = 120
REVENUE_STATUSES = ("paid", "shipped", "delivered")


class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.cache = get_cache()
        self.products = ProductRepository(db)

    # ---- business ------------------------------------------------------
    def business_metrics(self, days: int = 30) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)
        prior = since - timedelta(days=days)

        def window(start: datetime, end: datetime) -> dict[str, float]:
            row = self.db.execute(
                select(func.count(Order.id), func.coalesce(func.sum(Order.total_amount), 0.0),
                       func.coalesce(func.avg(Order.total_amount), 0.0),
                       func.count(func.distinct(Order.user_id)))
                .where(Order.placed_at >= start, Order.placed_at < end,
                       Order.status.in_(REVENUE_STATUSES))
            ).one()
            return {"orders": int(row[0] or 0), "revenue": float(row[1] or 0.0),
                    "aov": float(row[2] or 0.0), "buyers": int(row[3] or 0)}

        current = window(since, now)
        previous = window(prior, since)

        sessions = int(self.db.execute(
            select(func.count(func.distinct(UserEvent.session_id))).where(UserEvent.occurred_at >= since)
        ).scalar_one() or 0)
        active_users = int(self.db.execute(
            select(func.count(func.distinct(UserEvent.user_id))).where(UserEvent.occurred_at >= since)
        ).scalar_one() or 0)
        purchase_sessions = int(self.db.execute(
            select(func.count(func.distinct(UserEvent.session_id)))
            .where(UserEvent.occurred_at >= since, UserEvent.event_type == "purchase")
        ).scalar_one() or 0)

        # Retention: buyers in this window who had also bought before it.
        repeat_buyers = int(self.db.execute(
            select(func.count(func.distinct(Order.user_id)))
            .where(Order.placed_at >= since, Order.status.in_(REVENUE_STATUSES),
                   Order.user_id.in_(
                       select(Order.user_id).where(Order.placed_at < since,
                                                   Order.status.in_(REVENUE_STATUSES))
                   ))
        ).scalar_one() or 0)

        def change(now_value: float, before: float) -> float | None:
            if before <= 0:
                return None
            return round((now_value - before) / before * 100, 2)

        return {
            "window_days": days,
            "revenue": round(current["revenue"], 2),
            "revenue_change_pct": change(current["revenue"], previous["revenue"]),
            "orders": current["orders"],
            "orders_change_pct": change(current["orders"], previous["orders"]),
            "average_order_value": round(current["aov"], 2),
            "aov_change_pct": change(current["aov"], previous["aov"]),
            "unique_buyers": current["buyers"],
            "active_users": active_users,
            "sessions": sessions,
            "conversion_rate": round(purchase_sessions / sessions, 4) if sessions else 0.0,
            "repeat_buyers": repeat_buyers,
            "retention_rate": round(repeat_buyers / current["buyers"], 4) if current["buyers"] else 0.0,
            "total_customers": int(self.db.execute(select(func.count(User.id))).scalar_one() or 0),
            "total_products": int(self.db.execute(
                select(func.count(Product.id)).where(Product.is_active.is_(True))
            ).scalar_one() or 0),
        }

    def revenue_timeseries(self, days: int = 30) -> list[dict[str, Any]]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(func.date(Order.placed_at), func.coalesce(func.sum(Order.total_amount), 0.0),
                   func.count(Order.id))
            .where(Order.placed_at >= since, Order.status.in_(REVENUE_STATUSES))
            .group_by(func.date(Order.placed_at))
            .order_by(func.date(Order.placed_at))
        ).all()
        return [{"date": str(day), "revenue": round(float(revenue or 0.0), 2), "orders": int(orders or 0)}
                for day, revenue, orders in rows]

    # ---- AI / ML -------------------------------------------------------
    def ai_metrics(self, days: int = 30) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)

        rec = self.db.execute(
            select(func.count(RecommendationLog.id),
                   func.sum(case((RecommendationLog.clicked.is_(True), 1), else_=0)),
                   func.sum(case((RecommendationLog.converted.is_(True), 1), else_=0)))
            .where(RecommendationLog.served_at >= since)
        ).one()
        impressions = int(rec[0] or 0)
        rec_clicks = int(rec[1] or 0)
        rec_conversions = int(rec[2] or 0)

        search = self.db.execute(
            select(func.count(SearchEvent.id),
                   func.sum(case((SearchEvent.clicked_product_id.isnot(None), 1), else_=0)),
                   func.sum(case((SearchEvent.converted.is_(True), 1), else_=0)),
                   func.avg(SearchEvent.latency_ms),
                   func.sum(case((SearchEvent.result_count == 0, 1), else_=0)))
            .where(SearchEvent.occurred_at >= since)
        ).one()
        searches = int(search[0] or 0)
        search_clicks = int(search[1] or 0)
        search_conversions = int(search[2] or 0)
        zero_results = int(search[4] or 0)

        by_strategy = self.db.execute(
            select(RecommendationLog.strategy, func.count(RecommendationLog.id),
                   func.sum(case((RecommendationLog.clicked.is_(True), 1), else_=0)))
            .where(RecommendationLog.served_at >= since)
            .group_by(RecommendationLog.strategy)
        ).all()

        return {
            "window_days": days,
            "recommendation": {
                "impressions": impressions,
                "clicks": rec_clicks,
                "conversions": rec_conversions,
                "ctr": round(rec_clicks / impressions, 5) if impressions else 0.0,
                "conversion_rate": round(rec_conversions / impressions, 5) if impressions else 0.0,
                "by_strategy": [
                    {"strategy": s, "impressions": int(n), "clicks": int(c or 0),
                     "ctr": round(int(c or 0) / int(n), 5) if n else 0.0}
                    for s, n, c in by_strategy
                ],
            },
            "search": {
                "searches": searches,
                "clicks": search_clicks,
                "conversions": search_conversions,
                "ctr": round(search_clicks / searches, 5) if searches else 0.0,
                "conversion_rate": round(search_conversions / searches, 5) if searches else 0.0,
                "abandonment_rate": round((searches - search_clicks) / searches, 5) if searches else 0.0,
                "zero_result_rate": round(zero_results / searches, 5) if searches else 0.0,
                "avg_latency_ms": round(float(search[3] or 0.0), 2),
            },
        }

    # ---- customers -----------------------------------------------------
    def customer_analytics(self, days: int = 30) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)

        new_users = int(self.db.execute(
            select(func.count(User.id)).where(User.created_at >= since)
        ).scalar_one() or 0)
        returning = int(self.db.execute(
            select(func.count(func.distinct(Order.user_id)))
            .where(Order.placed_at >= since, Order.status.in_(REVENUE_STATUSES))
        ).scalar_one() or 0)

        version = self.db.execute(
            select(CustomerSegment.model_version).order_by(CustomerSegment.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        segments = []
        if version:
            rows = self.db.execute(
                select(CustomerSegment.segment_label, func.count(CustomerSegment.id),
                       func.avg(CustomerSegment.monetary))
                .where(CustomerSegment.model_version == version)
                .group_by(CustomerSegment.segment_label)
                .order_by(func.count(CustomerSegment.id).desc())
            ).all()
            segments = [
                {"label": label, "name": label.replace("_", " ").title(), "count": int(count),
                 "avg_monetary": round(float(monetary or 0), 4)}
                for label, count, monetary in rows
            ]

        top_customers = self.db.execute(
            select(User.id, User.full_name, User.email, func.sum(Order.total_amount).label("spend"),
                   func.count(Order.id))
            .join(Order, Order.user_id == User.id)
            .where(Order.status.in_(REVENUE_STATUSES))
            .group_by(User.id, User.full_name, User.email)
            .order_by(func.sum(Order.total_amount).desc())
            .limit(10)
        ).all()

        return {
            "window_days": days,
            "new_users": new_users,
            "returning_buyers": returning,
            "segments": segments,
            "high_value_count": sum(s["count"] for s in segments if s["label"] == "high_value"),
            "at_risk_count": sum(s["count"] for s in segments if s["label"] == "at_risk"),
            "top_customers": [
                {"user_id": int(uid), "name": name, "email": email,
                 "total_spend": round(float(spend or 0), 2), "orders": int(orders)}
                for uid, name, email, spend, orders in top_customers
            ],
        }

    # ---- products ------------------------------------------------------
    def product_analytics(self, limit: int = 10) -> dict[str, Any]:
        best = self.products.best_sellers(limit=limit)
        low_stock = self.products.low_inventory(threshold=10, limit=limit)
        high_conversion = self.products.conversion_stats(limit=limit, best=True)
        poor = self.products.conversion_stats(limit=limit, best=False)

        def brief(product: Product) -> dict[str, Any]:
            return {
                "id": product.id, "title": product.title, "sku": product.sku,
                "price": round(float(product.price), 2), "rating": float(product.rating_avg),
                "inventory": int(product.inventory),
            }

        return {
            "best_sellers": [{**brief(row["product"]), "units_sold": row["units_sold"],
                              "revenue": row["revenue"]} for row in best],
            "low_inventory": [brief(p) for p in low_stock],
            "high_conversion": [{**brief(row["product"]), "conversion_rate": row["conversion_rate"],
                                 "views": row["views"]} for row in high_conversion],
            "poor_performers": [{**brief(row["product"]), "conversion_rate": row["conversion_rate"],
                                 "views": row["views"]} for row in poor],
            "category_breakdown": self.category_breakdown(),
        }

    def category_breakdown(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.db.execute(
            select(Category.name, Category.slug, func.count(func.distinct(Product.id)),
                   func.coalesce(func.sum(OrderItem.line_total), 0.0))
            .join(Product, Product.category_id == Category.id)
            .outerjoin(OrderItem, OrderItem.product_id == Product.id)
            .group_by(Category.id, Category.name, Category.slug)
            .order_by(func.coalesce(func.sum(OrderItem.line_total), 0.0).desc())
            .limit(limit)
        ).all()
        return [
            {"category": name, "slug": slug, "products": int(products), "revenue": round(float(revenue or 0), 2)}
            for name, slug, products, revenue in rows
        ]

    # ---- composite -----------------------------------------------------
    def dashboard(self, days: int = 30) -> dict[str, Any]:
        key = self.cache.key("analytics:dashboard", days)

        def produce() -> dict[str, Any]:
            from app.sentiment.service import SentimentService

            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "window_days": days,
                "business": self.business_metrics(days),
                "ai": self.ai_metrics(days),
                "customers": self.customer_analytics(days),
                "products": self.product_analytics(),
                "sentiment": SentimentService(self.db).distribution_overview(),
                "revenue_timeseries": self.revenue_timeseries(days),
                "event_funnel": self.event_funnel(days),
            }

        return self.cache.get_or_set(key, produce, ANALYTICS_TTL)

    def event_funnel(self, days: int = 30) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(UserEvent.event_type, func.count(UserEvent.id))
            .where(UserEvent.occurred_at >= since)
            .group_by(UserEvent.event_type)
        ).all()
        counts = {event_type: int(count) for event_type, count in rows}
        views = counts.get("product_view", 0)
        carts = counts.get("add_to_cart", 0)
        purchases = counts.get("purchase", 0)
        return {
            "counts": counts,
            "view_to_cart": round(carts / views, 5) if views else 0.0,
            "cart_to_purchase": round(purchases / carts, 5) if carts else 0.0,
            "view_to_purchase": round(purchases / views, 5) if views else 0.0,
        }
