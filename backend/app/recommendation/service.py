"""Recommendation service.

Wraps the trained hybrid recommender and adds the serving concerns the model
itself should not know about: cold-start routing, cache-aside reads, impression
logging (which powers CTR/conversion metrics) and human-readable explanations.

Every surface degrades gracefully: if the model artifact is missing, the service
falls back to database-driven popularity so the storefront still works.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.config import settings
from app.core.errors import ModelNotAvailableError, NotFoundError
from app.core.logging_config import get_logger
from app.ml.model_store import get_model_store
from app.models import OrderItem, Product, RecommendationLog, UserEvent
from app.services.personalization import PersonalizationService

logger = get_logger("recommendation")

REC_CACHE_TTL = 180
TRENDING_WINDOW_DAYS = 14


class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        self.cache = get_cache()
        self.store = get_model_store()
        self.personalization = PersonalizationService(db)

    # ---- helpers -------------------------------------------------------
    @property
    def _artifact(self):
        return self.store.artifact("recommendation")

    def _try_artifact(self):
        loaded = self.store.try_get("recommendation")
        return loaded.artifact if loaded else None

    def _hydrate(self, scored: Sequence[tuple[int, float, dict[str, float]]]) -> list[dict[str, Any]]:
        """Attach product rows to scored ids, preserving rank order."""
        if not scored:
            return []
        ids = [pid for pid, _, _ in scored]
        products = {
            p.id: p for p in self.db.execute(select(Product).where(Product.id.in_(ids))).scalars().all()
        }
        out: list[dict[str, Any]] = []
        for rank, (pid, score, components) in enumerate(scored, start=1):
            product = products.get(pid)
            if product is None:
                continue
            out.append({
                "product": product,
                "product_id": pid,
                "score": round(float(score), 6),
                "rank": rank,
                "components": components,
                "explanation": self._explain(components, product),
            })
        return out

    @staticmethod
    def _explain(components: dict[str, float], product: Product) -> str:
        """Turn score components into a sentence a shopper would understand."""
        if not components:
            return f"Popular in {product.category.name}" if product.category else "Popular right now"
        labels = {
            "collaborative": "customers with similar taste bought this",
            "content": "it is similar to items you viewed",
            "popularity": "it is trending across the store",
            "personalization": "it matches your preferred categories and brands",
            "business": "it is well stocked and highly rated",
        }
        top = sorted(components.items(), key=lambda kv: -kv[1])
        drivers = [labels[name] for name, value in top[:2] if value > 0.05 and name in labels]
        if not drivers:
            return "Recommended based on overall store trends"
        return "Recommended because " + " and ".join(drivers)

    # ---- surfaces ------------------------------------------------------
    def for_user(self, user_id: int | None, limit: int = 10, *, surface: str = "homepage",
                 exclude: Iterable[int] = (), weights: dict[str, float] | None = None,
                 session_id: str | None = None, log_impressions: bool = True) -> dict[str, Any]:
        profile = self.personalization.get_profile(user_id, session_id)
        artifact = self._try_artifact()

        if artifact is None:
            items = self._popularity_fallback(limit, exclude)
            return self._package(items, "popularity_fallback", "unavailable", surface, user_id, log_impressions)

        # Cold start: unknown to the model or too few interactions.
        if user_id is None or not artifact.knows_user(user_id) or profile.is_cold_start:
            strategy = "cold_start"
            preferred_category = None
            if profile.category_affinity:
                preferred_category = max(profile.category_affinity.items(), key=lambda kv: kv[1])[0]
            with self.store.timed("recommendation", entity_type="user", entity_id=user_id) as box:
                scored_raw = artifact.cold_start(k=limit + len(list(exclude)), category_id=preferred_category)
                if len(scored_raw) < limit:
                    scored_raw = artifact.cold_start(k=limit + len(list(exclude)))
                box["prediction"] = float(scored_raw[0][1]) if scored_raw else 0.0
            excluded = set(exclude)
            scored = [(pid, score, {"popularity": score}) for pid, score in scored_raw if pid not in excluded][:limit]
        else:
            strategy = "hybrid"
            merged = {**settings.recommendation_weights, **(artifact.weights or {}), **(weights or {})}
            with self.store.timed("recommendation", entity_type="user", entity_id=user_id) as box:
                scored = artifact.recommend_for_user(
                    user_id, k=limit, exclude=exclude, weights=merged,
                    category_affinity=profile.category_affinity, brand_affinity=profile.brand_affinity,
                )
                box["prediction"] = float(scored[0][1]) if scored else 0.0

        items = self._hydrate(scored)
        version = self.store.get("recommendation").version
        return self._package(items, strategy, version, surface, user_id, log_impressions)

    def similar_products(self, product_id: int, limit: int = 8) -> dict[str, Any]:
        product = self.db.get(Product, product_id)
        if product is None:
            raise NotFoundError(f"Product {product_id} was not found.")
        artifact = self._try_artifact()
        if artifact is None:
            items = self._same_category_fallback(product, limit)
            return self._package(items, "same_category_fallback", "unavailable", "product_detail", None, False)

        key = self.cache.key("similar", product_id, limit)
        cached = self.cache.get_json(key)
        if cached is None:
            with self.store.timed("recommendation", entity_type="product", entity_id=product_id) as box:
                pairs = artifact.similar_items(product_id, k=limit)
                box["prediction"] = float(pairs[0][1]) if pairs else 0.0
            cached = [[int(pid), float(score)] for pid, score in pairs]
            self.cache.set_json(key, cached, REC_CACHE_TTL)

        scored = [(int(pid), float(score), {"content": float(score)}) for pid, score in cached]
        items = self._hydrate(scored)
        for item in items:
            item["explanation"] = f"Similar to {product.title}"
        return self._package(items, "content_similarity", self.store.get("recommendation").version,
                             "product_detail", None, False)

    def frequently_bought_together(self, product_id: int, limit: int = 4) -> dict[str, Any]:
        product = self.db.get(Product, product_id)
        if product is None:
            raise NotFoundError(f"Product {product_id} was not found.")
        artifact = self._try_artifact()

        pairs: list[tuple[int, float]] = []
        if artifact is not None:
            pairs = artifact.frequently_bought_together(product_id, k=limit)
        if not pairs:
            pairs = self._cooccurrence_from_db(product_id, limit)

        scored = [(pid, score, {"co_purchase": score}) for pid, score in pairs]
        items = self._hydrate(scored)
        for item in items:
            item["explanation"] = "Frequently bought together with this item"
        strategy = "co_purchase_model" if artifact is not None and pairs else "co_purchase_sql"
        return self._package(items, strategy, "n/a", "product_detail", None, False)

    def trending(self, limit: int = 10, category_id: int | None = None) -> dict[str, Any]:
        key = self.cache.key("trending", limit, category_id or "all")

        def produce() -> list[list[float]]:
            since = datetime.now(timezone.utc) - timedelta(days=TRENDING_WINDOW_DAYS)
            stmt = (
                select(UserEvent.product_id, func.count(UserEvent.id).label("n"))
                .join(Product, Product.id == UserEvent.product_id)
                .where(UserEvent.occurred_at >= since,
                       UserEvent.event_type.in_(("product_view", "add_to_cart", "purchase")),
                       Product.is_active.is_(True))
                .group_by(UserEvent.product_id)
                .order_by(func.count(UserEvent.id).desc())
                .limit(limit)
            )
            if category_id is not None:
                stmt = stmt.where(Product.category_id == category_id)
            return [[int(pid), float(n)] for pid, n in self.db.execute(stmt).all()]

        rows = self.cache.get_or_set(key, produce, REC_CACHE_TTL)
        if not rows:
            items = self._popularity_fallback(limit, ())
            return self._package(items, "popularity_fallback", "n/a", "trending", None, False)
        top = max((r[1] for r in rows), default=1.0) or 1.0
        scored = [(int(pid), float(n) / top, {"trending": round(float(n) / top, 5)}) for pid, n in rows]
        items = self._hydrate(scored)
        for item in items:
            item["explanation"] = f"Trending over the last {TRENDING_WINDOW_DAYS} days"
        return self._package(items, "trending_events", "n/a", "trending", None, False)

    def recently_viewed(self, user_id: int | None, session_id: str | None, limit: int = 10) -> dict[str, Any]:
        profile = self.personalization.get_profile(user_id, session_id)
        ids = profile.recently_viewed[:limit]
        if not ids:
            return self._package([], "recently_viewed", "n/a", "homepage", user_id, False)
        products = {p.id: p for p in self.db.execute(select(Product).where(Product.id.in_(ids))).scalars().all()}
        items = [
            {"product": products[pid], "product_id": pid, "score": 1.0 - i * 0.01, "rank": i + 1,
             "components": {}, "explanation": "You viewed this recently"}
            for i, pid in enumerate(ids) if pid in products
        ]
        return self._package(items, "recently_viewed", "n/a", "homepage", user_id, False)

    def deals(self, limit: int = 10) -> dict[str, Any]:
        rows = self.db.execute(
            select(Product).where(Product.is_active.is_(True), Product.discount_pct > 0, Product.inventory > 0)
            .order_by(Product.discount_pct.desc(), Product.rating_avg.desc()).limit(limit)
        ).scalars().all()
        items = [
            {"product": p, "product_id": p.id, "score": float(p.discount_pct) / 100.0, "rank": i + 1,
             "components": {"discount": round(float(p.discount_pct) / 100.0, 4)},
             "explanation": f"{p.discount_pct:.0f}% off right now"}
            for i, p in enumerate(rows)
        ]
        return self._package(items, "deals", "n/a", "homepage", None, False)

    def continue_shopping(self, user_id: int | None, session_id: str | None, limit: int = 8) -> dict[str, Any]:
        """Items left in the cart, then similar items to the last thing viewed."""
        profile = self.personalization.get_profile(user_id, session_id)
        ids = [pid for pid in profile.cart if pid not in profile.purchased][:limit]
        items: list[dict[str, Any]] = []
        if ids:
            products = {p.id: p for p in self.db.execute(select(Product).where(Product.id.in_(ids))).scalars().all()}
            items = [
                {"product": products[pid], "product_id": pid, "score": 1.0, "rank": i + 1, "components": {},
                 "explanation": "Still in your cart"}
                for i, pid in enumerate(ids) if pid in products
            ]
        if len(items) < limit and profile.recently_viewed:
            try:
                similar = self.similar_products(profile.recently_viewed[0], limit - len(items))
                for extra in similar["items"]:
                    extra["explanation"] = "Because you viewed a similar item"
                    items.append(extra)
            except (NotFoundError, ModelNotAvailableError):
                pass
        return self._package(items[:limit], "continue_shopping", "n/a", "homepage", user_id, False)

    def personalized_homepage(self, user_id: int | None, session_id: str | None = None,
                              per_section: int = 8) -> dict[str, Any]:
        """Assemble every homepage rail in one call, tailored to the profile."""
        profile = self.personalization.get_profile(user_id, session_id)
        sections: list[dict[str, Any]] = []

        def add(title: str, key: str, payload: dict[str, Any], subtitle: str = "") -> None:
            if payload["items"]:
                sections.append({
                    "key": key, "title": title, "subtitle": subtitle,
                    "strategy": payload["strategy"], "items": payload["items"],
                })

        if not profile.is_cold_start:
            add("Recommended for you", "for_you",
                self.for_user(user_id, per_section, surface="homepage", session_id=session_id),
                "Based on your browsing and purchase history")
            add("Continue shopping", "continue",
                self.continue_shopping(user_id, session_id, per_section))
            add("Recently viewed", "recent", self.recently_viewed(user_id, session_id, per_section))
        else:
            add("Popular right now", "for_you",
                self.for_user(user_id, per_section, surface="homepage", session_id=session_id),
                "Top picks to get you started")

        add("Trending this week", "trending", self.trending(per_section))
        add("Today's deals", "deals", self.deals(per_section), "Biggest discounts in the catalogue")

        # A rail for the shopper's strongest category affinity.
        if profile.category_affinity:
            top_category = max(profile.category_affinity.items(), key=lambda kv: kv[1])[0]
            category = self.db.execute(
                select(Product.category_id, func.count(Product.id)).where(Product.category_id == top_category)
                .group_by(Product.category_id)
            ).first()
            if category:
                payload = self.trending(per_section, category_id=int(top_category))
                from app.models import Category

                category_row = self.db.get(Category, int(top_category))
                add(f"More in {category_row.name}" if category_row else "More for you", "category_affinity",
                    payload, "Matched to your favourite category")

        return {
            "user_id": user_id,
            "is_personalized": not profile.is_cold_start,
            "segment": profile.segment,
            "sections": sections,
            "profile_summary": {
                "interaction_count": profile.interaction_count,
                "order_count": profile.order_count,
                "top_categories": list(profile.category_affinity)[:3],
            },
        }

    # ---- fallbacks -----------------------------------------------------
    def _popularity_fallback(self, limit: int, exclude: Iterable[int]) -> list[dict[str, Any]]:
        excluded = set(exclude)
        rows = self.db.execute(
            select(Product).where(Product.is_active.is_(True), Product.inventory > 0)
            .order_by(Product.rating_count.desc(), Product.rating_avg.desc())
            .limit(limit + len(excluded))
        ).scalars().all()
        return [
            {"product": p, "product_id": p.id, "score": 0.5, "rank": i + 1, "components": {},
             "explanation": "Popular across the store"}
            for i, p in enumerate([p for p in rows if p.id not in excluded][:limit])
        ]

    def _same_category_fallback(self, product: Product, limit: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            select(Product).where(Product.category_id == product.category_id, Product.id != product.id,
                                  Product.is_active.is_(True))
            .order_by(Product.rating_avg.desc()).limit(limit)
        ).scalars().all()
        return [
            {"product": p, "product_id": p.id, "score": 0.4, "rank": i + 1, "components": {},
             "explanation": f"Also in {product.category.name}" if product.category else "Related product"}
            for i, p in enumerate(rows)
        ]

    def _cooccurrence_from_db(self, product_id: int, limit: int) -> list[tuple[int, float]]:
        """SQL co-purchase: products appearing in the same orders as this one."""
        peer = select(OrderItem.order_id).where(OrderItem.product_id == product_id).subquery()
        rows = self.db.execute(
            select(OrderItem.product_id, func.count(OrderItem.id).label("n"))
            .where(OrderItem.order_id.in_(select(peer.c.order_id)), OrderItem.product_id != product_id)
            .group_by(OrderItem.product_id)
            .order_by(func.count(OrderItem.id).desc())
            .limit(limit)
        ).all()
        top = max((float(n) for _, n in rows), default=1.0) or 1.0
        return [(int(pid), float(n) / top) for pid, n in rows]

    # ---- packaging & logging -------------------------------------------
    def _package(self, items: list[dict[str, Any]], strategy: str, version: str, surface: str,
                 user_id: int | None, log_impressions: bool) -> dict[str, Any]:
        if log_impressions and items:
            self._log_impressions(items, surface, strategy, version, user_id)
        return {"items": items, "strategy": strategy, "model_version": version, "count": len(items)}

    def _log_impressions(self, items: list[dict[str, Any]], surface: str, strategy: str, version: str,
                         user_id: int | None) -> None:
        try:
            now = datetime.now(timezone.utc)
            self.db.bulk_insert_mappings(RecommendationLog, [
                {
                    "user_id": user_id, "product_id": item["product_id"], "surface": surface,
                    "strategy": strategy, "model_version": version, "score": float(item["score"]),
                    "rank": item["rank"], "explanation": item.get("components", {}),
                    "clicked": False, "converted": False, "served_at": now,
                }
                for item in items
            ])
            self.db.commit()
        except Exception as exc:  # noqa: BLE001 - impression logging is best-effort
            self.db.rollback()
            logger.warning("impression_log_failed", error=str(exc))

    def mark_clicked(self, user_id: int | None, product_id: int, surface: str = "homepage") -> int:
        """Attribute a click back to the most recent impression of that product."""
        since = datetime.now(timezone.utc) - timedelta(hours=6)
        stmt = (
            select(RecommendationLog)
            .where(RecommendationLog.product_id == product_id, RecommendationLog.served_at >= since,
                   RecommendationLog.clicked.is_(False))
            .order_by(RecommendationLog.served_at.desc()).limit(1)
        )
        if user_id is not None:
            stmt = stmt.where(RecommendationLog.user_id == user_id)
        row = self.db.execute(stmt).scalars().first()
        if row is None:
            return 0
        row.clicked = True
        self.db.commit()
        return 1

    def mark_converted(self, user_id: int | None, product_id: int) -> int:
        since = datetime.now(timezone.utc) - timedelta(days=2)
        stmt = (
            select(RecommendationLog)
            .where(RecommendationLog.product_id == product_id, RecommendationLog.served_at >= since,
                   RecommendationLog.converted.is_(False))
            .order_by(RecommendationLog.served_at.desc()).limit(1)
        )
        if user_id is not None:
            stmt = stmt.where(RecommendationLog.user_id == user_id)
        row = self.db.execute(stmt).scalars().first()
        if row is None:
            return 0
        row.converted = True
        row.clicked = True
        self.db.commit()
        return 1
