"""Personalization: build a user preference profile from behaviour.

The profile is the single source of personalization for the homepage, search
ranking and recommendations. It is derived from events, orders and reviews, and
cached because it is read on nearly every request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.logging_config import get_logger
from app.models import EVENT_WEIGHTS, Order, OrderItem, Product, Review, SearchEvent, User, UserEvent

logger = get_logger("personalization")

PROFILE_TTL = 300
HALF_LIFE_DAYS = 45.0


@dataclass
class UserProfile:
    user_id: int | None
    is_known: bool
    category_affinity: dict[int, float] = field(default_factory=dict)
    brand_affinity: dict[int, float] = field(default_factory=dict)
    recently_viewed: list[int] = field(default_factory=list)
    purchased: list[int] = field(default_factory=list)
    cart: list[int] = field(default_factory=list)
    wishlist: list[int] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    preferred_price_range: tuple[float, float] = (0.0, 0.0)
    avg_order_value: float = 0.0
    order_count: int = 0
    total_spend: float = 0.0
    session_count: int = 0
    review_count: int = 0
    last_active: str | None = None
    segment: str | None = None
    price_sensitivity: float = 0.5
    interaction_count: int = 0

    @property
    def is_cold_start(self) -> bool:
        return self.interaction_count < 3

    def preference_vector(self, dim: int = 16) -> list[float]:
        """Compact dense summary of the profile - exposed via the API for debugging."""
        vector = np.zeros(dim, dtype=np.float32)
        for category_id, weight in self.category_affinity.items():
            vector[int(category_id) % dim] += weight
        for brand_id, weight in self.brand_affinity.items():
            vector[(int(brand_id) * 7) % dim] += weight * 0.5
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return [round(float(v), 5) for v in vector]

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "is_known": self.is_known,
            "is_cold_start": self.is_cold_start,
            "interaction_count": self.interaction_count,
            "category_affinity": {str(k): round(v, 5) for k, v in
                                  sorted(self.category_affinity.items(), key=lambda kv: -kv[1])[:10]},
            "brand_affinity": {str(k): round(v, 5) for k, v in
                               sorted(self.brand_affinity.items(), key=lambda kv: -kv[1])[:10]},
            "recently_viewed": self.recently_viewed[:20],
            "purchased": self.purchased[:20],
            "cart": self.cart,
            "wishlist": self.wishlist,
            "search_terms": self.search_terms[:10],
            "preferred_price_range": [round(self.preferred_price_range[0], 2),
                                      round(self.preferred_price_range[1], 2)],
            "avg_order_value": round(self.avg_order_value, 2),
            "order_count": self.order_count,
            "total_spend": round(self.total_spend, 2),
            "session_count": self.session_count,
            "review_count": self.review_count,
            "segment": self.segment,
            "price_sensitivity": round(self.price_sensitivity, 4),
            "last_active": self.last_active,
            "preference_vector": self.preference_vector(),
        }


class PersonalizationService:
    def __init__(self, db: Session):
        self.db = db
        self.cache = get_cache()

    def invalidate(self, user_id: int) -> None:
        self.cache.invalidate(self.cache.key("profile", user_id))

    def get_profile(self, user_id: int | None, session_id: str | None = None) -> UserProfile:
        if user_id is None:
            return self._anonymous_profile(session_id)
        key = self.cache.key("profile", user_id)
        cached = self.cache.get_json(key)
        if cached is not None:
            return self._from_cache(cached)
        profile = self._build(user_id)
        self.cache.set_json(key, self._to_cache(profile), PROFILE_TTL)
        return profile

    # ---- serialisation ------------------------------------------------
    @staticmethod
    def _to_cache(profile: UserProfile) -> dict[str, Any]:
        return {
            "user_id": profile.user_id, "is_known": profile.is_known,
            "category_affinity": {str(k): v for k, v in profile.category_affinity.items()},
            "brand_affinity": {str(k): v for k, v in profile.brand_affinity.items()},
            "recently_viewed": profile.recently_viewed, "purchased": profile.purchased,
            "cart": profile.cart, "wishlist": profile.wishlist, "search_terms": profile.search_terms,
            "preferred_price_range": list(profile.preferred_price_range),
            "avg_order_value": profile.avg_order_value, "order_count": profile.order_count,
            "total_spend": profile.total_spend, "session_count": profile.session_count,
            "review_count": profile.review_count, "last_active": profile.last_active,
            "segment": profile.segment, "price_sensitivity": profile.price_sensitivity,
            "interaction_count": profile.interaction_count,
        }

    @staticmethod
    def _from_cache(data: dict[str, Any]) -> UserProfile:
        return UserProfile(
            user_id=data.get("user_id"), is_known=data.get("is_known", False),
            category_affinity={int(k): float(v) for k, v in (data.get("category_affinity") or {}).items()},
            brand_affinity={int(k): float(v) for k, v in (data.get("brand_affinity") or {}).items()},
            recently_viewed=data.get("recently_viewed", []), purchased=data.get("purchased", []),
            cart=data.get("cart", []), wishlist=data.get("wishlist", []),
            search_terms=data.get("search_terms", []),
            preferred_price_range=tuple(data.get("preferred_price_range", [0.0, 0.0])),  # type: ignore[arg-type]
            avg_order_value=data.get("avg_order_value", 0.0), order_count=data.get("order_count", 0),
            total_spend=data.get("total_spend", 0.0), session_count=data.get("session_count", 0),
            review_count=data.get("review_count", 0), last_active=data.get("last_active"),
            segment=data.get("segment"), price_sensitivity=data.get("price_sensitivity", 0.5),
            interaction_count=data.get("interaction_count", 0),
        )

    def _anonymous_profile(self, session_id: str | None) -> UserProfile:
        """Session-scoped profile for signed-out visitors."""
        profile = UserProfile(user_id=None, is_known=False)
        if not session_id:
            return profile
        rows = self.db.execute(
            select(UserEvent.event_type, UserEvent.product_id, Product.category_id, Product.brand_id,
                   Product.price, UserEvent.occurred_at)
            .join(Product, Product.id == UserEvent.product_id)
            .where(UserEvent.session_id == session_id)
            .order_by(UserEvent.occurred_at.desc())
            .limit(200)
        ).all()
        self._accumulate(profile, rows)
        return profile

    def _build(self, user_id: int) -> UserProfile:
        user = self.db.get(User, user_id)
        profile = UserProfile(user_id=user_id, is_known=user is not None)
        if user is None:
            return profile
        profile.price_sensitivity = float(user.price_sensitivity or 0.5)
        profile.last_active = user.last_active_at.isoformat() if user.last_active_at else None

        rows = self.db.execute(
            select(UserEvent.event_type, UserEvent.product_id, Product.category_id, Product.brand_id,
                   Product.price, UserEvent.occurred_at)
            .join(Product, Product.id == UserEvent.product_id)
            .where(UserEvent.user_id == user_id)
            .order_by(UserEvent.occurred_at.desc())
            .limit(600)
        ).all()
        self._accumulate(profile, rows)

        order_stats = self.db.execute(
            select(func.count(Order.id), func.coalesce(func.sum(Order.total_amount), 0.0),
                   func.coalesce(func.avg(Order.total_amount), 0.0))
            .where(Order.user_id == user_id, Order.status.in_(("paid", "shipped", "delivered")))
        ).one()
        profile.order_count = int(order_stats[0] or 0)
        profile.total_spend = float(order_stats[1] or 0.0)
        profile.avg_order_value = float(order_stats[2] or 0.0)

        profile.purchased = [
            int(pid) for (pid,) in self.db.execute(
                select(OrderItem.product_id).join(Order, Order.id == OrderItem.order_id)
                .where(Order.user_id == user_id).order_by(Order.placed_at.desc()).limit(50)
            ).all()
        ]
        profile.session_count = int(self.db.execute(
            select(func.count(func.distinct(UserEvent.session_id))).where(UserEvent.user_id == user_id)
        ).scalar_one() or 0)
        profile.review_count = int(self.db.execute(
            select(func.count(Review.id)).where(Review.user_id == user_id)
        ).scalar_one() or 0)
        profile.search_terms = [
            q for (q,) in self.db.execute(
                select(SearchEvent.normalized_query).where(SearchEvent.user_id == user_id)
                .order_by(SearchEvent.occurred_at.desc()).limit(20)
            ).all() if q
        ]
        profile.segment = self._latest_segment(user_id)
        return profile

    def _accumulate(self, profile: UserProfile, rows) -> None:
        """Time-decayed affinity accumulation over the event stream."""
        now = datetime.now(timezone.utc)
        category_scores: dict[int, float] = {}
        brand_scores: dict[int, float] = {}
        prices: list[float] = []
        seen_views: list[int] = []
        cart: list[int] = []
        wishlist: list[int] = []
        interactions = 0

        for event_type, product_id, category_id, brand_id, price, occurred_at in rows:
            weight = EVENT_WEIGHTS.get(event_type, 0.0)
            if weight == 0.0:
                continue
            interactions += 1
            if occurred_at is not None:
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
                age_days = max((now - occurred_at).total_seconds() / 86400.0, 0.0)
                decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
            else:
                decay = 0.5
            contribution = weight * decay
            if category_id is not None:
                category_scores[int(category_id)] = category_scores.get(int(category_id), 0.0) + contribution
            if brand_id is not None:
                brand_scores[int(brand_id)] = brand_scores.get(int(brand_id), 0.0) + contribution
            if weight > 0 and price is not None:
                prices.append(float(price))
            if event_type in ("product_view", "click") and product_id not in seen_views:
                seen_views.append(int(product_id))
            elif event_type == "add_to_cart" and product_id not in cart:
                cart.append(int(product_id))
            elif event_type == "remove_from_cart" and product_id in cart:
                cart.remove(int(product_id))
            elif event_type == "wishlist" and product_id not in wishlist:
                wishlist.append(int(product_id))

        profile.category_affinity = _normalise(category_scores)
        profile.brand_affinity = _normalise(brand_scores)
        profile.recently_viewed = seen_views[:30]
        profile.cart = cart[:30]
        profile.wishlist = wishlist[:30]
        profile.interaction_count = interactions
        if prices:
            profile.preferred_price_range = (
                float(np.percentile(prices, 10)), float(np.percentile(prices, 90))
            )

    def _latest_segment(self, user_id: int) -> str | None:
        from app.models import CustomerSegment

        row = self.db.execute(
            select(CustomerSegment.segment_label)
            .where(CustomerSegment.user_id == user_id)
            .order_by(CustomerSegment.created_at.desc())
            .limit(1)
        ).first()
        return row[0] if row else None


def _normalise(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    total = max(scores.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in scores.items()}
