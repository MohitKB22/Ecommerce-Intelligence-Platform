"""Behavioural telemetry: product events, search events, served recommendations."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.mixins import PKMixin

if TYPE_CHECKING:
    from app.models.user import User

EVENT_TYPES = (
    "product_view",
    "search",
    "click",
    "add_to_cart",
    "remove_from_cart",
    "wishlist",
    "purchase",
    "review",
)

# Implicit-feedback strength used by the collaborative recommender.
EVENT_WEIGHTS: dict[str, float] = {
    "product_view": 1.0,
    "click": 1.5,
    "wishlist": 3.0,
    "add_to_cart": 4.0,
    "remove_from_cart": -1.0,
    "purchase": 6.0,
    "review": 3.0,
    "search": 0.0,
}


class UserEvent(Base, PKMixin):
    __tablename__ = "user_events"

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="web")
    event_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    user: Mapped[User | None] = relationship(back_populates="events")

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('product_view','search','click','add_to_cart','remove_from_cart',"
            "'wishlist','purchase','review')",
            name="ck_user_event_type",
        ),
        Index("ix_user_events_user_time", "user_id", "occurred_at"),
        Index("ix_user_events_product_type", "product_id", "event_type"),
        Index("ix_user_events_type_time", "event_type", "occurred_at"),
    )

    @property
    def weight(self) -> float:
        return EVENT_WEIGHTS.get(self.event_type, 0.0)


class SearchEvent(Base, PKMixin):
    __tablename__ = "search_events"

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    normalized_query: Mapped[str] = mapped_column(String(255), nullable=False, index=True, default="")
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clicked_product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id", ondelete="SET NULL"))
    clicked_position: Mapped[int | None] = mapped_column(Integer)
    converted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    filters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (Index("ix_search_events_query_time", "normalized_query", "occurred_at"),)

    @property
    def abandoned(self) -> bool:
        return self.clicked_product_id is None


class RecommendationLog(Base, PKMixin):
    """Impression log for served recommendations - powers CTR/conversion metrics."""

    __tablename__ = "recommendations"

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    surface: Mapped[str] = mapped_column(String(40), nullable=False, default="homepage", index=True)
    strategy: Mapped[str] = mapped_column(String(40), nullable=False, default="hybrid", index=True)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, default="unknown")
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    explanation: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    clicked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    converted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    served_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (
        Index("ix_recommendations_user_surface", "user_id", "surface"),
        Index("ix_recommendations_strategy_time", "strategy", "served_at"),
    )
