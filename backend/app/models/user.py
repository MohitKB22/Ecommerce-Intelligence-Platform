"""User domain: accounts and ML-derived customer segments."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.mixins import PKMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.commerce import Order, Review
    from app.models.events import UserEvent


class User(Base, PKMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    country: Mapped[str] = mapped_column(String(2), nullable=False, default="US")
    signup_source: Mapped[str] = mapped_column(String(40), nullable=False, default="organic")
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    preferred_categories: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    preferred_brands: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    price_sensitivity: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    orders: Mapped[list[Order]] = relationship(back_populates="user", cascade="all, delete-orphan")
    reviews: Mapped[list[Review]] = relationship(back_populates="user", cascade="all, delete-orphan")
    events: Mapped[list[UserEvent]] = relationship(back_populates="user", cascade="all, delete-orphan")
    segment_assignments: Mapped[list[CustomerSegment]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("price_sensitivity >= 0 AND price_sensitivity <= 1", name="ck_user_price_sensitivity"),
        CheckConstraint("role IN ('user','admin','analyst')", name="ck_user_role"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.email}>"


class CustomerSegment(Base, PKMixin, TimestampMixin):
    """One row per user per segmentation model run."""

    __tablename__ = "customer_segments"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    segment_label: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    recency_days: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frequency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    monetary: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_order_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    product_diversity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    session_frequency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rfm_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    user: Mapped[User] = relationship(back_populates="segment_assignments")

    __table_args__ = (
        Index("ix_segment_user_version", "user_id", "model_version", unique=True),
        Index("ix_segment_label_version", "segment_label", "model_version"),
    )
