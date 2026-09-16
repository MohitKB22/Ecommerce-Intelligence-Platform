"""Commerce domain: orders, order items and reviews."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.mixins import PKMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.catalog import Product
    from app.models.user import User

ORDER_STATUSES = ("pending", "paid", "shipped", "delivered", "cancelled", "returned")


class Order(Base, PKMixin, TimestampMixin):
    __tablename__ = "orders"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    order_number: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="paid", index=True)
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    discount_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="web")
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(back_populates="order", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("total_amount >= 0", name="ck_order_total_non_negative"),
        CheckConstraint(
            "status IN ('pending','paid','shipped','delivered','cancelled','returned')", name="ck_order_status"
        ),
        Index("ix_orders_user_placed", "user_id", "placed_at"),
    )

    @property
    def is_revenue(self) -> bool:
        return self.status in ("paid", "shipped", "delivered")


class OrderItem(Base, PKMixin, TimestampMixin):
    __tablename__ = "order_items"

    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="RESTRICT"), nullable=False, index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    discount_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    line_total: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship(back_populates="order_items")

    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_order_item_quantity_positive"),
        CheckConstraint("unit_price >= 0", name="ck_order_item_price_non_negative"),
        Index("ix_order_items_product_order", "product_id", "order_id"),
    )


class Review(Base, PKMixin, TimestampMixin):
    __tablename__ = "reviews"

    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    verified_purchase: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    helpful_votes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # populated by the sentiment pipeline
    sentiment_label: Mapped[str | None] = mapped_column(String(10), index=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    sentiment_model_version: Mapped[str | None] = mapped_column(String(40))
    aspects: Mapped[dict | None] = mapped_column(JSON)

    product: Mapped[Product] = relationship(back_populates="reviews")
    user: Mapped[User] = relationship(back_populates="reviews")

    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_review_rating_range"),
        CheckConstraint(
            "sentiment_label IS NULL OR sentiment_label IN ('positive','neutral','negative')",
            name="ck_review_sentiment_label",
        ),
        Index("ix_reviews_product_rating", "product_id", "rating"),
        Index("ix_reviews_product_user", "product_id", "user_id", unique=True),
    )
