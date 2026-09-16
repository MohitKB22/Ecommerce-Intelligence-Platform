"""Catalog domain: categories, brands, products."""
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
    from app.models.commerce import OrderItem, Review


class Category(Base, PKMixin, TimestampMixin):
    __tablename__ = "categories"

    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id", ondelete="SET NULL"), index=True)

    parent: Mapped[Category | None] = relationship(remote_side="Category.id", back_populates="children")
    children: Mapped[list[Category]] = relationship(back_populates="parent")
    products: Mapped[list[Product]] = relationship(back_populates="category")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Category {self.slug}>"


class Brand(Base, PKMixin, TimestampMixin):
    __tablename__ = "brands"

    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    reputation_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    products: Mapped[list[Product]] = relationship(back_populates="brand")

    __table_args__ = (CheckConstraint("reputation_score >= 0 AND reputation_score <= 1", name="ck_brand_reputation"),)


class Product(Base, PKMixin, TimestampMixin):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id", ondelete="RESTRICT"), nullable=False, index=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id", ondelete="RESTRICT"), nullable=False, index=True)

    price: Mapped[float] = mapped_column(Float, nullable=False)
    base_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    discount_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")

    rating_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, index=True)
    rating_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inventory: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    image_url: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    specifications: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    launched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    category: Mapped[Category] = relationship(back_populates="products")
    brand: Mapped[Brand] = relationship(back_populates="products")
    reviews: Mapped[list[Review]] = relationship(back_populates="product", cascade="all, delete-orphan")
    order_items: Mapped[list[OrderItem]] = relationship(back_populates="product")

    __table_args__ = (
        CheckConstraint("price >= 0", name="ck_product_price_non_negative"),
        CheckConstraint("discount_pct >= 0 AND discount_pct < 100", name="ck_product_discount_range"),
        CheckConstraint("inventory >= 0", name="ck_product_inventory_non_negative"),
        CheckConstraint("rating_avg >= 0 AND rating_avg <= 5", name="ck_product_rating_range"),
        Index("ix_products_category_active", "category_id", "is_active"),
        Index("ix_products_price_rating", "price", "rating_avg"),
    )

    @property
    def effective_price(self) -> float:
        return round(self.price * (1 - self.discount_pct / 100.0), 2)

    @property
    def in_stock(self) -> bool:
        return self.inventory > 0

    @property
    def searchable_text(self) -> str:
        tags = " ".join(self.tags or [])
        specs = " ".join(f"{k} {v}" for k, v in (self.specifications or {}).items())
        return f"{self.title} {self.description} {tags} {specs}".strip()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Product {self.sku} {self.title[:24]!r}>"
