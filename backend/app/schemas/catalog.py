"""Catalog schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, computed_field

from app.schemas.common import ORMModel


class BrandOut(ORMModel):
    id: int
    slug: str
    name: str
    reputation_score: float


class CategoryOut(ORMModel):
    id: int
    slug: str
    name: str
    description: str | None = None
    parent_id: int | None = None


class CategorySummary(ORMModel):
    id: int
    slug: str
    name: str
    product_count: int
    avg_rating: float


class ProductSummary(ORMModel):
    id: int
    sku: str
    title: str
    price: float
    discount_pct: float
    currency: str
    rating_avg: float
    rating_count: int
    inventory: int
    image_url: str
    category_id: int
    brand_id: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_price(self) -> float:
        return round(self.price * (1 - self.discount_pct / 100.0), 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def in_stock(self) -> bool:
        return self.inventory > 0


class ProductDetail(ProductSummary):
    description: str
    specifications: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    launched_at: datetime | None = None
    is_active: bool
    category: CategoryOut | None = None
    brand: BrandOut | None = None


class ReviewOut(ORMModel):
    id: int
    product_id: int
    user_id: int
    rating: int
    title: str
    body: str
    verified_purchase: bool
    helpful_votes: int
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    aspects: dict[str, str] | None = None
    created_at: datetime


class ReviewCreate(ORMModel):
    product_id: int = Field(..., ge=1)
    user_id: int = Field(..., ge=1)
    rating: int = Field(..., ge=1, le=5)
    title: str = Field("", max_length=200)
    body: str = Field(..., min_length=5, max_length=5000)
