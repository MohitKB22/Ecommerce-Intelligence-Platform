"""Recommendation schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.catalog import ProductSummary


class RecommendationItem(BaseModel):
    product: ProductSummary
    score: float
    rank: int
    components: dict[str, float] = Field(
        default_factory=dict, description="Per-signal contribution to the final score"
    )
    explanation: str = Field(..., description="Human-readable reason this item was recommended")


class RecommendationResponse(BaseModel):
    items: list[RecommendationItem]
    strategy: str = Field(..., description="hybrid | cold_start | trending_events | popularity_fallback | ...")
    model_version: str
    count: int


class HomepageSection(BaseModel):
    key: str
    title: str
    subtitle: str = ""
    strategy: str
    items: list[RecommendationItem]


class HomepageResponse(BaseModel):
    user_id: int | None
    is_personalized: bool
    segment: str | None = None
    sections: list[HomepageSection]
    profile_summary: dict[str, Any] = Field(default_factory=dict)
