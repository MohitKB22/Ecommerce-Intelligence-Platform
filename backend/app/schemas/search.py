"""Search schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.catalog import ProductSummary


class RankingSignalsOut(BaseModel):
    text: float
    semantic: float
    popularity: float
    rating: float
    conversion: float
    personal: float
    availability: float


class SearchHit(BaseModel):
    product: ProductSummary
    score: float
    signals: RankingSignalsOut
    explanation: str


class SearchFacets(BaseModel):
    categories: list[dict[str, Any]] = Field(default_factory=list)
    brands: list[dict[str, Any]] = Field(default_factory=list)
    price: dict[str, float] = Field(default_factory=dict)
    rating: dict[str, int] = Field(default_factory=dict)
    availability: dict[str, int] = Field(default_factory=dict)
    on_sale: int = 0


class SearchResults(BaseModel):
    query: str
    normalized_query: str
    corrected_query: str | None = Field(
        None, description="Set when typo correction changed the query"
    )
    strategy: str = Field(..., description="hybrid | semantic_fallback | browse | popularity_fallback")
    total: int
    took_ms: float
    weights: dict[str, float]
    hits: list[SearchHit]
    facets: SearchFacets


class SuggestionOut(BaseModel):
    text: str
    type: str = Field(..., description="product | query | term")
    product_id: int | None = None
    popularity: int | None = None
