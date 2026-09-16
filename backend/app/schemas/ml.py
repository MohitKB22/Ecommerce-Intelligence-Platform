"""ML-facing schemas: sentiment, forecasting, pricing, segments, monitoring."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AspectSentiment(BaseModel):
    aspect: str
    label: str
    mentions: int
    positive: int
    negative: int
    neutral: int
    sentiment: float = Field(..., description="(positive - negative) / mentions, in [-1, 1]")


class ProductSentiment(BaseModel):
    product_id: int
    review_count: int
    distribution: dict[str, int] = Field(default_factory=dict)
    distribution_pct: dict[str, float] = Field(default_factory=dict)
    average_score: float = 0.0
    average_rating: float = 0.0
    aspects: list[AspectSentiment] = Field(default_factory=list)
    top_praise: list[AspectSentiment] = Field(default_factory=list)
    top_complaints: list[AspectSentiment] = Field(default_factory=list)
    keywords: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, examples=[
        "Battery life is excellent but the ear cushions are uncomfortable."
    ])


class SentimentResult(BaseModel):
    label: str
    score: float
    aspects: dict[str, str] = Field(default_factory=dict)
    positive_aspects: list[str] = Field(default_factory=list)
    negative_aspects: list[str] = Field(default_factory=list)
    model_version: str
    note: str | None = None


class ForecastPoint(BaseModel):
    day: int
    date: str | None = None
    predicted: float
    lower: float
    upper: float
    model: str


class ForecastResponse(BaseModel):
    product_id: int
    product_title: str
    horizon_days: int
    model_version: str
    model_type: str
    is_trained_product: bool
    points: list[ForecastPoint]
    horizons: dict[str, dict[str, float]]
    current_inventory: int
    stockout_risk: dict[str, Any]
    history: list[dict[str, Any]] = Field(default_factory=list)
    note: str | None = None


class FeatureContribution(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: str


class PricePredictionResponse(BaseModel):
    product_id: int
    product_title: str
    current_price: float
    effective_price: float
    predicted_price: float
    price_trend: str = Field(..., description="up | down | stable")
    delta_pct: float
    confidence: float
    lower_bound: float
    upper_bound: float
    expected_demand_90d: float
    explanation: str
    feature_contributions: list[FeatureContribution] = Field(default_factory=list)
    global_importance: list[dict[str, Any]] = Field(default_factory=list)
    model_version: str


class SegmentSummary(BaseModel):
    label: str
    name: str
    description: str = ""
    customer_count: int
    share: float
    avg_monetary: float
    avg_frequency: float
    avg_recency_days: float
    avg_order_value: float
    avg_rfm_score: float
    avg_confidence: float


class SegmentOverview(BaseModel):
    available: bool
    model_version: str | None = None
    total_customers: int = 0
    segments: list[SegmentSummary] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    cluster_profiles: dict[str, Any] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    note: str | None = None


class ModelRegistryOut(BaseModel):
    name: str
    version: str
    stage: str
    algorithm: str
    dataset_version: str
    training_rows: int
    trained_at: str | None
    training_duration_s: float
    metrics: dict[str, Any]
    params: dict[str, Any]
    feature_names: list[str]
    is_active: bool
    notes: str
    is_loaded: bool
    live_stats: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded | error")
    environment: str
    version: str
    checks: dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
