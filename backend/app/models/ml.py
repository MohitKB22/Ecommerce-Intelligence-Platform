"""ML operational tables: feature store, model registry, predictions, forecasts."""
from __future__ import annotations

from datetime import datetime

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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.mixins import PKMixin, TimestampMixin


class ProductFeature(Base, PKMixin):
    """Versioned, timestamped product feature values (feature store backing table)."""

    __tablename__ = "product_features"

    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    feature_version: Mapped[str] = mapped_column(String(40), nullable=False, default="v1", index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("product_id", "feature_name", "feature_version", name="uq_product_feature"),
        Index("ix_product_features_name_version", "feature_name", "feature_version"),
    )


class UserFeature(Base, PKMixin):
    __tablename__ = "user_features"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    feature_version: Mapped[str] = mapped_column(String(40), nullable=False, default="v1", index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("user_id", "feature_name", "feature_version", name="uq_user_feature"),
        Index("ix_user_features_name_version", "feature_name", "feature_version"),
    )


class ModelRegistryEntry(Base, PKMixin, TimestampMixin):
    __tablename__ = "model_registry"

    name: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    stage: Mapped[str] = mapped_column(String(20), nullable=False, default="production", index=True)
    algorithm: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    dataset_version: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    training_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_duration_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    feature_names: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    baseline_stats: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
        CheckConstraint("stage IN ('development','staging','production','archived')", name="ck_model_stage"),
        Index("ix_model_registry_name_active", "name", "is_active"),
    )


class ModelPrediction(Base, PKMixin):
    """Sampled inference log used for latency, error-rate and drift monitoring."""

    __tablename__ = "model_predictions"

    model_name: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False, default="product")
    entity_id: Mapped[int | None] = mapped_column(Integer, index=True)
    prediction: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    actual: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    error_code: Mapped[str | None] = mapped_column(String(60))
    features: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (Index("ix_model_predictions_name_time", "model_name", "predicted_at"),)


class Forecast(Base, PKMixin):
    __tablename__ = "forecasts"

    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    target_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    predicted_demand: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lower_bound: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    upper_bound: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    actual_demand: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, default="v1")
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("product_id", "target_date", "model_version", name="uq_forecast_product_date"),
        CheckConstraint("horizon_days > 0", name="ck_forecast_horizon_positive"),
        Index("ix_forecasts_product_horizon", "product_id", "horizon_days"),
    )


class PricePrediction(Base, PKMixin):
    __tablename__ = "price_predictions"

    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    expected_demand: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price_trend: Mapped[str] = mapped_column(String(10), nullable=False, default="stable", index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    feature_contributions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, default="v1")
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    __table_args__ = (
        CheckConstraint("price_trend IN ('up','down','stable')", name="ck_price_trend"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_price_confidence"),
        Index("ix_price_predictions_product_time", "product_id", "generated_at"),
    )
