"""Price prediction serving with explainability."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from ml.inference.price_service import explain_price
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.errors import ModelNotAvailableError, NotFoundError
from app.core.logging_config import get_logger
from app.ml.model_store import get_model_store
from app.models import Order, OrderItem, Product, UserEvent

logger = get_logger("pricing")
PRICE_TTL = 900


class PricingService:
    def __init__(self, db: Session):
        self.db = db
        self.store = get_model_store()
        self.cache = get_cache()

    def _feature_vector(self, product: Product, feature_names: list[str]) -> np.ndarray:
        """Rebuild the training feature vector for a single product at serve time."""
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=90)

        units, revenue = self.db.execute(
            select(func.coalesce(func.sum(OrderItem.quantity), 0),
                   func.coalesce(func.sum(OrderItem.line_total), 0.0))
            .join(Order, Order.id == OrderItem.order_id)
            .where(OrderItem.product_id == product.id, Order.placed_at >= since,
                   Order.status.in_(("paid", "shipped", "delivered")))
        ).one()
        views = int(self.db.execute(
            select(func.count(UserEvent.id)).where(
                UserEvent.product_id == product.id,
                UserEvent.event_type.in_(("product_view", "click")))
        ).scalar_one() or 0)
        purchases = int(self.db.execute(
            select(func.count(UserEvent.id)).where(
                UserEvent.product_id == product.id, UserEvent.event_type == "purchase")
        ).scalar_one() or 0)

        category_prices = [
            float(p) for (p,) in self.db.execute(
                select(Product.price).where(Product.category_id == product.category_id,
                                            Product.id != product.id)
            ).all()
        ]
        median_price = float(np.median(category_prices)) if category_prices else float(product.price)
        p25 = float(np.quantile(category_prices, 0.25)) if category_prices else float(product.price)
        p75 = float(np.quantile(category_prices, 0.75)) if category_prices else float(product.price)

        specs = product.specifications or {}
        launched = product.launched_at
        if launched is not None and launched.tzinfo is None:
            launched = launched.replace(tzinfo=timezone.utc)
        days_since_launch = float((now - launched).days) if launched else 365.0

        values = {
            "brand_reputation": float(product.brand.reputation_score) if product.brand else 0.5,
            "rating_avg": float(product.rating_avg),
            "rating_count": float(np.log1p(product.rating_count)),
            "inventory": float(product.inventory),
            "discount_pct": float(product.discount_pct),
            "units_sold_90d": float(np.log1p(float(units or 0))),
            "revenue_90d": float(np.log1p(float(revenue or 0.0))),
            "sales_velocity": float(units or 0) / 90.0,
            "views_90d": float(np.log1p(views)),
            "conversion_rate": (purchases / views) if views >= 3 else 0.0,
            "category_median_price": median_price,
            "category_p25_price": p25,
            "category_p75_price": p75,
            "days_since_launch": max(0.0, min(days_since_launch, 3650.0)),
            "spec_weight_g": float(specs.get("weight_g", 0) or 0),
            "warranty_months": float(specs.get("warranty_months", 12) or 12),
            "description_length": float(len(product.description or "")),
            "tag_count": float(len(product.tags or [])),
        }
        return np.array([values.get(name, 0.0) for name in feature_names], dtype=float)

    def predict(self, product_id: int) -> dict[str, Any]:
        product = self.db.get(Product, product_id)
        if product is None:
            raise NotFoundError(f"Product {product_id} was not found.")

        loaded = self.store.try_get("price_prediction")
        if loaded is None:
            raise ModelNotAvailableError(
                "The price prediction model has not been trained yet. "
                "Run: python -m ml.training.train_price_prediction"
            )

        key = self.cache.key("price", product_id)
        cached = self.cache.get_json(key)
        if cached is not None:
            return cached

        artifact = loaded.artifact
        features = self._feature_vector(product, artifact.feature_names)
        with self.store.timed("price_prediction", entity_type="product", entity_id=product_id) as box:
            result = explain_price(artifact, features, float(product.price), int(product.category_id))
            box["prediction"] = result["predicted_price"]

        expected_demand = float(np.expm1(features[artifact.feature_names.index("units_sold_90d")])) \
            if "units_sold_90d" in artifact.feature_names else 0.0

        contributions = [
            {
                "feature": name,
                "label": name.replace("_", " ").title(),
                "contribution": value,
                "direction": "increases price" if value > 0 else "decreases price",
            }
            for name, value in result["contributions"].items()
        ]
        payload = {
            "product_id": product_id,
            "product_title": product.title,
            "current_price": round(float(product.price), 2),
            "effective_price": round(float(product.effective_price), 2),
            "predicted_price": result["predicted_price"],
            "price_trend": result["price_trend"],
            "delta_pct": result["delta_pct"],
            "confidence": result["confidence"],
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
            "expected_demand_90d": round(expected_demand, 2),
            "explanation": self._narrative(result, product),
            "feature_contributions": contributions,
            "global_importance": sorted(
                ({"feature": k, "importance": v} for k, v in (artifact.feature_importance or {}).items()),
                key=lambda d: -d["importance"],
            )[:8],
            "model_version": loaded.version,
        }
        self.cache.set_json(key, payload, PRICE_TTL)
        return payload

    @staticmethod
    def _narrative(result: dict[str, Any], product: Product) -> str:
        trend = result["price_trend"]
        delta = result["delta_pct"]
        top = list(result["contributions"].items())[:2]
        drivers = ", ".join(name.replace("_", " ") for name, _ in top) or "market context"
        if trend == "up":
            return (f"The model prices this {abs(delta):.1f}% above its current listing, driven mainly by "
                    f"{drivers}. There may be room to raise the price.")
        if trend == "down":
            return (f"The model prices this {abs(delta):.1f}% below its current listing, driven mainly by "
                    f"{drivers}. It may be priced above what the market supports.")
        return (f"The current price is consistent with what the model expects for this product "
                f"(within {abs(delta):.1f}%), based on {drivers}.")

    def opportunities(self, limit: int = 10, direction: str = "up") -> list[dict[str, Any]]:
        """Products whose listed price diverges most from the model's estimate."""
        from app.models import PricePrediction

        stmt = (
            select(PricePrediction, Product)
            .join(Product, Product.id == PricePrediction.product_id)
            .where(PricePrediction.price_trend == direction)
            .order_by(func.abs(PricePrediction.predicted_price - PricePrediction.current_price).desc())
            .limit(limit)
        )
        rows = self.db.execute(stmt).all()
        return [
            {
                "product_id": prediction.product_id,
                "title": product.title,
                "current_price": round(float(prediction.current_price), 2),
                "predicted_price": round(float(prediction.predicted_price), 2),
                "delta_pct": round(
                    (prediction.predicted_price - prediction.current_price) / prediction.current_price * 100, 2
                ) if prediction.current_price else 0.0,
                "confidence": round(float(prediction.confidence), 4),
                "trend": prediction.price_trend,
            }
            for prediction, product in rows
        ]
