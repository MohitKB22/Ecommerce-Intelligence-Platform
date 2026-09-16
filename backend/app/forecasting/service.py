"""Demand forecasting serving layer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ml.inference.forecast_service import forward_forecast, horizon_totals
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.errors import ModelNotAvailableError, NotFoundError
from app.core.logging_config import get_logger
from app.ml.model_store import get_model_store
from app.models import Order, OrderItem, Product

logger = get_logger("forecasting")
FORECAST_TTL = 900
HORIZONS = (7, 14, 30)


class ForecastingService:
    def __init__(self, db: Session):
        self.db = db
        self.store = get_model_store()
        self.cache = get_cache()

    def forecast(self, product_id: int, horizon: int = 30, include_history: bool = True) -> dict[str, Any]:
        product = self.db.get(Product, product_id)
        if product is None:
            raise NotFoundError(f"Product {product_id} was not found.")

        loaded = self.store.try_get("forecasting")
        if loaded is None:
            raise ModelNotAvailableError(
                "The demand forecasting model has not been trained yet. "
                "Run: python -m ml.training.train_forecasting"
            )

        horizon = max(1, min(int(horizon), 120))
        key = self.cache.key("forecast", product_id, horizon)
        cached = self.cache.get_json(key)
        if cached is not None:
            if include_history:
                cached["history"] = self.history(product_id)
            return cached

        artifact = loaded.artifact
        meta = {
            "category_id": float(product.category_id), "brand_id": float(product.brand_id),
            "price": float(product.price), "inventory": float(product.inventory),
            "discount_pct": float(product.discount_pct), "rating_avg": float(product.rating_avg),
            "price_ratio": 1.0 - float(product.discount_pct) / 100.0,
            "on_promotion": 1.0 if product.discount_pct > 0 else 0.0,
        }
        with self.store.timed("forecasting", entity_type="product", entity_id=product_id) as box:
            points = forward_forecast(artifact, product_id, horizon, meta)
            box["prediction"] = float(points[0]["predicted"]) if points else 0.0

        totals = horizon_totals(points, HORIZONS)
        is_trained = product_id in set(artifact.trained_products)
        payload: dict[str, Any] = {
            "product_id": product_id,
            "product_title": product.title,
            "horizon_days": horizon,
            "model_version": loaded.version,
            "model_type": points[0]["model"] if points else "unknown",
            "is_trained_product": is_trained,
            "points": points,
            "horizons": totals,
            "current_inventory": int(product.inventory),
            "stockout_risk": self._stockout_risk(product, totals),
            "note": None if is_trained else (
                "This product has insufficient sales history for a learned forecast; "
                "a seasonal-naive estimate is shown."
            ),
        }
        self.cache.set_json(key, payload, FORECAST_TTL)
        if include_history:
            payload["history"] = self.history(product_id)
        return payload

    @staticmethod
    def _stockout_risk(product: Product, totals: dict[str, Any]) -> dict[str, Any]:
        """Compare projected demand against inventory on hand."""
        risk: dict[str, Any] = {}
        for label, values in totals.items():
            expected = values["predicted_units"]
            upper = values["upper_bound"]
            inventory = float(product.inventory)
            risk[label] = {
                "expected_demand": expected,
                "inventory": inventory,
                "will_stock_out": expected > inventory,
                "worst_case_stock_out": upper > inventory,
                "cover_ratio": round(inventory / expected, 3) if expected > 0 else None,
            }
        return risk

    def history(self, product_id: int, days: int = 120) -> list[dict[str, Any]]:
        """Actual weekly units sold - plotted against the forecast in the UI."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(func.date(Order.placed_at), func.sum(OrderItem.quantity))
            .join(Order, Order.id == OrderItem.order_id)
            .where(OrderItem.product_id == product_id, Order.placed_at >= since,
                   Order.status.in_(("paid", "shipped", "delivered")))
            .group_by(func.date(Order.placed_at))
            .order_by(func.date(Order.placed_at))
        ).all()
        return [{"date": str(day), "units": int(units or 0)} for day, units in rows]

    def accuracy(self) -> dict[str, Any]:
        """Report the hold-out metrics recorded at training time."""
        loaded = self.store.try_get("forecasting")
        if loaded is None:
            return {"available": False}
        metrics = loaded.card.metrics or {}
        return {
            "available": True,
            "model_version": loaded.version,
            "trained_at": loaded.card.trained_at,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "smape": metrics.get("smape"),
            "selected_model": metrics.get("selected_model"),
            "baseline": metrics.get("baseline_seasonal_naive"),
            "improvement_vs_baseline": metrics.get("improvement_vs_baseline_mae"),
            "horizons": {k: v for k, v in metrics.items() if k.startswith("horizon_")},
            "products_forecastable": metrics.get("products_forecastable"),
        }

    def top_demand(self, limit: int = 10) -> list[dict[str, Any]]:
        """Products with the highest projected 30-day demand."""
        loaded = self.store.try_get("forecasting")
        if loaded is None:
            return []
        artifact = loaded.artifact
        results = []
        for product_id in artifact.trained_products[:150]:
            points = forward_forecast(artifact, int(product_id), 30)
            total = sum(p["predicted"] for p in points)
            results.append({"product_id": int(product_id), "predicted_30d": round(total, 3)})
        results.sort(key=lambda r: -r["predicted_30d"])
        top = results[:limit]
        ids = [r["product_id"] for r in top]
        products = {p.id: p for p in self.db.execute(select(Product).where(Product.id.in_(ids))).scalars().all()}
        for row in top:
            product = products.get(row["product_id"])
            row["title"] = product.title if product else "Unknown"
            row["inventory"] = int(product.inventory) if product else 0
        return top
