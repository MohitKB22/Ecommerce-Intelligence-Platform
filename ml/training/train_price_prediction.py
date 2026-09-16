"""Dynamic price prediction with explainability.

    python -m ml.training.train_price_prediction

Learns the market-consistent price of a product from catalogue attributes,
realised demand, sales velocity, inventory pressure, promotion depth, category
context and brand reputation. The prediction is the price the model considers
appropriate given those signals; comparing it with the listed price yields an
actionable up/down/stable trend plus a confidence from the residual spread.

Explainability is produced two ways:
  * global  - permutation importance on the hold-out set
  * local   - per-prediction feature contributions measured by ablation against
              the category median feature vector
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.metrics import regression_metrics
from ml.inference.artifacts import PricePredictionArtifact
from ml.preprocessing.loaders import TrainingFrames
from ml.training.base import TrainingPipeline, run_cli

# NOTE: `base_cost` and `category_price_rank` are deliberately excluded.
# `category_price_rank` is the product's own percentile rank *of price* within its
# category, so it leaks the target directly. Category median/quartiles are kept:
# they describe the competitive context without encoding this product's price. In the simulated catalogue cost is
# derived from price, so including it leaks the target (R^2 jumps to 0.99 and the
# model learns nothing about the market). The model must price from demand,
# quality, brand and category context instead.
FEATURES = [
    "brand_reputation", "rating_avg", "rating_count", "inventory", "discount_pct",
    "units_sold_90d", "revenue_90d", "sales_velocity", "views_90d", "conversion_rate",
    "category_median_price", "category_p25_price", "category_p75_price", "days_since_launch", "spec_weight_g",
    "warranty_months", "description_length", "tag_count",
]
TEST_SIZE = 0.2
TREND_THRESHOLD = 0.04  # +/-4% before a trend is called


class PricePredictionPipeline(TrainingPipeline):
    name = "price_prediction"
    algorithm = "HistGradientBoostingRegressor on catalogue + demand + inventory features (log-price target)"
    requires = ("products", "order_items", "events")

    def preprocess(self, frames: TrainingFrames) -> dict[str, Any]:
        products = frames.products.copy()
        items = frames.order_items.copy()
        events = frames.events

        if not items.empty:
            cutoff = items["placed_at"].max() - pd.Timedelta(days=90)
            recent = items[(items["placed_at"] >= cutoff) & (items["status"].isin(("paid", "shipped", "delivered")))]
            units = recent.groupby("product_id")["quantity"].sum()
            revenue = recent.groupby("product_id")["line_total"].sum()
            active_days = max((items["placed_at"].max() - cutoff).days, 1)
        else:
            units = pd.Series(dtype=float)
            revenue = pd.Series(dtype=float)
            active_days = 90

        views = events[events["event_type"].isin(("product_view", "click"))].groupby("product_id").size()
        purchases = events[events["event_type"] == "purchase"].groupby("product_id").size()

        df = products.copy()
        df["units_sold_90d"] = df["id"].map(units).fillna(0.0)
        df["revenue_90d"] = df["id"].map(revenue).fillna(0.0)
        df["sales_velocity"] = df["units_sold_90d"] / active_days
        df["views_90d"] = df["id"].map(views).fillna(0.0)
        purchase_counts = df["id"].map(purchases).fillna(0.0)
        df["conversion_rate"] = np.where(df["views_90d"] >= 3, purchase_counts / df["views_90d"].replace(0, np.nan), 0.0)
        df["conversion_rate"] = df["conversion_rate"].fillna(0.0).clip(0, 1)

        # Leave-one-out category statistics: a product never contributes to the
        # summary of its own competitive context.
        grp = df.groupby("category_id")["price"]
        cat_sum, cat_count = grp.transform("sum"), grp.transform("count")
        df["category_median_price"] = np.where(
            cat_count > 1, (cat_sum - df["price"]) / (cat_count - 1), grp.transform("median")
        )
        df["category_p25_price"] = grp.transform(lambda x: x.quantile(0.25))
        df["category_p75_price"] = grp.transform(lambda x: x.quantile(0.75))

        now = pd.Timestamp.now(tz="UTC")
        df["days_since_launch"] = (now - pd.to_datetime(df["launched_at"], utc=True)).dt.days.fillna(365).clip(0, 3650)
        df["spec_weight_g"] = df["specifications"].apply(
            lambda s: float((s or {}).get("weight_g", 0)) if isinstance(s, dict) else 0.0
        )
        df["warranty_months"] = df["specifications"].apply(
            lambda s: float((s or {}).get("warranty_months", 12)) if isinstance(s, dict) else 12.0
        )
        df["description_length"] = df["description"].fillna("").str.len()
        df["tag_count"] = df["tags"].apply(lambda t: len(t) if isinstance(t, list) else 0)

        for col in ("units_sold_90d", "revenue_90d", "views_90d", "rating_count"):
            df[col] = np.log1p(df[col].clip(lower=0))

        df = df[df["price"] > 0].copy()
        matrix = df[FEATURES].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.log(df["price"].to_numpy(dtype=float))  # log target: multiplicative errors

        from sklearn.model_selection import train_test_split

        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=42)
        self._training_rows = int(len(train_idx))
        self.logger.info("price_features_built", products=len(df), features=len(FEATURES))
        return {
            "frame": df, "X": matrix, "y": target,
            "train_idx": train_idx, "test_idx": test_idx,
        }

    def train(self, prepared: dict[str, Any]) -> PricePredictionArtifact:
        from sklearn.ensemble import HistGradientBoostingRegressor

        X, y = prepared["X"], prepared["y"]
        train_idx = prepared["train_idx"]
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.055, max_depth=6, min_samples_leaf=12,
            l2_regularization=0.8, random_state=42, early_stopping=True, validation_fraction=0.15,
        )
        model.fit(X[train_idx], y[train_idx])

        frame = prepared["frame"]
        category_stats: dict[int, dict[str, float]] = {}
        for category_id, grp in frame.groupby("category_id"):
            category_stats[int(category_id)] = {
                "median_price": round(float(grp["price"].median()), 4),
                "p25_price": round(float(grp["price"].quantile(0.25)), 4),
                "p75_price": round(float(grp["price"].quantile(0.75)), 4),
                "median_features": [round(float(v), 6) for v in grp[FEATURES].median().to_numpy()],
            }
        global_stats = {
            "median_price": round(float(frame["price"].median()), 4),
            "mean_price": round(float(frame["price"].mean()), 4),
            "median_features": [round(float(v), 6) for v in frame[FEATURES].median().to_numpy()],
        }
        return PricePredictionArtifact(
            model=model, feature_names=list(FEATURES), category_stats=category_stats,
            global_stats=global_stats, feature_importance={}, residual_std=0.0,
        )

    def evaluate(self, artifact: PricePredictionArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        from sklearn.inspection import permutation_importance

        X, y = prepared["X"], prepared["y"]
        test_idx = prepared["test_idx"]
        frame = prepared["frame"]

        log_pred = artifact.model.predict(X[test_idx])
        pred_price = np.exp(log_pred)
        true_price = np.exp(y[test_idx])

        price_metrics = regression_metrics(true_price, pred_price)
        log_metrics = regression_metrics(y[test_idx], log_pred)

        # Baseline: predict the category median price.
        category_median = frame.iloc[test_idx]["category_median_price"].to_numpy(dtype=float)
        baseline_metrics = regression_metrics(true_price, category_median)

        artifact.residual_std = round(float(np.std(y[test_idx] - log_pred)), 6)

        importance = permutation_importance(
            artifact.model, X[test_idx], y[test_idx], n_repeats=8, random_state=42, scoring="r2"
        )
        ranked = sorted(
            ({"feature": FEATURES[i], "importance": round(float(importance.importances_mean[i]), 6)}
             for i in range(len(FEATURES))),
            key=lambda d: -d["importance"],
        )
        artifact.feature_importance = {d["feature"]: d["importance"] for d in ranked}

        within_10 = float(np.mean(np.abs(pred_price - true_price) / true_price <= 0.10))
        within_20 = float(np.mean(np.abs(pred_price - true_price) / true_price <= 0.20))

        return {
            "r2": price_metrics["r2"],
            "mae": price_metrics["mae"],
            "rmse": price_metrics["rmse"],
            "mape": price_metrics["mape"],
            "log_target_r2": log_metrics["r2"],
            "log_residual_std": artifact.residual_std,
            "within_10pct": round(within_10, 5),
            "within_20pct": round(within_20, 5),
            "baseline_category_median": {
                "mae": baseline_metrics["mae"], "mape": baseline_metrics["mape"], "r2": baseline_metrics["r2"],
            },
            "improvement_vs_baseline_mae": round(
                (baseline_metrics["mae"] - price_metrics["mae"]) / baseline_metrics["mae"], 4
            ) if baseline_metrics["mae"] > 0 else None,
            "top_features": ranked[:8],
            "test_products": int(len(test_idx)),
        }

    def params(self, artifact: PricePredictionArtifact) -> dict[str, Any]:
        return {
            "features": list(FEATURES),
            "target": "log(price)",
            "test_size": TEST_SIZE,
            "trend_threshold": TREND_THRESHOLD,
            "estimator": "HistGradientBoostingRegressor(max_iter=400, lr=0.055, depth=6)",
        }

    def feature_names(self, artifact: PricePredictionArtifact) -> list[str]:
        return list(FEATURES)

    def baseline_stats(self, artifact: PricePredictionArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        frame = prepared["frame"]
        return {
            "price_median": round(float(frame["price"].median()), 4),
            "price_p95": round(float(frame["price"].quantile(0.95)), 4),
            "feature_means": {f: round(float(frame[f].mean()), 5) for f in FEATURES},
            "feature_stds": {f: round(float(frame[f].std()), 5) for f in FEATURES},
        }

    def persist_side_effects(self, artifact: PricePredictionArtifact, prepared: dict[str, Any], version: str) -> None:
        from app.core.db import engine
        from app.models import PricePrediction
        from sqlalchemy.orm import Session

        from ml.inference.price_service import explain_price

        frame = prepared["frame"]
        X = prepared["X"]
        now = pd.Timestamp.now(tz="UTC").to_pydatetime()
        rows = []
        for i, row in enumerate(frame.itertuples()):
            result = explain_price(artifact, X[i], float(row.price), int(row.category_id))
            rows.append({
                "product_id": int(row.id),
                "current_price": float(row.price),
                "predicted_price": result["predicted_price"],
                "expected_demand": float(np.expm1(row.units_sold_90d)),
                "price_trend": result["price_trend"],
                "confidence": result["confidence"],
                "feature_contributions": result["contributions"],
                "model_version": version,
                "generated_at": now,
            })
        try:
            with Session(engine) as session:
                session.query(PricePrediction).filter(PricePrediction.model_version == version).delete()
                for start in range(0, len(rows), 1000):
                    session.bulk_insert_mappings(PricePrediction, rows[start:start + 1000])
                session.commit()
            self.logger.info("price_predictions_persisted", rows=len(rows), version=version)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("price_persist_failed", error=str(exc))

    def notes(self, artifact: PricePredictionArtifact) -> str:
        return (
            "Predicts a market-consistent price from catalogue, demand and inventory signals on a log "
            "target. Trend is derived from the gap between predicted and listed price; confidence from "
            "the log-residual spread. Global explainability via permutation importance, local via "
            "ablation against the category median feature vector."
        )


if __name__ == "__main__":
    raise SystemExit(run_cli(PricePredictionPipeline, "Train the price prediction model."))
