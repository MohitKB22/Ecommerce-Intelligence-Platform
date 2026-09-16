"""Demand forecasting (7 / 14 / 30 day horizons).

    python -m ml.training.train_forecasting

Two models are trained and compared on the same temporal hold-out:
  * baseline  - seasonal naive (day-of-week profile scaled by recent level)
  * advanced  - gradient-boosted regressor over lag, rolling, calendar,
                price and inventory features

The advanced model is only promoted if it actually beats the baseline on MAE;
otherwise the baseline is kept. Prediction intervals come from the empirical
residual quantiles of the winning model on the validation window.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.metrics import forecast_metrics
from ml.inference.artifacts import ForecastArtifact
from ml.preprocessing.loaders import TrainingFrames
from ml.training.base import TrainingPipeline, run_cli

HORIZONS = (7, 14, 30)          # reported in days
LAGS = (1, 2, 3, 4, 8, 12)      # in WEEKS
ROLLING_WINDOWS = (4, 8, 12)    # in WEEKS
MIN_HISTORY_WEEKS = 20
MIN_TOTAL_UNITS = 25
HOLDOUT_WEEKS = 8


class ForecastingPipeline(TrainingPipeline):
    name = "forecasting"
    algorithm = ("Seasonal-naive baseline + HistGradientBoostingRegressor residual correction "
                 "on weekly lag/rolling/calendar features")
    requires = ("products", "daily_demand")

    # ---- feature engineering -------------------------------------------
    def preprocess(self, frames: TrainingFrames) -> dict[str, Any]:
        """Build a WEEKLY demand panel.

        Daily per-SKU demand in a long-tail catalogue is dominated by zeros
        (the median product here sells well under one unit a day), which makes
        daily point forecasts both unstable and uninformative. Aggregating to
        ISO weeks is the standard retail practice: it preserves the seasonality
        that matters for replenishment while giving the model a target with real
        variance. The 7/14/30-day horizons the API exposes are derived from the
        weekly forecast.
        """
        demand = frames.daily_demand.copy()
        if demand.empty:
            raise ValueError("no demand history available")
        demand["date"] = pd.to_datetime(demand["date"], utc=True).dt.floor("D")
        # Monday-anchored week start. Computed by subtracting the weekday offset
        # rather than via to_period(), whose start_time anchor does not line up
        # with a date_range(freq="W-MON") index and silently produces an all-NaN
        # reindex (every product forecasting to zero).
        demand["week"] = (demand["date"] - pd.to_timedelta(demand["date"].dt.dayofweek, unit="D")).dt.floor("D")

        weekly = demand.groupby(["product_id", "week"], as_index=False).agg(
            units=("units", "sum"), avg_price=("avg_price", "mean")
        )

        products = frames.products.set_index("id")
        totals = weekly.groupby("product_id")["units"].sum()
        spans = weekly.groupby("product_id")["week"].nunique()
        eligible = totals[totals >= MIN_TOTAL_UNITS].index.intersection(
            spans[spans >= MIN_HISTORY_WEEKS].index
        )
        if len(eligible) == 0:
            raise ValueError("no product has enough demand history to train a forecaster")

        start, end = weekly["week"].min(), weekly["week"].max()
        full_index = pd.date_range(start, end, freq="7D")  # anchored on the observed first week

        panels = []
        for product_id in eligible:
            series = (
                weekly[weekly["product_id"] == product_id]
                .set_index("week")[["units", "avg_price"]]
                .reindex(full_index)
            )
            series["units"] = series["units"].fillna(0.0)     # a week with no sale is zero demand
            series["avg_price"] = series["avg_price"].ffill().bfill()
            series["product_id"] = product_id
            series["week"] = series.index
            panels.append(series.reset_index(drop=True))

        panel = pd.concat(panels, ignore_index=True)
        for col in ("category_id", "brand_id", "price", "inventory", "discount_pct", "rating_avg"):
            panel[col] = panel["product_id"].map(products[col])
        panel["avg_price"] = panel["avg_price"].fillna(panel["price"])

        panel = panel.sort_values(["product_id", "week"]).reset_index(drop=True)
        grouped = panel.groupby("product_id")["units"]
        for lag in LAGS:
            panel[f"lag_{lag}"] = grouped.shift(lag)
        shifted = grouped.shift(1)
        for window in ROLLING_WINDOWS:
            panel[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=2).mean().reset_index(drop=True)
            panel[f"roll_std_{window}"] = shifted.rolling(window, min_periods=2).std().reset_index(drop=True)
        panel["roll_max_12"] = shifted.rolling(12, min_periods=2).max().reset_index(drop=True)
        panel["trend_4_12"] = panel["roll_mean_4"] - panel["roll_mean_12"]

        panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
        panel["month"] = panel["week"].dt.month
        panel["quarter"] = panel["week"].dt.quarter
        panel["week_index"] = ((panel["week"] - start).dt.days // 7).astype(int)
        if panel["units"].sum() <= 0:
            raise ValueError("weekly demand panel is entirely zero - aggregation is misaligned")
        panel["price_ratio"] = (panel["avg_price"] / panel["price"].replace(0, np.nan)).fillna(1.0)
        panel["on_promotion"] = (panel["price_ratio"] < 0.95).astype(int)
        panel["woy_sin"] = np.sin(2 * np.pi * panel["week_of_year"] / 52)
        panel["woy_cos"] = np.cos(2 * np.pi * panel["week_of_year"] / 52)
        panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
        panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)

        feature_names = (
            [f"lag_{lag}" for lag in LAGS]
            + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
            + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
            + ["roll_max_12", "trend_4_12", "week_of_year", "month", "quarter", "week_index",
               "price_ratio", "on_promotion", "woy_sin", "woy_cos", "month_sin", "month_cos",
               "category_id", "brand_id", "price", "inventory", "discount_pct", "rating_avg"]
        )
        panel = panel.dropna(subset=[f"lag_{max(LAGS)}"]).reset_index(drop=True)

        split_week = end - pd.Timedelta(weeks=HOLDOUT_WEEKS)
        train = panel[panel["week"] <= split_week]
        test = panel[panel["week"] > split_week]
        if train.empty or test.empty:
            raise ValueError("temporal split produced an empty train or test set")

        self._training_rows = int(len(train))
        self.logger.info("forecast_panel_built", grain="weekly", products=len(eligible),
                         train_rows=len(train), test_rows=len(test), split=str(split_week.date()))
        return {
            "panel": panel, "train": train, "test": test, "feature_names": feature_names,
            "products": list(eligible), "start": start, "end": end, "split_date": split_week,
        }

    # ---- training -------------------------------------------------------
    def train(self, prepared: dict[str, Any]) -> ForecastArtifact:
        from sklearn.ensemble import HistGradientBoostingRegressor

        train, features = prepared["train"], prepared["feature_names"]
        panel = prepared["panel"]
        overall = float(panel["units"].mean()) or 1.0
        month_means = panel.groupby("month")["units"].mean()
        global_seasonal = [float(month_means.get(m, overall) / overall) for m in range(1, 13)]

        baseline_profile: dict[int, dict[str, float]] = {}
        history_tail: dict[int, list[float]] = {}
        for product_id, grp in panel.groupby("product_id"):
            recent = grp.tail(4)
            product_mean = float(grp["units"].mean()) or 1.0
            by_month = grp.groupby("month")["units"].mean()
            baseline_profile[int(product_id)] = {
                "recent_mean": float(recent["units"].mean()),
                "overall_mean": product_mean,
                **{f"month_{m}": float(by_month.get(m, product_mean) / product_mean) for m in range(1, 13)},
            }
            history_tail[int(product_id)] = [float(v) for v in grp["units"].tail(12)]

        artifact = ForecastArtifact(
            model=None,
            baseline_profile=baseline_profile,
            feature_names=list(features),
            global_dow_factor=global_seasonal,
            residual_std=0.0,
            trained_products=[int(p) for p in prepared["products"]],
            history_tail=history_tail,
            last_date=str(prepared["end"].date()),
        )

        # The gradient booster learns the *residual* against the seasonal-naive
        # level rather than raw demand. Plain regression on raw units loses to
        # the naive baseline here (long-tail weekly demand is dominated by its
        # own recent level); correcting the baseline instead of replacing it
        # gives the model an easier, better-conditioned target.
        X = train[features].to_numpy(dtype=float)
        y = train["units"].to_numpy(dtype=float)
        baseline_train = self._baseline_predict(artifact, train)
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=6, min_samples_leaf=15,
            l2_regularization=1.0, random_state=42, early_stopping=True, validation_fraction=0.15,
        )
        model.fit(X, y - baseline_train)
        artifact.model = model
        return artifact

    # ---- evaluation -----------------------------------------------------
    @staticmethod
    def _baseline_predict(artifact: ForecastArtifact, rows: pd.DataFrame) -> np.ndarray:
        """Seasonal naive: recent 4-week level x month-of-year factor."""
        out = np.zeros(len(rows), dtype=float)
        for i, row in enumerate(rows.itertuples()):
            profile = artifact.baseline_profile.get(int(row.product_id))
            if profile is None:
                continue
            factor = profile.get(f"month_{int(row.month)}", 1.0)
            out[i] = max(profile["recent_mean"] * factor, 0.0)
        return out

    def evaluate(self, artifact: ForecastArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        test, features = prepared["test"], prepared["feature_names"]
        y_true = test["units"].to_numpy(dtype=float)

        baseline = self._baseline_predict(artifact, test)
        correction = artifact.model.predict(test[features].to_numpy(dtype=float))
        advanced = np.clip(baseline + correction, 0, None)

        advanced_metrics = forecast_metrics(y_true, advanced)
        baseline_metrics = forecast_metrics(y_true, baseline)
        use_advanced = advanced_metrics["mae"] <= baseline_metrics["mae"]
        chosen = advanced if use_advanced else baseline

        artifact.residual_std = round(float(np.std(y_true - chosen)), 5)
        self._use_advanced = use_advanced

        # Horizon accuracy: cumulative units over the first N days of the hold-out,
        # per product. This is the number that actually drives replenishment.
        test = test.copy()
        test["_pred"] = chosen
        first_week = test["week"].min()
        horizon_metrics: dict[str, Any] = {}
        for horizon in HORIZONS:
            n_weeks = max(1, int(round(horizon / 7)))
            window = test[test["week"] < first_week + pd.Timedelta(weeks=n_weeks)]
            if window.empty:
                continue
            actual = window.groupby("product_id")["units"].sum()
            predicted = window.groupby("product_id")["_pred"].sum()
            scale = horizon / (7 * n_weeks)  # e.g. 30d from 4 weeks
            common = actual.index.intersection(predicted.index)
            horizon_metrics[f"horizon_{horizon}d"] = forecast_metrics(
                (actual.loc[common] * scale).to_numpy(), (predicted.loc[common] * scale).to_numpy()
            )

        nonzero = y_true > 0
        return {
            "grain": "weekly",
            "mae": advanced_metrics["mae"] if use_advanced else baseline_metrics["mae"],
            "rmse": advanced_metrics["rmse"] if use_advanced else baseline_metrics["rmse"],
            "mape": advanced_metrics["mape"] if use_advanced else baseline_metrics["mape"],
            "smape": advanced_metrics["smape"] if use_advanced else baseline_metrics["smape"],
            "mape_nonzero_weeks": forecast_metrics(y_true[nonzero], chosen[nonzero])["mape"]
            if nonzero.any() else None,
            "selected_model": "naive_plus_gbm_residual" if use_advanced else "seasonal_naive",
            "advanced_model": advanced_metrics,
            "baseline_seasonal_naive": baseline_metrics,
            "improvement_vs_baseline_mae": round(
                (baseline_metrics["mae"] - advanced_metrics["mae"]) / baseline_metrics["mae"], 4
            ) if baseline_metrics["mae"] > 0 else None,
            "residual_std": artifact.residual_std,
            "products_forecastable": len(artifact.trained_products),
            "holdout_weeks": HOLDOUT_WEEKS,
            "mean_weekly_units": round(float(prepared["panel"]["units"].mean()), 4),
            **horizon_metrics,
        }

    def params(self, artifact: ForecastArtifact) -> dict[str, Any]:
        return {
            "grain": "weekly",
            "horizons_days": list(HORIZONS),
            "lags_weeks": list(LAGS),
            "rolling_windows_weeks": list(ROLLING_WINDOWS),
            "min_history_weeks": MIN_HISTORY_WEEKS,
            "min_total_units": MIN_TOTAL_UNITS,
            "holdout_weeks": HOLDOUT_WEEKS,
            "estimator": "HistGradientBoostingRegressor(max_iter=400, lr=0.05, depth=6) on naive residual",
            "target": "units - seasonal_naive(product, month)",
            "selected_model": "naive_plus_gbm_residual" if getattr(self, "_use_advanced", True) else "seasonal_naive",
        }

    def feature_names(self, artifact: ForecastArtifact) -> list[str]:
        return list(artifact.feature_names)

    def baseline_stats(self, artifact: ForecastArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        panel = prepared["panel"]
        return {
            "units_mean": round(float(panel["units"].mean()), 5),
            "units_std": round(float(panel["units"].std()), 5),
            "units_p95": round(float(panel["units"].quantile(0.95)), 5),
            "zero_demand_ratio": round(float((panel["units"] == 0).mean()), 5),
            "weeks_observed": int(panel["week"].nunique()),
        }

    def persist_side_effects(self, artifact: ForecastArtifact, prepared: dict[str, Any], version: str) -> None:
        """Generate and store forward forecasts for every trained product."""
        from app.core.db import engine
        from app.models import Forecast
        from sqlalchemy.orm import Session

        from ml.inference.forecast_service import forward_forecast

        end = pd.Timestamp(prepared["end"]).tz_localize("UTC") if pd.Timestamp(prepared["end"]).tzinfo is None \
            else pd.Timestamp(prepared["end"])
        rows = []
        for product_id in artifact.trained_products[:400]:  # bounded write volume
            points = forward_forecast(artifact, int(product_id), max(HORIZONS))
            for point in points:
                rows.append({
                    "product_id": int(product_id),
                    "horizon_days": int(point["day"]),
                    "target_date": (end + timedelta(days=int(point["day"]))).to_pydatetime(),
                    "predicted_demand": float(point["predicted"]),
                    "lower_bound": float(point["lower"]),
                    "upper_bound": float(point["upper"]),
                    "model_version": version,
                    "generated_at": end.to_pydatetime(),
                })
        try:
            with Session(engine) as session:
                session.query(Forecast).filter(Forecast.model_version == version).delete()
                for start in range(0, len(rows), 1000):
                    session.bulk_insert_mappings(Forecast, rows[start:start + 1000])
                session.commit()
            self.logger.info("forecasts_persisted", rows=len(rows), version=version)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("forecast_persist_failed", error=str(exc))

    def notes(self, artifact: ForecastArtifact) -> str:
        selected = "naive + GBM residual" if getattr(self, "_use_advanced", True) else "seasonal naive"
        return (
            f"Daily demand forecaster; {selected} selected on hold-out MAE. Prediction intervals use "
            "the empirical residual spread. Products with insufficient history fall back to the "
            "seasonal-naive profile at inference time."
        )


if __name__ == "__main__":
    raise SystemExit(run_cli(ForecastingPipeline, "Train the demand forecasting model."))
