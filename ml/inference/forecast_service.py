"""Recursive multi-step forecasting on top of a trained ForecastArtifact.

The model is trained on a weekly grain (see `ml/training/train_forecasting.py`
for why). This module rolls it forward week by week, feeding each prediction
back in as the next step's lag feature, then expands the weekly path into the
daily series the API exposes.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from ml.inference.artifacts import ForecastArtifact

Z_95 = 1.96
WEEK_LAGS = (1, 2, 3, 4, 8, 12)
WEEK_ROLLING = (4, 8, 12)


def _parse_last_date(artifact: ForecastArtifact) -> datetime:
    try:
        return datetime.fromisoformat(str(artifact.last_date)).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def weekly_forecast(artifact: ForecastArtifact, product_id: int, n_weeks: int,
                    product_meta: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Roll the weekly model forward `n_weeks` steps."""
    product_id = int(product_id)
    n_weeks = max(1, int(n_weeks))
    history = list(artifact.history_tail.get(product_id, []))
    profile = artifact.baseline_profile.get(product_id)
    last_date = _parse_last_date(artifact)

    # Unknown product, no history, or no model -> seasonal-naive fallback.
    if not history or profile is None or artifact.model is None:
        level = profile.get("recent_mean", 0.0) if profile else 0.0
        spread = max(artifact.residual_std, 0.5)
        out = []
        for step in range(1, n_weeks + 1):
            target = last_date + timedelta(weeks=step)
            factor = profile.get(f"month_{target.month}", 1.0) if profile else 1.0
            value = max(level * factor, 0.0)
            out.append({
                "week": step,
                "date": target.date().isoformat(),
                "predicted": round(float(value), 4),
                "lower": round(max(value - Z_95 * spread, 0.0), 4),
                "upper": round(float(value + Z_95 * spread), 4),
                "model": "seasonal_naive",
            })
        return out

    meta = product_meta or {}
    feature_index = {name: i for i, name in enumerate(artifact.feature_names)}
    working = list(history)
    points: list[dict[str, Any]] = []

    for step in range(1, n_weeks + 1):
        target = last_date + timedelta(weeks=step)
        row = np.zeros(len(artifact.feature_names), dtype=float)

        def put(name: str, value: float, _row: np.ndarray = row) -> None:
            """`_row` is bound per iteration so the closure never captures a stale array."""
            idx = feature_index.get(name)
            if idx is not None:
                _row[idx] = float(value)

        series = np.asarray(working, dtype=float)
        mean_all = float(series.mean()) if series.size else 0.0
        for lag in WEEK_LAGS:
            put(f"lag_{lag}", series[-lag] if series.size >= lag else mean_all)
        for window in WEEK_ROLLING:
            tail = series[-window:] if series.size else np.array([0.0])
            put(f"roll_mean_{window}", tail.mean())
            put(f"roll_std_{window}", tail.std() if tail.size > 1 else 0.0)
        put("roll_max_12", series[-12:].max() if series.size else 0.0)
        roll4 = series[-4:].mean() if series.size else 0.0
        roll12 = series[-12:].mean() if series.size else 0.0
        put("trend_4_12", roll4 - roll12)

        iso_week = target.isocalendar()[1]
        put("week_of_year", iso_week)
        put("month", target.month)
        put("quarter", (target.month - 1) // 3 + 1)
        put("week_index", len(history) + step)
        put("woy_sin", math.sin(2 * math.pi * iso_week / 52))
        put("woy_cos", math.cos(2 * math.pi * iso_week / 52))
        put("month_sin", math.sin(2 * math.pi * target.month / 12))
        put("month_cos", math.cos(2 * math.pi * target.month / 12))
        put("price_ratio", meta.get("price_ratio", 1.0))
        put("on_promotion", meta.get("on_promotion", 0.0))
        for name in ("category_id", "brand_id", "price", "inventory", "discount_pct", "rating_avg"):
            put(name, meta.get(name, 0.0))

        # Prediction = seasonal-naive level + learned residual correction.
        factor = profile.get(f"month_{target.month}", 1.0)
        baseline = float(max(profile.get("recent_mean", 0.0) * factor, 0.0))
        try:
            correction = float(artifact.model.predict(row.reshape(1, -1))[0])
            predicted = float(max(baseline + correction, 0.0))
            model_used = "naive_plus_gbm_residual"
        except Exception:  # noqa: BLE001 - never let inference failure break the response
            predicted = baseline
            model_used = "seasonal_naive_fallback"

        working.append(predicted)
        spread = max(artifact.residual_std, 0.35) * math.sqrt(step)  # uncertainty grows with horizon
        points.append({
            "week": step,
            "date": target.date().isoformat(),
            "predicted": round(predicted, 4),
            "lower": round(max(predicted - Z_95 * spread, 0.0), 4),
            "upper": round(predicted + Z_95 * spread, 4),
            "model": model_used,
        })
    return points


def forward_forecast(artifact: ForecastArtifact, product_id: int, horizon_days: int,
                     product_meta: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Daily forecast points derived from the weekly model path."""
    horizon_days = max(1, int(horizon_days))
    n_weeks = math.ceil(horizon_days / 7)
    weekly = weekly_forecast(artifact, product_id, n_weeks, product_meta)
    last_date = _parse_last_date(artifact)

    daily: list[dict[str, Any]] = []
    for day in range(1, horizon_days + 1):
        week_slot = weekly[min((day - 1) // 7, len(weekly) - 1)]
        target = last_date + timedelta(days=day)
        daily.append({
            "day": day,
            "date": target.date().isoformat(),
            "predicted": round(week_slot["predicted"] / 7.0, 4),
            "lower": round(week_slot["lower"] / 7.0, 4),
            "upper": round(week_slot["upper"] / 7.0, 4),
            "model": week_slot["model"],
        })
    return daily


def horizon_totals(points: list[dict[str, Any]],
                   horizons: tuple[int, ...] = (7, 14, 30)) -> dict[str, dict[str, float]]:
    """Cumulative predicted demand over each horizon - the replenishment number."""
    out: dict[str, dict[str, float]] = {}
    for h in horizons:
        window = points[:h]
        if not window:
            continue
        total = sum(p["predicted"] for p in window)
        out[f"next_{h}_days"] = {
            "predicted_units": round(total, 3),
            "lower_bound": round(sum(p["lower"] for p in window), 3),
            "upper_bound": round(sum(p["upper"] for p in window), 3),
            "daily_average": round(total / len(window), 4),
        }
    return out
