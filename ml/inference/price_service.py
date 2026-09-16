"""Price inference and local explainability."""
from __future__ import annotations

from typing import Any

import numpy as np

from ml.inference.artifacts import PricePredictionArtifact

TREND_THRESHOLD = 0.04


def explain_price(artifact: PricePredictionArtifact, features: np.ndarray, current_price: float,
                  category_id: int | None = None, top_n: int = 6) -> dict[str, Any]:
    """Predict a price and explain which features pushed it up or down.

    Local attribution is measured by ablation: each feature is replaced with the
    category-median value and the change in predicted log-price is recorded.
    Positive contribution == this feature justifies a higher price.
    """
    features = np.asarray(features, dtype=float).reshape(1, -1)
    reference = artifact.category_stats.get(int(category_id or -1), {}).get(
        "median_features", artifact.global_stats.get("median_features")
    )
    reference = np.asarray(reference, dtype=float) if reference is not None else features.copy().ravel()
    if reference.shape[0] != features.shape[1]:
        reference = features.ravel().copy()

    log_pred = float(artifact.model.predict(features)[0])
    predicted_price = float(np.exp(log_pred))

    contributions: dict[str, float] = {}
    batch = np.repeat(features, len(artifact.feature_names), axis=0)
    for i in range(len(artifact.feature_names)):
        batch[i, i] = reference[i]
    try:
        ablated = artifact.model.predict(batch)
        for i, name in enumerate(artifact.feature_names):
            contributions[name] = round(float(log_pred - ablated[i]), 6)
    except Exception:  # noqa: BLE001 - explanation must never break prediction
        contributions = {}

    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    delta = (predicted_price - current_price) / current_price if current_price > 0 else 0.0
    if delta > TREND_THRESHOLD:
        trend = "up"
    elif delta < -TREND_THRESHOLD:
        trend = "down"
    else:
        trend = "stable"

    # Confidence shrinks as the residual spread grows.
    spread = max(artifact.residual_std, 1e-6)
    confidence = float(np.clip(1.0 - min(spread / 0.5, 1.0) * 0.7 - min(abs(delta), 0.5) * 0.4, 0.05, 0.99))

    lower = float(np.exp(log_pred - 1.96 * spread))
    upper = float(np.exp(log_pred + 1.96 * spread))
    return {
        "predicted_price": round(predicted_price, 2),
        "price_trend": trend,
        "delta_pct": round(delta * 100, 3),
        "confidence": round(confidence, 4),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "contributions": dict(ranked),
    }
