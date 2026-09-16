"""Configurable search ranking.

The final score is a weighted blend of seven signals. Weights come from settings
(so they are tunable without a redeploy) and every component is returned with the
result, which is what makes search results explainable in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings


@dataclass(slots=True)
class RankingSignals:
    text: float = 0.0
    semantic: float = 0.0
    popularity: float = 0.0
    rating: float = 0.0
    conversion: float = 0.0
    personal: float = 0.0
    availability: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "text": round(self.text, 5),
            "semantic": round(self.semantic, 5),
            "popularity": round(self.popularity, 5),
            "rating": round(self.rating, 5),
            "conversion": round(self.conversion, 5),
            "personal": round(self.personal, 5),
            "availability": round(self.availability, 5),
        }


@dataclass
class RankedResult:
    product_id: int
    score: float
    signals: RankingSignals
    explanation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "score": round(self.score, 6),
            "signals": self.signals.as_dict(),
            "explanation": self.explanation,
        }


def default_weights() -> dict[str, float]:
    return dict(settings.ranking_weights)


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return default_weights()
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def score_result(signals: RankingSignals, weights: dict[str, float] | None = None) -> float:
    w = weights or default_weights()
    return (
        w.get("text", 0.0) * signals.text
        + w.get("semantic", 0.0) * signals.semantic
        + w.get("popularity", 0.0) * signals.popularity
        + w.get("rating", 0.0) * signals.rating
        + w.get("conversion", 0.0) * signals.conversion
        + w.get("personal", 0.0) * signals.personal
        + w.get("availability", 0.0) * signals.availability
    )


def explain(signals: RankingSignals, weights: dict[str, float] | None = None, top_n: int = 2) -> str:
    """Human-readable reason for a result's position."""
    w = weights or default_weights()
    contributions = {
        "keyword match": w.get("text", 0.0) * signals.text,
        "semantic similarity": w.get("semantic", 0.0) * signals.semantic,
        "popularity": w.get("popularity", 0.0) * signals.popularity,
        "customer rating": w.get("rating", 0.0) * signals.rating,
        "conversion rate": w.get("conversion", 0.0) * signals.conversion,
        "your preferences": w.get("personal", 0.0) * signals.personal,
        "availability": w.get("availability", 0.0) * signals.availability,
    }
    ranked = sorted(contributions.items(), key=lambda kv: -kv[1])
    drivers = [name for name, value in ranked[:top_n] if value > 0.001]
    if not drivers:
        return "Matched your query"
    return "Ranked by " + " and ".join(drivers)


def minmax_normalise(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return dict.fromkeys(scores, 1.0 if hi > 0 else 0.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
