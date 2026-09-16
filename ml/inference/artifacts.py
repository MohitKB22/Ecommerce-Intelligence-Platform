"""Serialisable model artifacts.

These classes are the boundary between training and serving: a training
pipeline builds one, the registry pickles it, and the API loads it and calls its
scoring methods. Defining them here (rather than inside a training script) keeps
unpickling stable and lets the backend depend on inference code only.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _topk(scores: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, range(min(k + (len(exclude) if exclude else 0), scores.size)))[
        : k + (len(exclude) if exclude else 0)
    ]
    idx = idx[np.argsort(-scores[idx])]
    if exclude:
        idx = [int(i) for i in idx if int(i) not in exclude][:k]
    else:
        idx = [int(i) for i in idx[:k]]
    return list(idx)


@dataclass
class RecommenderArtifact:
    """Hybrid recommender: latent CF factors + item-item CF + content + popularity."""

    item_ids: np.ndarray                     # (n_items,) product ids in matrix order
    user_ids: np.ndarray                     # (n_users,) user ids in matrix order
    item_factors: np.ndarray                 # (n_items, f) truncated-SVD item embeddings
    user_factors: np.ndarray                 # (n_users, f) user embeddings in the same space
    neighbour_idx: np.ndarray                # (n_items, K) item-item CF neighbours
    neighbour_sim: np.ndarray                # (n_items, K) cosine similarity of those neighbours
    content_vectors: np.ndarray              # (n_items, d) L2-normalised text embeddings
    popularity: np.ndarray                   # (n_items,) normalised long-run popularity
    trending: np.ndarray                     # (n_items,) recent-window popularity
    conversion_rate: np.ndarray              # (n_items,) view -> purchase rate
    margin: np.ndarray                       # (n_items,) normalised gross margin (business score)
    availability: np.ndarray                 # (n_items,) 1.0 in stock, 0.0 otherwise
    rating: np.ndarray                       # (n_items,) mean rating scaled to 0-1
    category_of_item: np.ndarray             # (n_items,) category id
    brand_of_item: np.ndarray                # (n_items,) brand id
    price: np.ndarray                        # (n_items,) effective price
    cooccurrence_idx: np.ndarray             # (n_items, C) frequently-bought-together neighbours
    cooccurrence_score: np.ndarray           # (n_items, C) lift-weighted co-purchase strength
    user_history: dict[int, list[int]] = field(default_factory=dict)   # user_id -> item positions
    popularity_blend: np.ndarray | None = None   # cached 0.7*popularity + 0.3*trending
    embedder: Any = None                     # text embedder for query-time encoding
    weights: dict[str, float] = field(default_factory=dict)
    version: str = ""

    # ---- index helpers ----------------------------------------------
    def __post_init__(self) -> None:
        self._item_pos = {int(pid): i for i, pid in enumerate(self.item_ids)}
        self._user_pos = {int(uid): i for i, uid in enumerate(self.user_ids)}

    def item_position(self, product_id: int) -> int | None:
        return self._item_pos.get(int(product_id))

    def user_position(self, user_id: int) -> int | None:
        return self._user_pos.get(int(user_id))

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_pos

    # ---- scoring components -----------------------------------------
    def collaborative_scores(self, user_id: int) -> np.ndarray:
        pos = self.user_position(user_id)
        if pos is None:
            return np.zeros(self.n_items, dtype=np.float32)
        raw = self.item_factors @ self.user_factors[pos]
        return _minmax(raw)

    def item_cf_scores(self, seed_positions: Sequence[int], seed_weights: Sequence[float] | None = None) -> np.ndarray:
        """Score the catalogue by similarity to a set of seed items (item-item CF)."""
        scores = np.zeros(self.n_items, dtype=np.float32)
        if not len(seed_positions):
            return scores
        weights = list(seed_weights) if seed_weights is not None else [1.0] * len(seed_positions)
        for pos, w in zip(seed_positions, weights):
            if pos is None or pos < 0 or pos >= self.n_items:
                continue
            np.add.at(scores, self.neighbour_idx[pos], self.neighbour_sim[pos] * float(w))
        return _minmax(scores)

    def content_scores(self, seed_positions: Sequence[int], seed_weights: Sequence[float] | None = None) -> np.ndarray:
        if not len(seed_positions) or self.content_vectors.size == 0:
            return np.zeros(self.n_items, dtype=np.float32)
        weights = np.asarray(list(seed_weights) if seed_weights is not None else [1.0] * len(seed_positions),
                             dtype=np.float32)
        profile = (self.content_vectors[list(seed_positions)] * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(self.n_items, dtype=np.float32)
        return _minmax(self.content_vectors @ (profile / norm))

    def personalization_scores(self, category_affinity: dict[int, float],
                               brand_affinity: dict[int, float]) -> np.ndarray:
        scores = np.zeros(self.n_items, dtype=np.float32)
        if category_affinity:
            cat_lookup = np.array([category_affinity.get(int(c), 0.0) for c in self.category_of_item],
                                  dtype=np.float32)
            scores += 0.65 * cat_lookup
        if brand_affinity:
            brand_lookup = np.array([brand_affinity.get(int(b), 0.0) for b in self.brand_of_item], dtype=np.float32)
            scores += 0.35 * brand_lookup
        return _minmax(scores)

    def business_scores(self) -> np.ndarray:
        return _minmax(0.55 * self.margin + 0.25 * self.conversion_rate + 0.20 * self.availability)

    def popularity_scores(self) -> np.ndarray:
        if self.popularity_blend is not None:
            return self.popularity_blend
        return _minmax(0.7 * self.popularity + 0.3 * self.trending)

    # ---- public recommenders ----------------------------------------
    def recommend_for_user(self, user_id: int, k: int = 10, *, exclude: Iterable[int] = (),
                           weights: dict[str, float] | None = None,
                           category_affinity: dict[int, float] | None = None,
                           brand_affinity: dict[int, float] | None = None,
                           seed_positions: Sequence[int] | None = None,
                           seed_weights: Sequence[float] | None = None) -> list[tuple[int, float, dict[str, float]]]:
        """Return [(product_id, final_score, component_breakdown)] ranked desc."""
        w = {**self.weights, **(weights or {})}
        history = self.user_history.get(int(user_id), [])
        seeds = list(seed_positions) if seed_positions is not None else history[-25:]

        collab = self.collaborative_scores(user_id)
        item_cf = self.item_cf_scores(seeds, seed_weights)
        collaborative = _minmax(0.6 * collab + 0.4 * item_cf) if seeds else collab
        content = self.content_scores(seeds, seed_weights)
        popularity = self.popularity_scores()
        personal = self.personalization_scores(category_affinity or {}, brand_affinity or {})
        business = self.business_scores()

        # Popularity is by far the strongest single ranking signal, and raw CF /
        # content scores are heavily correlated with it. Projecting the
        # popularity direction out of them means each component contributes
        # genuinely orthogonal information instead of re-voting for popularity.
        collaborative = _residualise(collaborative, popularity)
        content = _residualise(content, popularity)

        final = (
            w.get("collaborative", 0.35) * collaborative
            + w.get("content", 0.25) * content
            + w.get("popularity", 0.15) * popularity
            + w.get("personalization", 0.15) * personal
            + w.get("business", 0.10) * business
        )
        final = final * (0.35 + 0.65 * self.availability)  # damp out-of-stock items

        excluded = {p for p in (self.item_position(pid) for pid in exclude) if p is not None}
        excluded.update(history)
        top = _topk(final, k, excluded)
        return [
            (
                int(self.item_ids[i]),
                float(final[i]),
                {
                    "collaborative": round(float(collaborative[i]), 5),
                    "content": round(float(content[i]), 5),
                    "popularity": round(float(popularity[i]), 5),
                    "personalization": round(float(personal[i]), 5),
                    "business": round(float(business[i]), 5),
                },
            )
            for i in top
        ]

    def similar_items(self, product_id: int, k: int = 10) -> list[tuple[int, float]]:
        """Content + behavioural similarity blend, works for brand-new products."""
        pos = self.item_position(product_id)
        if pos is None:
            return []
        content_sim = self.content_vectors @ self.content_vectors[pos] if self.content_vectors.size else np.zeros(
            self.n_items, dtype=np.float32
        )
        behaviour = np.zeros(self.n_items, dtype=np.float32)
        np.add.at(behaviour, self.neighbour_idx[pos], self.neighbour_sim[pos])
        same_category = (self.category_of_item == self.category_of_item[pos]).astype(np.float32)
        blended = 0.5 * _minmax(content_sim) + 0.35 * _minmax(behaviour) + 0.15 * same_category
        blended[pos] = -1.0
        top = _topk(blended, k)
        return [(int(self.item_ids[i]), round(float(blended[i]), 5)) for i in top]

    def frequently_bought_together(self, product_id: int, k: int = 5) -> list[tuple[int, float]]:
        pos = self.item_position(product_id)
        if pos is None:
            return []
        idx = self.cooccurrence_idx[pos]
        score = self.cooccurrence_score[pos]
        pairs = [(int(self.item_ids[i]), float(s)) for i, s in zip(idx, score) if s > 0 and int(i) != pos]
        return sorted(pairs, key=lambda x: -x[1])[:k]

    def cold_start(self, k: int = 10, category_id: int | None = None) -> list[tuple[int, float]]:
        """New-user fallback: popularity + trending + rating, optionally per category."""
        score = _minmax(0.45 * self.popularity + 0.3 * self.trending + 0.25 * self.rating)
        score = score * (0.3 + 0.7 * self.availability)
        if category_id is not None:
            mask = (self.category_of_item == int(category_id)).astype(np.float32)
            score = score * mask
        top = _topk(score, k)
        return [(int(self.item_ids[i]), round(float(score[i]), 5)) for i in top if score[i] > 0]

    def trending_items(self, k: int = 10) -> list[tuple[int, float]]:
        top = _topk(self.trending * (0.3 + 0.7 * self.availability), k)
        return [(int(self.item_ids[i]), round(float(self.trending[i]), 5)) for i in top]


def _residualise(component: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Remove the linear projection of `base` from `component`, then rescale.

    Standard de-biasing: without it, a component that merely re-encodes
    popularity adds noise to a blend that already scores popularity directly.
    """
    b = base - base.mean()
    denom = float((b * b).sum())
    if denom < 1e-9:
        return component
    beta = float(((component - component.mean()) * b).sum()) / denom
    return _minmax(component - beta * base)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass
class SegmentationArtifact:
    scaler: Any
    kmeans: Any
    feature_names: list[str]
    cluster_labels: dict[int, str]           # cluster id -> human segment name
    cluster_profiles: dict[int, dict[str, float]]
    version: str = ""

    def assign(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (cluster_ids, confidence) for a feature matrix."""
        scaled = self.scaler.transform(features)
        clusters = self.kmeans.predict(scaled)
        distances = self.kmeans.transform(scaled)
        nearest = np.sort(distances, axis=1)
        if nearest.shape[1] > 1:
            # confidence = relative margin between best and second-best centroid
            confidence = 1.0 - (nearest[:, 0] / np.clip(nearest[:, 1], 1e-6, None))
        else:
            confidence = np.ones(len(clusters))
        return clusters, np.clip(confidence, 0.0, 1.0)

    def label_for(self, cluster_id: int) -> str:
        return self.cluster_labels.get(int(cluster_id), "unclassified")


@dataclass
class SentimentArtifact:
    vectorizer: Any
    classifier: Any
    labels: list[str]
    aspect_keywords: dict[str, list[str]]
    negation_terms: list[str]
    aspect_phrases: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    version: str = ""

    def predict(self, texts: Sequence[str]) -> tuple[list[str], list[float]]:
        if not texts:
            return [], []
        X = self.vectorizer.transform(list(texts))
        proba = self.classifier.predict_proba(X)
        idx = proba.argmax(axis=1)
        classes = list(self.classifier.classes_)
        labels = [str(classes[i]) for i in idx]
        # signed polarity score in [-1, 1]: P(positive) - P(negative)
        pos_i = classes.index("positive") if "positive" in classes else None
        neg_i = classes.index("negative") if "negative" in classes else None
        scores = []
        for row in proba:
            p = float(row[pos_i]) if pos_i is not None else 0.0
            n = float(row[neg_i]) if neg_i is not None else 0.0
            scores.append(round(p - n, 5))
        return labels, scores

    def extract_aspects(self, text: str) -> dict[str, str]:
        """Clause-level aspect polarity with negation handling.

        Polarity comes from the signed score P(positive) - P(negative) rather
        than the arg-max label. Whole-review training makes the arg-max label
        strongly positive-biased on short clauses, which collapsed aspect
        polarity accuracy; a symmetric threshold on the signed score restores it.
        Explicit lexicon evidence overrides the classifier when the clause
        contains a known polarity phrase for that aspect.
        """
        import re

        result: dict[str, str] = {}
        confidence: dict[str, float] = {}
        clauses = [c for c in re.split(
            r"[.!?;,]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\band yet\b",
            (text or "").lower()) if c.strip()]
        if not clauses:
            return result

        _, scores = self.predict(clauses)
        for clause, score in zip(clauses, scores):
            negated = any(term in clause for term in self.negation_terms)
            for aspect, keywords in self.aspect_keywords.items():
                if not any(kw in clause for kw in keywords):
                    continue

                lexical = self._lexicon_polarity(aspect, clause)
                if lexical is not None:
                    polarity, strength = lexical, 1.0
                else:
                    effective = -score if negated else score
                    if effective > 0.10:
                        polarity = "positive"
                    elif effective < -0.10:
                        polarity = "negative"
                    else:
                        polarity = "neutral"
                    strength = abs(effective)

                if strength >= confidence.get(aspect, -1.0):
                    result[aspect] = polarity
                    confidence[aspect] = strength
        return result

    def _lexicon_polarity(self, aspect: str, clause: str) -> str | None:
        """Direct evidence from the aspect lexicon, when the artifact carries it."""
        phrases = (self.aspect_phrases or {}).get(aspect)
        if not phrases:
            return None
        for polarity in ("positive", "negative"):
            for phrase in phrases.get(polarity, ()):
                # match on a distinctive fragment rather than the whole sentence
                fragment = phrase.lower()[:28]
                if fragment and fragment in clause:
                    return polarity
        return None


@dataclass
class ForecastArtifact:
    model: Any
    baseline_profile: dict[int, dict[str, float]]   # product_id -> {mean, dow factors, residual std}
    feature_names: list[str]
    global_dow_factor: list[float]
    residual_std: float
    trained_products: list[int]
    history_tail: dict[int, list[float]]            # product_id -> last N daily units
    last_date: str = ""
    version: str = ""

    def seasonal_naive(self, product_id: int, horizon: int) -> list[float]:
        tail = self.history_tail.get(int(product_id), [])
        if not tail:
            return [0.0] * horizon
        window = tail[-7:] if len(tail) >= 7 else tail
        return [float(window[i % len(window)]) for i in range(horizon)]


@dataclass
class PricePredictionArtifact:
    model: Any
    feature_names: list[str]
    category_stats: dict[int, dict[str, float]]
    global_stats: dict[str, float]
    feature_importance: dict[str, float]
    residual_std: float
    version: str = ""
