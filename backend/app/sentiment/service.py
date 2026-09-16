"""Review sentiment serving + product-level aggregation."""
from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.errors import NotFoundError
from app.core.logging_config import get_logger
from app.ml.model_store import get_model_store
from app.models import Product, Review

logger = get_logger("sentiment")
SENTIMENT_TTL = 600


class SentimentService:
    def __init__(self, db: Session):
        self.db = db
        self.store = get_model_store()
        self.cache = get_cache()

    def analyse_text(self, text: str) -> dict[str, Any]:
        """Score arbitrary text - used by the API and by review submission."""
        loaded = self.store.try_get("sentiment")
        if loaded is None:
            return {"label": "unknown", "score": 0.0, "aspects": {}, "model_version": "unavailable",
                    "note": "Sentiment model has not been trained yet."}
        artifact = loaded.artifact
        with self.store.timed("sentiment", entity_type="review") as box:
            labels, scores = artifact.predict([text])
            aspects = artifact.extract_aspects(text)
            box["prediction"] = float(scores[0]) if scores else 0.0
        return {
            "label": labels[0] if labels else "neutral",
            "score": scores[0] if scores else 0.0,
            "aspects": aspects,
            "positive_aspects": [a for a, p in aspects.items() if p == "positive"],
            "negative_aspects": [a for a, p in aspects.items() if p == "negative"],
            "model_version": loaded.version,
        }

    def product_sentiment(self, product_id: int) -> dict[str, Any]:
        product = self.db.get(Product, product_id)
        if product is None:
            raise NotFoundError(f"Product {product_id} was not found.")

        key = self.cache.key("sentiment", product_id)
        cached = self.cache.get_json(key)
        if cached is not None:
            return cached

        rows = self.db.execute(
            select(Review.sentiment_label, Review.sentiment_score, Review.rating, Review.aspects, Review.body)
            .where(Review.product_id == product_id)
        ).all()
        if not rows:
            payload = {
                "product_id": product_id, "review_count": 0, "distribution": {},
                "average_score": 0.0, "aspects": [], "top_complaints": [], "top_praise": [],
                "keywords": [], "summary": "No reviews yet for this product.",
            }
            self.cache.set_json(key, payload, SENTIMENT_TTL)
            return payload

        # Score any reviews the batch pipeline has not yet labelled.
        unlabelled = [r for r in rows if r[0] is None]
        if unlabelled and self.store.try_get("sentiment") is not None:
            self._backfill(product_id)
            rows = self.db.execute(
                select(Review.sentiment_label, Review.sentiment_score, Review.rating, Review.aspects, Review.body)
                .where(Review.product_id == product_id)
            ).all()

        labels = [r[0] for r in rows if r[0]]
        scores = [float(r[1]) for r in rows if r[1] is not None]
        distribution = Counter(labels)
        total = sum(distribution.values()) or 1

        aspect_tally: dict[str, Counter] = {}
        for _, _, _, aspects, _ in rows:
            for aspect, polarity in (aspects or {}).items():
                aspect_tally.setdefault(aspect, Counter())[polarity] += 1

        aspect_summary = []
        for aspect, counter in aspect_tally.items():
            mentions = sum(counter.values())
            positive = counter.get("positive", 0)
            negative = counter.get("negative", 0)
            aspect_summary.append({
                "aspect": aspect,
                "label": aspect.replace("_", " ").title(),
                "mentions": mentions,
                "positive": positive,
                "negative": negative,
                "neutral": counter.get("neutral", 0),
                "sentiment": round((positive - negative) / mentions, 4) if mentions else 0.0,
            })
        aspect_summary.sort(key=lambda a: -a["mentions"])

        payload = {
            "product_id": product_id,
            "review_count": len(rows),
            "distribution": {
                "positive": distribution.get("positive", 0),
                "neutral": distribution.get("neutral", 0),
                "negative": distribution.get("negative", 0),
            },
            "distribution_pct": {
                "positive": round(distribution.get("positive", 0) / total, 4),
                "neutral": round(distribution.get("neutral", 0) / total, 4),
                "negative": round(distribution.get("negative", 0) / total, 4),
            },
            "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "average_rating": round(float(sum(r[2] for r in rows) / len(rows)), 2),
            "aspects": aspect_summary,
            "top_praise": [a for a in aspect_summary if a["sentiment"] > 0.2][:5],
            "top_complaints": sorted([a for a in aspect_summary if a["sentiment"] < -0.1],
                                     key=lambda a: a["sentiment"])[:5],
            "keywords": self._keywords([r[4] for r in rows if r[4]]),
            "summary": self._summary(distribution, aspect_summary, total),
        }
        self.cache.set_json(key, payload, SENTIMENT_TTL)
        return payload

    def _backfill(self, product_id: int) -> int:
        """Label any reviews the batch job has not processed yet."""
        loaded = self.store.try_get("sentiment")
        if loaded is None:
            return 0
        artifact = loaded.artifact
        pending = self.db.execute(
            select(Review).where(Review.product_id == product_id, Review.sentiment_label.is_(None)).limit(500)
        ).scalars().all()
        if not pending:
            return 0
        texts = [f"{r.title}. {r.body}" for r in pending]
        labels, scores = artifact.predict(texts)
        for review, label, score, text in zip(pending, labels, scores, texts):
            review.sentiment_label = label
            review.sentiment_score = float(score)
            review.sentiment_model_version = loaded.version
            review.aspects = artifact.extract_aspects(text)
        self.db.commit()
        logger.info("sentiment_backfilled", product_id=product_id, rows=len(pending))
        return len(pending)

    @staticmethod
    def _keywords(bodies: list[str], top_n: int = 12) -> list[dict[str, Any]]:
        from ml.features.embeddings import tokenize

        counter: Counter = Counter()
        for body in bodies:
            counter.update(set(tokenize(body)))
        return [{"term": term, "count": count} for term, count in counter.most_common(top_n) if len(term) > 3]

    @staticmethod
    def _summary(distribution: Counter, aspects: list[dict[str, Any]], total: int) -> str:
        positive_share = distribution.get("positive", 0) / total
        praise = next((a["label"] for a in aspects if a["sentiment"] > 0.2), None)
        complaint = next((a["label"] for a in sorted(aspects, key=lambda x: x["sentiment"])
                          if a["sentiment"] < -0.1), None)
        if positive_share >= 0.7:
            base = f"Reviews are strongly positive ({positive_share:.0%} favourable)."
        elif positive_share >= 0.45:
            base = f"Reviews are mixed but lean positive ({positive_share:.0%} favourable)."
        else:
            base = f"Reviews are mostly critical (only {positive_share:.0%} favourable)."
        if praise:
            base += f" Customers consistently praise {praise.lower()}."
        if complaint:
            base += f" The most common complaint concerns {complaint.lower()}."
        return base

    def distribution_overview(self) -> dict[str, Any]:
        rows = self.db.execute(
            select(Review.sentiment_label, func.count(Review.id))
            .where(Review.sentiment_label.isnot(None))
            .group_by(Review.sentiment_label)
        ).all()
        counts = {label: int(count) for label, count in rows}
        total = sum(counts.values()) or 1
        return {
            "counts": counts,
            "percentages": {k: round(v / total, 4) for k, v in counts.items()},
            "total_labelled": sum(counts.values()),
            "total_reviews": int(self.db.execute(select(func.count(Review.id))).scalar_one() or 0),
        }
