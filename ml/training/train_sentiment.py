"""Review sentiment + aspect extraction.

    python -m ml.training.train_sentiment

Sentiment labels are derived from the star rating (the standard weak-supervision
signal for review corpora): 4-5 positive, 3 neutral, 1-2 negative. The model
learns to predict that label from review *text alone* using word+char TF-IDF and
calibrated logistic regression, so at inference time it can score text that has
no rating attached.

Aspect extraction runs sentence-level polarity over an aspect keyword lexicon
with negation handling, and is evaluated against the aspect ground truth
recorded by the data generator.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.taxonomy import ASPECT_LEXICON
from ml.evaluation.metrics import classification_report_dict
from ml.inference.artifacts import SentimentArtifact
from ml.preprocessing.loaders import TrainingFrames
from ml.training.base import REPO_ROOT, TrainingPipeline, run_cli

NEGATION_TERMS = ["not ", "no ", "never", "isn't", "wasn't", "doesn't", "didn't", "don't", "hardly", "barely",
                  "lacks", "lacking", "without"]
LABELS = ["negative", "neutral", "positive"]
TEST_SIZE = 0.25


def rating_to_label(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


class SentimentPipeline(TrainingPipeline):
    name = "sentiment"
    algorithm = "word+char TF-IDF -> calibrated LogisticRegression (3-class) + lexicon aspect extraction"
    requires = ("reviews",)

    def preprocess(self, frames: TrainingFrames) -> dict[str, Any]:
        reviews = frames.reviews.copy()
        reviews["text"] = (reviews["title"].fillna("") + ". " + reviews["body"].fillna("")).str.strip()
        reviews = reviews[reviews["text"].str.len() > 10]
        if reviews.empty:
            raise ValueError("no usable review text")
        reviews["label"] = reviews["rating"].astype(int).map(rating_to_label)

        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            reviews, test_size=TEST_SIZE, random_state=42, stratify=reviews["label"]
        )
        self._training_rows = int(len(train_df))
        self.logger.info(
            "sentiment_split", train=len(train_df), test=len(test_df),
            distribution=reviews["label"].value_counts().to_dict(),
        )
        return {"train": train_df, "test": test_df, "all": reviews}

    def train(self, prepared: dict[str, Any]) -> SentimentArtifact:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        train_df = prepared["train"]
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                     max_features=40000)),
        ])
        X = vectorizer.fit_transform(train_df["text"])
        classifier = LogisticRegression(
            max_iter=2000, C=4.0, class_weight="balanced", random_state=42,
        )
        classifier.fit(X, train_df["label"])
        return SentimentArtifact(
            vectorizer=vectorizer,
            classifier=classifier,
            labels=list(LABELS),
            aspect_keywords={a: meta["keywords"] for a, meta in ASPECT_LEXICON.items()},
            negation_terms=list(NEGATION_TERMS),
            aspect_phrases={
                a: {"positive": list(meta["positive"]), "negative": list(meta["negative"])}
                for a, meta in ASPECT_LEXICON.items()
            },
        )

    def evaluate(self, artifact: SentimentArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        test_df = prepared["test"]
        predicted, scores = artifact.predict(test_df["text"].tolist())
        report = classification_report_dict(test_df["label"].tolist(), predicted, labels=LABELS)

        # Majority-class baseline for context.
        majority = test_df["label"].mode().iloc[0]
        baseline = classification_report_dict(
            test_df["label"].tolist(), [majority] * len(test_df), labels=LABELS
        )
        metrics: dict[str, Any] = {
            **report,
            "baseline_majority_accuracy": baseline["accuracy"],
            "baseline_majority_f1_macro": baseline["f1_macro"],
            "mean_polarity_score": round(float(np.mean(scores)), 5),
        }
        metrics.update(self._evaluate_aspects(artifact))
        return metrics

    def _evaluate_aspects(self, artifact: SentimentArtifact) -> dict[str, Any]:
        """Score aspect extraction against the generator's recorded ground truth."""
        truth_path = REPO_ROOT / "data" / "seed" / "review_aspect_truth.json"
        if not truth_path.exists():
            return {"aspect_evaluation": "unavailable (no ground-truth file)"}
        try:
            truth_records = json.loads(truth_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"aspect_evaluation": "unavailable (unreadable ground-truth file)"}

        reviews = self.frames.reviews.set_index("id") if self.frames is not None else pd.DataFrame()
        if reviews.empty:
            return {"aspect_evaluation": "unavailable (no reviews loaded)"}

        sample = truth_records[:1200]
        tp = fp = fn = 0
        polarity_hits = polarity_total = 0
        for record in sample:
            rid = record.get("id")
            truth = record.get("aspects_truth") or {}
            if rid not in reviews.index or not truth:
                continue
            row = reviews.loc[rid]
            text = f"{row['title']}. {row['body']}"
            predicted = artifact.extract_aspects(text)
            for aspect, polarity in truth.items():
                if aspect in predicted:
                    tp += 1
                    polarity_total += 1
                    polarity_hits += int(predicted[aspect] == polarity)
                else:
                    fn += 1
            for aspect in predicted:
                if aspect not in truth:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "aspect_detection_precision": round(precision, 5),
            "aspect_detection_recall": round(recall, 5),
            "aspect_detection_f1": round(f1, 5),
            "aspect_polarity_accuracy": round(polarity_hits / polarity_total, 5) if polarity_total else 0.0,
            "aspect_samples_evaluated": len(sample),
        }

    def params(self, artifact: SentimentArtifact) -> dict[str, Any]:
        return {
            "labels": LABELS,
            "test_size": TEST_SIZE,
            "label_rule": "rating>=4 positive, ==3 neutral, <=2 negative",
            "classifier": "LogisticRegression(C=4.0, class_weight=balanced)",
            "aspects": list(artifact.aspect_keywords),
        }

    def feature_names(self, artifact: SentimentArtifact) -> list[str]:
        return ["tfidf_word_1_2", "tfidf_char_wb_3_5"]

    def baseline_stats(self, artifact: SentimentArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        dist = prepared["all"]["label"].value_counts(normalize=True).to_dict()
        return {
            "label_distribution": {k: round(float(v), 5) for k, v in dist.items()},
            "mean_review_chars": round(float(prepared["all"]["text"].str.len().mean()), 2),
        }

    def persist_side_effects(self, artifact: SentimentArtifact, prepared: dict[str, Any], version: str) -> None:
        """Score every review in the database and store label/score/aspects."""
        from app.core.db import engine
        from app.models import Review
        from sqlalchemy.orm import Session

        reviews = prepared["all"]
        labels, scores = artifact.predict(reviews["text"].tolist())
        updates = []
        for (_, row), label, score in zip(reviews.iterrows(), labels, scores):
            updates.append({
                "id": int(row["id"]),
                "sentiment_label": label,
                "sentiment_score": float(score),
                "sentiment_model_version": version,
                "aspects": artifact.extract_aspects(row["text"]),
            })
        try:
            with Session(engine) as session:
                for start in range(0, len(updates), 1000):
                    session.bulk_update_mappings(Review, updates[start:start + 1000])
                session.commit()
            self.logger.info("review_sentiment_persisted", rows=len(updates), version=version)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("sentiment_persist_failed", error=str(exc))

    def notes(self, artifact: SentimentArtifact) -> str:
        return (
            "Three-class review sentiment from text only, trained with rating-derived weak labels. "
            "Aspect extraction splits reviews on contrastive conjunctions, scores each clause and "
            "applies negation inversion, then is scored against generator ground truth."
        )


if __name__ == "__main__":
    raise SystemExit(run_cli(SentimentPipeline, "Train the review sentiment model."))
