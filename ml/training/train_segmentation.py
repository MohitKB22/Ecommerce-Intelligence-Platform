"""Customer segmentation via RFM + K-Means.

    python -m ml.training.train_segmentation

Builds an RFM-plus feature matrix (recency, frequency, monetary, AOV, product
diversity, session frequency, discount affinity, tenure), standardises it,
selects k by silhouette score, then maps each cluster to a business-readable
segment name from its centroid profile.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.inference.artifacts import SegmentationArtifact
from ml.preprocessing.loaders import TrainingFrames
from ml.training.base import TrainingPipeline, run_cli

FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "product_diversity",
    "session_frequency", "discount_affinity", "tenure_days", "review_count",
]
CANDIDATE_K = (4, 5, 6, 7, 8)
SILHOUETTE_TOLERANCE = 0.88  # accept k with silhouette >= 88% of the best
REVENUE_STATUSES = ("paid", "shipped", "delivered")


class SegmentationPipeline(TrainingPipeline):
    name = "segmentation"
    algorithm = "RFM feature engineering + StandardScaler + K-Means (k selected by silhouette)"
    requires = ("users", "order_items", "events")

    def preprocess(self, frames: TrainingFrames) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        users = frames.users[frames.users["role"] != "admin"].copy()
        orders = frames.orders[frames.orders["status"].isin(REVENUE_STATUSES)].copy()
        items = frames.order_items[frames.order_items["status"].isin(REVENUE_STATUSES)].copy()
        events = frames.events
        reviews = frames.reviews

        order_agg = orders.groupby("user_id").agg(
            frequency=("id", "nunique"),
            monetary=("total_amount", "sum"),
            avg_order_value=("total_amount", "mean"),
            last_order=("placed_at", "max"),
            discount_total=("discount_amount", "sum"),
        )
        diversity = items.groupby("user_id")["product_id"].nunique().rename("product_diversity")
        sessions = events.groupby("user_id")["session_id"].nunique().rename("session_frequency")
        review_count = reviews.groupby("user_id").size().rename("review_count")

        df = users[["id", "created_at"]].rename(columns={"id": "user_id"}).set_index("user_id")
        df = df.join([order_agg, diversity, sessions, review_count])
        df["frequency"] = df["frequency"].fillna(0.0)
        df["monetary"] = df["monetary"].fillna(0.0)
        df["avg_order_value"] = df["avg_order_value"].fillna(0.0)
        df["product_diversity"] = df["product_diversity"].fillna(0.0)
        df["session_frequency"] = df["session_frequency"].fillna(0.0)
        df["review_count"] = df["review_count"].fillna(0.0)
        df["discount_total"] = df["discount_total"].fillna(0.0)

        # Never-purchased customers get the maximum observed recency, not NaN.
        max_recency = float((now - orders["placed_at"].min()).days) if not orders.empty else 365.0
        df["recency_days"] = (now - df["last_order"]).dt.days.astype(float)
        df["recency_days"] = df["recency_days"].fillna(max_recency).clip(0, max_recency)
        df["tenure_days"] = (now - df["created_at"]).dt.days.astype(float).fillna(0.0).clip(lower=0)
        df["discount_affinity"] = (df["discount_total"] / df["monetary"].replace(0, np.nan)).fillna(0.0).clip(0, 1)

        # Log-compress the heavy-tailed monetary columns before scaling.
        for col in ("monetary", "avg_order_value", "frequency", "product_diversity", "session_frequency",
                    "review_count"):
            df[col] = np.log1p(df[col].clip(lower=0))

        matrix = df[FEATURES].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        self._training_rows = int(len(matrix))
        self.logger.info("segmentation_features_built", users=len(matrix), features=len(FEATURES))
        return {"matrix": matrix, "frame": df, "user_ids": df.index.to_numpy()}

    def train(self, prepared: dict[str, Any]) -> SegmentationArtifact:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        matrix = prepared["matrix"]
        scaler = StandardScaler().fit(matrix)
        scaled = scaler.transform(matrix)

        candidates: list[tuple[Any, float, int, np.ndarray]] = []
        self._k_search: list[dict[str, float]] = []
        sample = min(len(scaled), 2000)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(scaled), size=sample, replace=False) if len(scaled) > sample else np.arange(len(scaled))
        for k in CANDIDATE_K:
            if k >= len(scaled):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(scaled)
            score = float(silhouette_score(scaled[idx], labels[idx])) if len(set(labels)) > 1 else -1.0
            self._k_search.append({"k": k, "silhouette": round(score, 5), "inertia": round(float(km.inertia_), 2)})
            candidates.append((km, score, k, labels))
        if not candidates:
            raise ValueError("unable to fit any K-Means configuration")

        # Silhouette alone favours very coarse clusterings. Business segmentation
        # needs actionable granularity, so we take the LARGEST k whose silhouette
        # is within `SILHOUETTE_TOLERANCE` of the best score - a standard
        # "one-standard-error"-style selection rule.
        best_score = max(c[1] for c in candidates)
        acceptable = [c for c in candidates if c[1] >= best_score * SILHOUETTE_TOLERANCE]
        best = max(acceptable, key=lambda c: c[2])

        kmeans, silhouette, k, labels = best
        self._silhouette = silhouette
        self._labels = labels
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        profiles = {
            int(i): {name: round(float(centroids[i][j]), 4) for j, name in enumerate(FEATURES)}
            for i in range(k)
        }
        cluster_labels = self._name_clusters(profiles, labels)
        self.logger.info("kmeans_selected", k=k, silhouette=round(silhouette, 4))
        return SegmentationArtifact(
            scaler=scaler, kmeans=kmeans, feature_names=list(FEATURES),
            cluster_labels=cluster_labels, cluster_profiles=profiles,
        )

    @staticmethod
    def _name_clusters(profiles: dict[int, dict[str, float]], labels: np.ndarray) -> dict[int, str]:
        """Map centroids to business segment names by rank on the driving features.

        Names are assigned greedily by best-fitting cluster so two clusters never
        collapse onto the same label.
        """
        ids = list(profiles)
        def rank(feature: str) -> dict[int, float]:
            values = np.array([profiles[c][feature] for c in ids])
            order = values.argsort().argsort() / max(len(ids) - 1, 1)
            return {c: float(order[i]) for i, c in enumerate(ids)}

        r_recency = rank("recency_days")
        r_freq = rank("frequency")
        r_money = rank("monetary")
        r_aov = rank("avg_order_value")
        r_disc = rank("discount_affinity")
        r_tenure = rank("tenure_days")
        r_div = rank("product_diversity")

        rules: list[tuple[str, Any]] = [
            ("high_value", lambda c: 0.5 * r_money[c] + 0.3 * r_aov[c] + 0.2 * (1 - r_recency[c])),
            ("at_risk", lambda c: 0.6 * r_recency[c] + 0.25 * r_tenure[c] + 0.15 * (1 - r_freq[c])),
            ("new_customer", lambda c: 0.7 * (1 - r_tenure[c]) + 0.3 * (1 - r_freq[c])),
            ("loyal_customer", lambda c: 0.4 * r_freq[c] + 0.3 * r_tenure[c] + 0.3 * (1 - r_recency[c])),
            ("discount_seeker", lambda c: 0.75 * r_disc[c] + 0.25 * r_freq[c]),
            ("frequent_buyer", lambda c: 0.55 * r_freq[c] + 0.45 * r_div[c]),
            ("occasional_buyer", lambda c: 0.5 * (1 - r_freq[c]) + 0.5 * (1 - r_money[c])),
        ]
        assigned: dict[int, str] = {}
        available = set(ids)
        for label, scorer in rules:
            if not available:
                break
            best_cluster = max(available, key=scorer)
            assigned[best_cluster] = label
            available.discard(best_cluster)
        for leftover in available:
            assigned[leftover] = "occasional_buyer"
        return assigned

    def evaluate(self, artifact: SegmentationArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

        scaled = artifact.scaler.transform(prepared["matrix"])
        labels = artifact.kmeans.predict(scaled)
        sizes = pd.Series(labels).value_counts().sort_index()
        distribution = {
            artifact.label_for(int(cluster)): int(count) for cluster, count in sizes.items()
        }
        return {
            "silhouette_score": round(float(self._silhouette), 5),
            "davies_bouldin_score": round(float(davies_bouldin_score(scaled, labels)), 5),
            "calinski_harabasz_score": round(float(calinski_harabasz_score(scaled, labels)), 2),
            "n_clusters": int(artifact.kmeans.n_clusters),
            "k_search": self._k_search,
            "segment_distribution": distribution,
            "largest_segment_share": round(float(sizes.max() / sizes.sum()), 5),
            "customers_segmented": int(len(labels)),
        }

    def params(self, artifact: SegmentationArtifact) -> dict[str, Any]:
        return {
            "k": int(artifact.kmeans.n_clusters),
            "candidate_k": list(CANDIDATE_K),
            "silhouette_tolerance": SILHOUETTE_TOLERANCE,
            "features": list(FEATURES),
            "scaler": "StandardScaler",
            "cluster_labels": {str(k): v for k, v in artifact.cluster_labels.items()},
        }

    def feature_names(self, artifact: SegmentationArtifact) -> list[str]:
        return list(FEATURES)

    def baseline_stats(self, artifact: SegmentationArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        frame = prepared["frame"][FEATURES]
        return {
            "feature_means": {c: round(float(frame[c].mean()), 5) for c in FEATURES},
            "feature_stds": {c: round(float(frame[c].std()), 5) for c in FEATURES},
        }

    def persist_side_effects(self, artifact: SegmentationArtifact, prepared: dict[str, Any], version: str) -> None:
        """Write each customer's segment assignment to `customer_segments`."""
        from app.core.db import engine
        from app.models import CustomerSegment
        from sqlalchemy.orm import Session

        clusters, confidence = artifact.assign(prepared["matrix"])
        frame = prepared["frame"]
        rows = []
        for i, user_id in enumerate(prepared["user_ids"]):
            cluster = int(clusters[i])
            row = frame.iloc[i]
            rows.append({
                "user_id": int(user_id),
                "segment_label": artifact.label_for(cluster),
                "cluster_id": cluster,
                "model_version": version,
                "recency_days": float(row["recency_days"]),
                "frequency": float(row["frequency"]),
                "monetary": float(row["monetary"]),
                "avg_order_value": float(row["avg_order_value"]),
                "product_diversity": float(row["product_diversity"]),
                "session_frequency": float(row["session_frequency"]),
                "rfm_score": float(
                    (1 - row["recency_days"] / max(frame["recency_days"].max(), 1)) * 0.4
                    + (row["frequency"] / max(frame["frequency"].max(), 1e-9)) * 0.3
                    + (row["monetary"] / max(frame["monetary"].max(), 1e-9)) * 0.3
                ),
                "confidence": float(confidence[i]),
            })
        try:
            with Session(engine) as session:
                session.query(CustomerSegment).filter(CustomerSegment.model_version == version).delete()
                for start in range(0, len(rows), 1000):
                    session.bulk_insert_mappings(CustomerSegment, rows[start:start + 1000])
                session.commit()
            self.logger.info("segments_persisted", rows=len(rows), version=version)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("segment_persist_failed", error=str(exc))

    def notes(self, artifact: SegmentationArtifact) -> str:
        return (
            "RFM + behavioural K-Means segmentation. k chosen by silhouette score across "
            f"{list(CANDIDATE_K)}. Cluster centroids are mapped to business segment names by "
            "greedy rank-based assignment so each name is used at most once."
        )


if __name__ == "__main__":
    raise SystemExit(run_cli(SegmentationPipeline, "Train the customer segmentation model."))
