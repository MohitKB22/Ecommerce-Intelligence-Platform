"""Hybrid recommendation training pipeline.

    python -m ml.training.train_recommendation

Builds, from the implicit-feedback event stream and the order history:
  * latent user/item factors (truncated SVD of the confidence-weighted matrix)
  * an item-item collaborative neighbourhood (cosine over item columns)
  * content embeddings over product text
  * popularity / trending / conversion / margin / availability signals
  * a lift-weighted co-purchase graph for "frequently bought together"

Evaluation uses a strict temporal split: the model only ever sees interactions
before the cut-off, and is scored on purchases that happen after it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.metrics import coverage, ranking_report
from ml.features.embeddings import build_embedder
from ml.inference.artifacts import RecommenderArtifact
from ml.inference.artifacts import _minmax as _minmax_local
from ml.inference.artifacts import _residualise as _residualise_local
from ml.preprocessing.loaders import TrainingFrames
from ml.training.base import TrainingPipeline, run_cli


def _bm25_weight(matrix, k1: float = 1.6, b: float = 0.75):
    """BM25 re-weighting of the implicit feedback matrix.

    Item IDF down-weights globally popular items and the length term normalises
    heavy users, so the latent factors encode taste rather than volume. This is
    the same weighting used by production implicit-feedback ALS implementations.
    """
    from scipy.sparse import csr_matrix as _csr

    n_users = matrix.shape[0]
    df = np.diff(matrix.tocsc().indptr)
    idf = np.log((n_users - df + 0.5) / (df + 0.5) + 1.0)
    user_len = np.asarray(matrix.sum(axis=1)).ravel()
    avg_len = float(user_len.mean()) or 1.0
    coo = matrix.tocoo()
    data = (
        coo.data * (k1 + 1.0)
        / (coo.data + k1 * (1.0 - b + b * user_len[coo.row] / avg_len))
        * idf[coo.col]
    )
    return _csr((data, (coo.row, coo.col)), shape=matrix.shape)


EVENT_CONFIDENCE = {
    "product_view": 1.0,
    "click": 1.5,
    "wishlist": 3.0,
    "add_to_cart": 4.0,
    "remove_from_cart": -1.0,
    "purchase": 6.0,
    "review": 3.0,
}
DEFAULT_WEIGHTS = {
    "collaborative": 0.35,
    "content": 0.25,
    "popularity": 0.15,
    "personalization": 0.15,
    "business": 0.10,
}
N_FACTORS = 32
N_NEIGHBOURS = 50
N_COOCCURRENCE = 12
TRENDING_WINDOW_DAYS = 30
TEST_FRACTION = 0.15


class RecommendationPipeline(TrainingPipeline):
    name = "recommendation"
    algorithm = "TruncatedSVD latent factors + item-item cosine CF + TF-IDF/SVD content + popularity blend"
    requires = ("products", "users", "events", "order_items")

    # ---- preprocessing ------------------------------------------------
    def preprocess(self, frames: TrainingFrames) -> dict[str, Any]:
        products = frames.products.sort_values("id").reset_index(drop=True)
        events = frames.events.dropna(subset=["product_id", "user_id"]).copy()
        events["product_id"] = events["product_id"].astype(int)
        events["user_id"] = events["user_id"].astype(int)
        events = events[events["event_type"].isin(EVENT_CONFIDENCE)]

        if events.empty:
            raise ValueError("no usable interaction events")

        cutoff = events["occurred_at"].quantile(1 - TEST_FRACTION)
        train_events = events[events["occurred_at"] <= cutoff]
        test_events = events[(events["occurred_at"] > cutoff) & (events["event_type"].isin(("purchase", "add_to_cart")))]

        self._training_rows = int(len(train_events))
        self.logger.info(
            "temporal_split",
            cutoff=str(cutoff),
            train_interactions=len(train_events),
            test_interactions=len(test_events),
        )
        return {
            "products": products,
            "events": events,
            "train_events": train_events,
            "test_events": test_events,
            "order_items": frames.order_items,
            "cutoff": cutoff,
            "max_time": events["occurred_at"].max(),
        }

    # ---- artifact construction ----------------------------------------
    def _build(self, products: pd.DataFrame, events: pd.DataFrame, order_items: pd.DataFrame,
               max_time: pd.Timestamp, embedder=None) -> RecommenderArtifact:
        from scipy.sparse import csr_matrix

        item_ids = products["id"].to_numpy()
        item_pos = {int(pid): i for i, pid in enumerate(item_ids)}
        n_items = len(item_ids)

        events = events[events["product_id"].isin(item_pos)]
        user_ids = np.sort(events["user_id"].unique())
        user_pos = {int(uid): i for i, uid in enumerate(user_ids)}
        n_users = len(user_ids)

        rows = events["user_id"].map(user_pos).to_numpy()
        cols = events["product_id"].map(item_pos).to_numpy()
        conf = events["event_type"].map(EVENT_CONFIDENCE).astype(float).to_numpy()

        # Recency decay: a view from last week is worth more than one from last year.
        age_days = (max_time - events["occurred_at"]).dt.total_seconds().to_numpy() / 86400.0
        conf = conf * np.exp(-age_days / 180.0)

        matrix = csr_matrix((conf, (rows, cols)), shape=(n_users, n_items))
        matrix.sum_duplicates()
        matrix.data = np.clip(matrix.data, 0.0, None)
        matrix.eliminate_zeros()
        # log1p damping stops power users from dominating the factorisation
        matrix.data = np.log1p(matrix.data)
        matrix = _bm25_weight(matrix)

        n_factors = int(min(N_FACTORS, max(2, min(matrix.shape) - 1)))
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=n_factors, random_state=42, algorithm="randomized")
        user_factors = svd.fit_transform(matrix).astype(np.float32)
        item_factors = svd.components_.T.astype(np.float32)

        # Item-item CF neighbourhood over L2-normalised item columns.
        item_matrix = matrix.T.tocsr().astype(np.float32)
        norms = np.sqrt(item_matrix.multiply(item_matrix).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        from scipy.sparse import diags

        normalised = diags(1.0 / norms) @ item_matrix
        sim = (normalised @ normalised.T).toarray()
        np.fill_diagonal(sim, 0.0)
        k_n = int(min(N_NEIGHBOURS, max(1, n_items - 1)))
        neighbour_idx = np.argpartition(-sim, kth=k_n - 1, axis=1)[:, :k_n]
        order = np.argsort(-np.take_along_axis(sim, neighbour_idx, axis=1), axis=1)
        neighbour_idx = np.take_along_axis(neighbour_idx, order, axis=1).astype(np.int32)
        neighbour_sim = np.take_along_axis(sim, neighbour_idx, axis=1).astype(np.float32)

        # Content embeddings over title + description + tags + specs.
        texts = [
            f"{r.title} {r.category_name} {r.brand_name} {' '.join(r.tags or [])} "
            f"{' '.join(f'{k} {v}' for k, v in (r.specifications or {}).items())} {r.description}"
            for r in products.itertuples()
        ]
        if embedder is None:
            embedder = build_embedder(dim=128)
            content_vectors = embedder.fit_transform(texts)
        else:
            content_vectors = embedder.transform(texts)

        # Behavioural aggregates.
        views = events[events["event_type"].isin(("product_view", "click"))].groupby("product_id").size()
        purchases = events[events["event_type"] == "purchase"].groupby("product_id").size()
        popularity = np.zeros(n_items, dtype=np.float32)
        for pid, count in purchases.items():
            popularity[item_pos[int(pid)]] = float(count)
        popularity = np.log1p(popularity)
        popularity = popularity / (popularity.max() or 1.0)

        recent_cut = max_time - pd.Timedelta(days=TRENDING_WINDOW_DAYS)
        recent = events[(events["occurred_at"] >= recent_cut) & (events["event_type"].isin(("purchase", "add_to_cart", "product_view")))]
        trending = np.zeros(n_items, dtype=np.float32)
        for pid, count in recent.groupby("product_id").size().items():
            trending[item_pos[int(pid)]] = float(count)
        trending = np.log1p(trending)
        trending = trending / (trending.max() or 1.0)

        conversion = np.zeros(n_items, dtype=np.float32)
        for pid in item_pos:
            v = float(views.get(pid, 0.0))
            p = float(purchases.get(pid, 0.0))
            conversion[item_pos[pid]] = p / v if v >= 3 else 0.0
        conversion = np.clip(conversion, 0, 1)

        price = products["price"].to_numpy(dtype=np.float32)
        discount = products["discount_pct"].to_numpy(dtype=np.float32)
        effective_price = price * (1 - discount / 100.0)
        cost = products["base_cost"].to_numpy(dtype=np.float32)
        margin = np.clip((effective_price - cost) / np.clip(effective_price, 1e-3, None), 0, 1).astype(np.float32)
        availability = (products["inventory"].to_numpy() > 0).astype(np.float32)
        rating = (products["rating_avg"].to_numpy(dtype=np.float32) / 5.0).clip(0, 1)

        # Co-purchase graph with lift weighting (frequently bought together).
        cooc_idx, cooc_score = self._cooccurrence(order_items, item_pos, n_items)

        history: dict[int, list[int]] = {}
        positive = events[events["event_type"].isin(("purchase", "add_to_cart", "wishlist", "click", "product_view"))]
        for uid, grp in positive.sort_values("occurred_at").groupby("user_id"):
            history[int(uid)] = [item_pos[int(p)] for p in grp["product_id"] if int(p) in item_pos][-60:]

        return RecommenderArtifact(
            item_ids=item_ids,
            user_ids=user_ids,
            item_factors=item_factors,
            user_factors=user_factors,
            neighbour_idx=neighbour_idx,
            neighbour_sim=neighbour_sim,
            content_vectors=content_vectors.astype(np.float32),
            popularity=popularity,
            trending=trending,
            conversion_rate=conversion,
            margin=margin,
            availability=availability,
            rating=rating,
            category_of_item=products["category_id"].to_numpy(),
            brand_of_item=products["brand_id"].to_numpy(),
            price=effective_price,
            cooccurrence_idx=cooc_idx,
            cooccurrence_score=cooc_score,
            user_history=history,
            embedder=embedder,
            weights={**DEFAULT_WEIGHTS},
            popularity_blend=_minmax_local(0.7 * popularity + 0.3 * trending),
        )

    @staticmethod
    def _cooccurrence(order_items: pd.DataFrame, item_pos: dict[int, int], n_items: int):
        """Lift-weighted co-purchase neighbours from multi-item baskets."""
        from collections import defaultdict

        pair_counts: dict[tuple[int, int], float] = defaultdict(float)
        item_counts = np.zeros(n_items, dtype=np.float32)
        baskets = 0
        if not order_items.empty:
            for _, grp in order_items.groupby("order_id"):
                positions = sorted({item_pos[int(p)] for p in grp["product_id"] if int(p) in item_pos})
                if not positions:
                    continue
                baskets += 1
                for p in positions:
                    item_counts[p] += 1
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        pair_counts[(positions[i], positions[j])] += 1.0

        k = int(min(N_COOCCURRENCE, max(1, n_items - 1)))
        scores: list[dict[int, float]] = [{} for _ in range(n_items)]
        total = max(baskets, 1)
        for (a, b), count in pair_counts.items():
            pa = item_counts[a] / total
            pb = item_counts[b] / total
            expected = pa * pb * total
            lift = count / expected if expected > 0 else 0.0
            weight = float(np.log1p(count) * min(lift, 25.0))
            scores[a][b] = weight
            scores[b][a] = weight

        idx = np.zeros((n_items, k), dtype=np.int32)
        val = np.zeros((n_items, k), dtype=np.float32)
        for i, neigh in enumerate(scores):
            if not neigh:
                continue
            top = sorted(neigh.items(), key=lambda kv: -kv[1])[:k]
            for slot, (j, s) in enumerate(top):
                idx[i, slot] = j
                val[i, slot] = s
        return idx, val

    # ---- training / evaluation ----------------------------------------
    def train(self, prepared: dict[str, Any]) -> RecommenderArtifact:
        # Evaluation artifact: sees only pre-cutoff interactions.
        self._eval_artifact = self._build(
            prepared["products"], prepared["train_events"],
            prepared["order_items"][prepared["order_items"]["placed_at"] <= prepared["cutoff"]],
            prepared["cutoff"],
        )
        # Blend weights are fitted on a validation slice of the post-cutoff
        # window, never on the held-out test slice used for the reported metrics.
        self._fitted_weights = self._fit_weights(prepared)

        # Production artifact: rebuilt on the full history, reusing the embedder.
        artifact = self._build(
            prepared["products"], prepared["events"], prepared["order_items"],
            prepared["max_time"], embedder=self._eval_artifact.embedder,
        )
        artifact.weights = dict(self._fitted_weights)
        self._eval_artifact.weights = dict(self._fitted_weights)
        return artifact

    # ---- weight fitting ------------------------------------------------
    @staticmethod
    def _relevance_map(art: RecommenderArtifact, df: pd.DataFrame) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for uid, grp in df.groupby("user_id"):
            uid = int(uid)
            if art.knows_user(uid):
                out[uid] = {int(p) for p in grp["product_id"]}
        return out

    def _split_test(self, prepared: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        test = prepared["test_events"]
        if test.empty:
            return test, test
        mid = test["occurred_at"].quantile(0.5)
        return test[test["occurred_at"] <= mid], test[test["occurred_at"] > mid]

    def _fit_weights(self, prepared: dict[str, Any]) -> dict[str, float]:
        """Coordinate search over blend weights, maximising NDCG@10 on validation."""
        art = self._eval_artifact
        val_df, _ = self._split_test(prepared)
        relevant = self._relevance_map(art, val_df)
        users = list(relevant)[:1500]
        default = dict(DEFAULT_WEIGHTS)
        if len(users) < 25:
            self.logger.warning("weight_fitting_skipped", reason="insufficient validation users",
                                users=len(users))
            return default

        components = {u: self._components(art, u) for u in users}
        relevant = {u: relevant[u] for u in users}

        def ndcg(weights: dict[str, float]) -> float:
            # Rank exactly as evaluation does, including history exclusion, so the
            # fitted weights are optimal under the conditions they are scored in.
            recs = {u: self._rank_for_user(art, u, components[u], weights) for u in users}
            return ranking_report(recs, relevant, ks=(10,))["ndcg@10"]

        # Popularity-only is the reference point the blend must beat.
        popularity_only = {"collaborative": 0.0, "content": 0.0, "popularity": 1.0,
                           "personalization": 0.0, "business": 0.0}
        best_w, best_score = popularity_only, ndcg(popularity_only)
        default_score = ndcg(default)
        if default_score > best_score * 1.002:
            best_w, best_score = default, default_score
        grid = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)
        # popularity is pinned at 1.0 and the others are searched relative to it,
        # then the whole vector is renormalised to sum to 1.
        for collab in grid:
            for content in grid:
                for business in (0.0, 0.02, 0.05, 0.1):
                    raw = {"collaborative": collab, "content": content, "popularity": 1.0,
                           "personalization": default["personalization"], "business": business}
                    total = sum(raw.values())
                    cand = {k: round(v / total, 5) for k, v in raw.items()}
                    score = ndcg(cand)
                    # Require a meaningful margin before adopting a more complex
                    # blend, so we do not chase validation noise.
                    if score > best_score * 1.002:
                        best_w, best_score = cand, score
        self.logger.info("weights_fitted", ndcg_at_10=round(best_score, 5), **{k: round(v, 4) for k, v in best_w.items()})
        return best_w

    @staticmethod
    def _components(art: RecommenderArtifact, user_id: int) -> dict[str, np.ndarray]:
        seeds = art.user_history.get(int(user_id), [])[-25:]
        popularity = art.popularity_scores()
        collab = art.collaborative_scores(user_id)
        item_cf = art.item_cf_scores(seeds)
        collaborative = _minmax_local(0.6 * collab + 0.4 * item_cf) if seeds else collab
        return {
            "collaborative": _residualise_local(collaborative, popularity),
            "content": _residualise_local(art.content_scores(seeds), popularity),
            "popularity": popularity,
            "personalization": np.zeros_like(popularity),
            "business": art.business_scores(),
        }

    def _rank_for_user(self, art: RecommenderArtifact, user_id: int, components: dict[str, np.ndarray],
                       weights: dict[str, float], k: int = 20) -> list[int]:
        score = sum(weights.get(name, 0.0) * vec for name, vec in components.items())
        score = score * (0.35 + 0.65 * art.availability)
        history = set(art.user_history.get(int(user_id), []))
        out: list[int] = []
        for i in np.argsort(-score):
            if int(i) in history:
                continue
            out.append(int(art.item_ids[i]))
            if len(out) >= k:
                break
        return out

    def evaluate(self, artifact: RecommenderArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        art: RecommenderArtifact = self._eval_artifact
        _, test_df = self._split_test(prepared)
        if test_df.empty:
            return {"warning": "no post-cutoff interactions available for evaluation"}

        relevant = self._relevance_map(art, test_df)
        users = list(relevant)[:600]
        if len(users) < 10:
            return {"warning": "not enough evaluable users in the test window", "users": len(users)}
        relevant = {u: relevant[u] for u in users}
        weights = self._fitted_weights

        hybrid: dict[int, list[int]] = {}
        popularity_only: dict[int, list[int]] = {}
        random_baseline: dict[int, list[int]] = {}
        rng = np.random.default_rng(7)
        pop_weights = {"popularity": 1.0}
        for user_id in users:
            comps = self._components(art, user_id)
            hybrid[user_id] = self._rank_for_user(art, user_id, comps, weights)
            popularity_only[user_id] = self._rank_for_user(art, user_id, comps, pop_weights)
            random_baseline[user_id] = [int(x) for x in rng.choice(art.item_ids, size=20, replace=False)]

        report = ranking_report(hybrid, relevant, ks=(5, 10, 20))
        pop_report = ranking_report(popularity_only, relevant, ks=(5, 10, 20))
        rnd_report = ranking_report(random_baseline, relevant, ks=(10,))

        metrics: dict[str, Any] = {**report}
        # Every baseline is scored under identical history-exclusion rules.
        metrics["baseline_popularity_ndcg@10"] = pop_report["ndcg@10"]
        metrics["baseline_popularity_precision@10"] = pop_report["precision@10"]
        metrics["baseline_random_ndcg@10"] = rnd_report["ndcg@10"]
        metrics["lift_vs_popularity_ndcg@10"] = (
            round((report["ndcg@10"] - pop_report["ndcg@10"]) / pop_report["ndcg@10"], 4)
            if pop_report["ndcg@10"] > 0 else None
        )
        metrics["lift_vs_random_ndcg@10"] = (
            round((report["ndcg@10"] - rnd_report["ndcg@10"]) / rnd_report["ndcg@10"], 4)
            if rnd_report["ndcg@10"] > 0 else None
        )
        metrics["catalogue_coverage@10"] = coverage(hybrid, artifact.n_items, k=10)
        metrics["fitted_weights"] = weights
        metrics["n_items"] = artifact.n_items
        metrics["n_users"] = artifact.n_users
        metrics["cooccurrence_pairs"] = int((artifact.cooccurrence_score > 0).sum())
        metrics["content_explained_variance"] = getattr(artifact.embedder, "explained_variance", 0.0)
        return metrics

    def params(self, artifact: RecommenderArtifact) -> dict[str, Any]:
        return {
            "n_factors": int(artifact.item_factors.shape[1]),
            "n_neighbours": int(artifact.neighbour_idx.shape[1]),
            "n_cooccurrence": int(artifact.cooccurrence_idx.shape[1]),
            "trending_window_days": TRENDING_WINDOW_DAYS,
            "test_fraction": TEST_FRACTION,
            "event_confidence": EVENT_CONFIDENCE,
            "weights": {**artifact.weights},
            "embedding_backend": getattr(artifact.embedder, "backend_name", "unknown"),
            "embedding_dim": int(artifact.content_vectors.shape[1]) if artifact.content_vectors.size else 0,
        }

    def feature_names(self, artifact: RecommenderArtifact) -> list[str]:
        return ["collaborative", "content", "popularity", "personalization", "business"]

    def baseline_stats(self, artifact: RecommenderArtifact, prepared: dict[str, Any]) -> dict[str, Any]:
        return {
            "popularity_mean": round(float(artifact.popularity.mean()), 5),
            "popularity_p90": round(float(np.quantile(artifact.popularity, 0.9)), 5),
            "trending_mean": round(float(artifact.trending.mean()), 5),
            "conversion_mean": round(float(artifact.conversion_rate.mean()), 5),
            "in_stock_ratio": round(float(artifact.availability.mean()), 5),
        }

    def notes(self, artifact: RecommenderArtifact) -> str:
        return (
            "Hybrid recommender. Scores blend latent-factor CF, item-item CF, content similarity, "
            "popularity/trending, personalization affinity and business margin. Cold-start users fall "
            "back to popularity+trending+rating; cold-start products are reachable via content similarity."
        )


if __name__ == "__main__":
    raise SystemExit(run_cli(RecommendationPipeline, "Train the hybrid recommendation model."))
