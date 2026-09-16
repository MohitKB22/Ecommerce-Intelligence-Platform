"""Unit tests for the ML building blocks: metrics, BM25, embeddings, registry."""
from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.metrics import (
    average_precision_at_k, classification_report_dict, coverage, forecast_metrics, map_at_k, ndcg_at_k,
    population_stability_index, precision_at_k, ranking_report, recall_at_k, regression_metrics,
)
from ml.features.bm25 import BM25Index, _damerau_levenshtein
from ml.features.embeddings import TfidfSvdEmbedder, normalise_text, tokenize
from ml.inference.artifacts import _minmax, _residualise, _topk


class TestRankingMetrics:
    def test_precision_and_recall(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 5, 9}
        assert precision_at_k(recommended, relevant, 5) == pytest.approx(0.4)
        assert recall_at_k(recommended, relevant, 5) == pytest.approx(2 / 3)

    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_ndcg_rewards_higher_positions(self):
        early = ndcg_at_k([1, 9, 9, 9], {1}, 4)
        late = ndcg_at_k([9, 9, 9, 1], {1}, 4)
        assert early > late

    def test_empty_inputs_do_not_raise(self):
        assert precision_at_k([], {1}, 5) == 0.0
        assert recall_at_k([1], set(), 5) == 0.0
        assert ndcg_at_k([], set(), 5) == 0.0
        assert map_at_k({}, {}, 5) == 0.0

    def test_average_precision_ordering(self):
        assert average_precision_at_k([1, 0, 0], {1}, 3) > average_precision_at_k([0, 0, 1], {1}, 3)

    def test_ranking_report_has_every_cutoff(self):
        report = ranking_report({"u": [1, 2, 3]}, {"u": {2}}, ks=(1, 3))
        for key in ("precision@1", "recall@1", "ndcg@1", "map@1",
                    "precision@3", "recall@3", "ndcg@3", "map@3"):
            assert key in report

    def test_coverage(self):
        assert coverage({"a": [1, 2], "b": [2, 3]}, catalogue_size=10, k=2) == pytest.approx(0.3)


class TestRegressionMetrics:
    def test_perfect_forecast(self):
        result = forecast_metrics([1, 2, 3], [1, 2, 3])
        assert result["mae"] == 0.0 and result["rmse"] == 0.0 and result["mape"] == 0.0

    def test_error_magnitudes(self):
        result = forecast_metrics([10, 10], [12, 8])
        assert result["mae"] == pytest.approx(2.0)
        assert result["rmse"] == pytest.approx(2.0)
        assert result["bias"] == pytest.approx(0.0)

    def test_zero_actuals_do_not_divide_by_zero(self):
        result = forecast_metrics([0, 0, 5], [1, 1, 5])
        assert np.isfinite(result["mape"])
        assert np.isfinite(result["smape"])

    def test_empty_input(self):
        assert forecast_metrics([], [])["support"] == 0

    def test_r2_present_for_regression(self):
        assert "r2" in regression_metrics([1, 2, 3, 4], [1.1, 2.1, 2.9, 4.2])


class TestClassificationMetrics:
    def test_perfect_classifier(self):
        report = classification_report_dict(["a", "b", "a"], ["a", "b", "a"], labels=["a", "b"])
        assert report["accuracy"] == 1.0 and report["f1_macro"] == 1.0

    def test_confusion_matrix_shape(self):
        report = classification_report_dict(["a", "b", "c"], ["a", "b", "a"], labels=["a", "b", "c"])
        assert len(report["confusion_matrix"]) == 3
        assert report["support"] == 3

    def test_empty_input(self):
        assert classification_report_dict([], [])["accuracy"] == 0.0


class TestDriftStatistic:
    def test_identical_distributions_have_no_drift(self):
        values = list(np.random.default_rng(0).normal(0, 1, 500))
        assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 500)
        shifted = rng.normal(3, 1, 500)
        assert population_stability_index(base, shifted) > 0.2

    def test_tiny_samples_return_zero(self):
        assert population_stability_index([1], [2]) == 0.0


class TestBM25:
    @pytest.fixture()
    def index(self) -> BM25Index:
        return BM25Index.build([
            (1, "wireless bluetooth headphones with deep bass for the gym"),
            (2, "yoga mat non slip fitness home workout"),
            (3, "gaming laptop fast processor sixteen gigabytes"),
            (4, "wireless earbuds sport waterproof running"),
        ])

    def test_matching_documents_are_scored(self, index):
        scores = index.score("wireless headphones")
        assert 1 in scores
        assert scores[1] > scores.get(4, 0)

    def test_non_matching_query_returns_nothing(self, index):
        assert index.score("submarine periscope") == {}

    def test_empty_query_returns_nothing(self, index):
        assert index.score("") == {}

    def test_limit_is_respected(self, index):
        assert len(index.score("wireless", limit=1)) <= 1

    def test_typo_correction_finds_the_right_term(self, index):
        assert "headphones" in index.closest_terms("headphnes")
        assert "wireless" in index.closest_terms("wirless")

    def test_known_term_returns_itself(self, index):
        assert index.closest_terms("laptop") == ["laptop"]

    def test_vocabulary_membership(self, index):
        assert index.term_exists("yoga")
        assert not index.term_exists("helicopter")

    def test_empty_corpus_is_safe(self):
        empty = BM25Index.build([])
        assert empty.size == 0
        assert empty.score("anything") == {}


class TestEditDistance:
    def test_distances(self):
        assert _damerau_levenshtein("cat", "cat") == 0
        assert _damerau_levenshtein("cat", "cut") == 1
        assert _damerau_levenshtein("cat", "act") == 1  # transposition

    def test_bounded_search_short_circuits(self):
        assert _damerau_levenshtein("abc", "xyzxyzxyz", max_distance=2) > 2


class TestEmbeddings:
    @pytest.fixture()
    def corpus(self) -> list[str]:
        return [
            "wireless bluetooth headphones deep bass gym audio",
            "wireless sport earbuds waterproof running audio",
            "gaming laptop fast processor computer",
            "yoga mat fitness home workout",
            "mechanical keyboard computer peripheral",
        ]

    def test_output_shape_and_normalisation(self, corpus):
        embedder = TfidfSvdEmbedder()
        vectors = embedder.fit_transform(corpus)
        assert vectors.shape[0] == len(corpus)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)

    def test_semantically_close_documents_are_closer(self, corpus):
        embedder = TfidfSvdEmbedder()
        vectors = embedder.fit_transform(corpus)
        audio_similarity = float(vectors[0] @ vectors[1])
        cross_domain = float(vectors[0] @ vectors[3])
        assert audio_similarity > cross_domain

    def test_is_deterministic(self, corpus):
        a = TfidfSvdEmbedder().fit_transform(corpus)
        b = TfidfSvdEmbedder().fit_transform(corpus)
        assert np.allclose(a, b)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            TfidfSvdEmbedder().transform(["x"])

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            TfidfSvdEmbedder().fit([])

    def test_tokenizer_drops_stopwords_and_punctuation(self):
        assert tokenize("The quick, brown fox!") == ["quick", "brown", "fox"]
        assert normalise_text("Hello,   World!!") == "hello world"


class TestArtifactHelpers:
    def test_minmax_scales_to_unit_range(self):
        scaled = _minmax(np.array([2.0, 4.0, 6.0]))
        assert scaled.min() == 0.0 and scaled.max() == 1.0

    def test_minmax_of_constant_is_zero(self):
        assert np.all(_minmax(np.array([5.0, 5.0])) == 0.0)

    def test_topk_returns_descending_indices(self):
        assert _topk(np.array([0.1, 0.9, 0.5]), 2) == [1, 2]

    def test_topk_honours_exclusions(self):
        assert 1 not in _topk(np.array([0.1, 0.9, 0.5, 0.7]), 2, exclude={1})

    def test_residualise_removes_the_base_direction(self):
        base = np.array([1.0, 2.0, 3.0, 4.0])
        duplicate = base * 2.0
        residual = _residualise(duplicate, base)
        # A component that is purely a scaled copy of the base carries no extra signal.
        assert float(np.std(residual)) < 1e-6

    def test_residualise_preserves_orthogonal_signal(self):
        base = np.array([1.0, 2.0, 3.0, 4.0])
        orthogonal = np.array([1.0, 0.0, 1.0, 0.0])
        assert float(np.std(_residualise(orthogonal, base))) > 0.0


class TestModelRegistry:
    def test_save_and_load_round_trip(self, tmp_path):
        from ml.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)
        card = registry.save("demo", {"weights": [1, 2, 3]}, algorithm="test",
                             metrics={"accuracy": 0.9}, training_rows=100)
        artifact, loaded_card = registry.load("demo")
        assert artifact == {"weights": [1, 2, 3]}
        assert loaded_card.version == card.version
        assert loaded_card.metrics["accuracy"] == 0.9

    def test_latest_version_wins(self, tmp_path):
        from ml.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)
        registry.save("m", {"v": 1}, version="v1")
        registry.save("m", {"v": 2}, version="v2")
        artifact, card = registry.load("m")
        assert card.version == "v2" and artifact == {"v": 2}

    def test_missing_model_raises(self, tmp_path):
        from ml.registry import ModelRegistry

        with pytest.raises(FileNotFoundError):
            ModelRegistry(tmp_path).load("nope")

    def test_exists_reports_accurately(self, tmp_path):
        from ml.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)
        assert not registry.exists("m")
        registry.save("m", {"a": 1})
        assert registry.exists("m")

    def test_promote_archives_the_previous_production_version(self, tmp_path):
        from ml.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)
        registry.save("m", {"v": 1}, version="v1")
        registry.save("m", {"v": 2}, version="v2", stage="development")
        registry.promote("m", "v2", stage="production")
        history = {card.version: card.stage for card in registry.history("m")}
        assert history["v2"] == "production"
        assert history["v1"] == "archived"
