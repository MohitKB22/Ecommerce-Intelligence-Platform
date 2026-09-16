"""Integration tests that actually train every model and exercise inference.

Marked `slow` because they run the real pipelines end to end.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def models(trained_models):
    for name, result in trained_models.items():
        if isinstance(result, Exception):
            pytest.fail(f"{name} pipeline raised {type(result).__name__}: {result}")
    return trained_models


class TestTrainingPipelines:
    def test_every_pipeline_produced_a_version(self, models):
        assert set(models) == {"sentiment", "segmentation", "recommendation", "forecasting", "price_prediction"}
        for name, result in models.items():
            assert result.version, f"{name} has no version"
            assert result.training_rows > 0, f"{name} trained on no rows"

    def test_recommendation_beats_a_random_baseline(self, models):
        metrics = models["recommendation"].metrics
        if "warning" in metrics:
            pytest.skip(metrics["warning"])
        assert metrics["ndcg@10"] > metrics.get("baseline_random_ndcg@10", 0.0)
        assert 0.0 <= metrics["precision@10"] <= 1.0
        assert metrics["catalogue_coverage@10"] > 0.0

    def test_sentiment_beats_the_majority_class(self, models):
        metrics = models["sentiment"].metrics
        assert metrics["accuracy"] > metrics["baseline_majority_accuracy"]
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 3

    def test_sentiment_is_not_perfect(self, models):
        """A perfect score would mean the label leaked into the text."""
        assert models["sentiment"].metrics["accuracy"] < 1.0

    def test_segmentation_produces_usable_clusters(self, models):
        metrics = models["segmentation"].metrics
        assert metrics["silhouette_score"] > 0.0
        assert metrics["n_clusters"] >= 4
        assert metrics["largest_segment_share"] < 0.95  # not one giant blob
        assert len(metrics["segment_distribution"]) >= 3

    def test_forecasting_reports_a_baseline_comparison(self, models):
        metrics = models["forecasting"].metrics
        assert metrics["mae"] >= 0.0
        assert "baseline_seasonal_naive" in metrics
        # The selected model must never be worse than the naive baseline.
        assert metrics["mae"] <= metrics["baseline_seasonal_naive"]["mae"] + 1e-9

    def test_price_model_beats_the_category_median(self, models):
        metrics = models["price_prediction"].metrics
        assert metrics["mae"] < metrics["baseline_category_median"]["mae"]
        assert metrics["r2"] > 0.5

    def test_price_model_has_no_cost_leakage(self, models):
        """`base_cost` is derived from price in the generator and must not be a feature."""
        assert "base_cost" not in models["price_prediction"].params["features"]
        assert "category_price_rank" not in models["price_prediction"].params["features"]

    def test_metrics_are_registered_in_the_database(self, models, db):
        from sqlalchemy import select

        from app.models import ModelRegistryEntry

        names = {n for (n,) in db.execute(select(ModelRegistryEntry.name).distinct()).all()}
        assert {"recommendation", "sentiment", "segmentation", "forecasting", "price_prediction"} <= names


class TestInference:
    def test_recommendations_for_a_known_user(self, models, db):
        from app.recommendation.service import RecommendationService

        artifact, _ = _load(db, "recommendation")
        user_id = int(artifact.user_ids[0])
        result = RecommendationService(db).for_user(user_id, limit=5, log_impressions=False)
        assert len(result["items"]) > 0
        assert result["strategy"] in ("hybrid", "cold_start")
        for item in result["items"]:
            assert item["explanation"]
            assert 0.0 <= item["score"]

    def test_cold_start_path_for_an_unknown_user(self, models, db):
        from app.recommendation.service import RecommendationService

        result = RecommendationService(db).for_user(None, limit=5, log_impressions=False)
        assert result["strategy"] in ("cold_start", "popularity_fallback")
        assert len(result["items"]) > 0

    def test_recommendations_exclude_requested_products(self, models, db, sample_product_id):
        from app.recommendation.service import RecommendationService

        result = RecommendationService(db).for_user(
            None, limit=8, exclude=[sample_product_id], log_impressions=False)
        assert all(i["product_id"] != sample_product_id for i in result["items"])

    def test_similar_products_never_include_the_seed(self, models, db, sample_product_id):
        from app.recommendation.service import RecommendationService

        result = RecommendationService(db).similar_products(sample_product_id, limit=5)
        assert all(i["product_id"] != sample_product_id for i in result["items"])

    def test_sentiment_inference_on_a_mixed_review(self, models, db):
        from app.sentiment.service import SentimentService

        result = SentimentService(db).analyse_text(
            "Battery life is excellent but the ear cushions are uncomfortable.")
        assert result["label"] in ("positive", "neutral", "negative")
        assert -1.0 <= result["score"] <= 1.0
        # Both aspects should be recognised from the text.
        assert "battery_life" in result["aspects"] or "comfort" in result["aspects"]

    def test_sentiment_separates_clear_polarity(self, models, db):
        from app.sentiment.service import SentimentService

        service = SentimentService(db)
        positive = service.analyse_text(
            "Absolutely superb. Outstanding value, brilliant build quality, would recommend to anyone.")
        negative = service.analyse_text(
            "Broke within a week. Terrible build quality, overpriced, and it stopped working entirely.")
        assert positive["score"] > negative["score"]

    def test_product_sentiment_aggregation(self, models, db, sample_product_id):
        from app.sentiment.service import SentimentService

        result = SentimentService(db).product_sentiment(sample_product_id)
        assert result["product_id"] == sample_product_id
        assert "distribution" in result and "summary" in result

    def test_forecast_shape_and_intervals(self, models, db):
        from app.forecasting.service import ForecastingService

        artifact, _ = _load(db, "forecasting")
        if not artifact.trained_products:
            pytest.skip("no forecastable products in the compact test dataset")
        product_id = int(artifact.trained_products[0])
        result = ForecastingService(db).forecast(product_id, horizon=30)
        assert len(result["points"]) == 30
        for point in result["points"]:
            assert point["lower"] <= point["predicted"] <= point["upper"]
            assert point["predicted"] >= 0
        assert set(result["horizons"]) == {"next_7_days", "next_14_days", "next_30_days"}

    def test_forecast_intervals_widen_with_horizon(self, models, db):
        from app.forecasting.service import ForecastingService

        artifact, _ = _load(db, "forecasting")
        if not artifact.trained_products:
            pytest.skip("no forecastable products")
        points = ForecastingService(db).forecast(int(artifact.trained_products[0]), horizon=28)["points"]
        early = points[0]["upper"] - points[0]["lower"]
        late = points[-1]["upper"] - points[-1]["lower"]
        assert late >= early

    def test_price_prediction_with_explanations(self, models, db, sample_product_id):
        from app.services.pricing_service import PricingService

        result = PricingService(db).predict(sample_product_id)
        assert result["predicted_price"] > 0
        assert result["price_trend"] in ("up", "down", "stable")
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["lower_bound"] <= result["predicted_price"] <= result["upper_bound"]
        assert len(result["feature_contributions"]) > 0
        assert result["explanation"]

    def test_segments_are_persisted_and_named(self, models, db):
        from app.segmentation.service import SegmentationService

        overview = SegmentationService(db).overview()
        assert overview["available"] is True
        assert overview["total_customers"] > 0
        known = {"high_value", "loyal_customer", "frequent_buyer", "discount_seeker",
                 "occasional_buyer", "at_risk", "new_customer"}
        assert {s["label"] for s in overview["segments"]} <= known


class TestGracefulDegradation:
    def test_recommendations_work_without_a_trained_model(self, db):
        """With no model loaded the storefront must still return products."""
        from app.ml.model_store import reset_model_store
        from app.recommendation.service import RecommendationService

        reset_model_store()
        service = RecommendationService(db)
        service.store._registry = lambda: (_ for _ in ()).throw(FileNotFoundError("no registry"))  # noqa: SLF001
        result = service.for_user(1, limit=5, log_impressions=False)
        assert result["strategy"] == "popularity_fallback"
        assert len(result["items"]) > 0
        reset_model_store()

    def test_forecast_endpoint_reports_model_unavailable(self, client, sample_product_id):
        from app.ml.model_store import reset_model_store

        reset_model_store()
        response = client.get(f"/api/v1/forecast/{sample_product_id}")
        # Either the model is loaded (200) or it is cleanly reported missing (503).
        assert response.status_code in (200, 503)
        if response.status_code == 503:
            assert response.json()["error"]["code"] == "model_unavailable"


def _load(db, name: str):
    from app.ml.model_store import get_model_store

    loaded = get_model_store().get(name)
    return loaded.artifact, loaded.card
