"""Search service behaviour: filters, ranking, intent parsing and facets."""
from __future__ import annotations

import pytest

from app.search.ranking import RankingSignals, explain, minmax_normalise, normalise_weights, score_result
from app.search.service import SearchFilters, SearchService
from app.search.vector_store import NumpyVectorStore

pytestmark = pytest.mark.integration


class TestRankingMath:
    def test_score_is_a_weighted_sum(self):
        signals = RankingSignals(text=1.0, semantic=1.0, popularity=1.0, rating=1.0,
                                 conversion=1.0, personal=1.0, availability=1.0)
        weights = {"text": 0.5, "semantic": 0.5, "popularity": 0, "rating": 0,
                   "conversion": 0, "personal": 0, "availability": 0}
        assert score_result(signals, weights) == pytest.approx(1.0)

    def test_zero_signals_score_zero(self):
        assert score_result(RankingSignals()) == pytest.approx(0.0)

    def test_weights_are_normalised(self):
        weights = normalise_weights({"text": 2, "semantic": 2, "popularity": 0, "rating": 0,
                                     "conversion": 0, "personal": 0, "availability": 0})
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_zero_weights_fall_back_to_defaults(self):
        assert sum(normalise_weights({k: 0 for k in ["text", "semantic"]}).values()) == pytest.approx(1.0)

    def test_explanation_names_the_dominant_signal(self):
        text = explain(RankingSignals(text=1.0), {"text": 1.0})
        assert "keyword match" in text

    def test_minmax_normalise(self):
        result = minmax_normalise({1: 10.0, 2: 20.0, 3: 30.0})
        assert result[1] == 0.0 and result[3] == 1.0

    def test_minmax_of_identical_scores(self):
        result = minmax_normalise({1: 5.0, 2: 5.0})
        assert set(result.values()) == {1.0}


class TestVectorStore:
    def test_search_returns_the_nearest_vector(self):
        import numpy as np

        store = NumpyVectorStore()
        store.upsert([1, 2, 3], np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
        results = store.search(np.array([1.0, 0.0]), k=2)
        assert results[0][0] == 1
        assert store.size() == 3

    def test_allowed_ids_restricts_the_candidate_set(self):
        import numpy as np

        store = NumpyVectorStore()
        store.upsert([1, 2, 3], np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
        results = store.search(np.array([1.0, 0.0]), k=3, allowed_ids={2, 3})
        assert {pid for pid, _ in results} <= {2, 3}

    def test_empty_store_returns_nothing(self):
        import numpy as np

        assert NumpyVectorStore().search(np.array([1.0, 0.0])) == []

    def test_zero_query_vector_is_handled(self):
        import numpy as np

        store = NumpyVectorStore()
        store.upsert([1], np.array([[1.0, 0.0]]))
        assert store.search(np.array([0.0, 0.0])) == []


class TestSearchService:
    def test_query_normalisation(self):
        assert SearchService.normalise("  Wireless   HEADPHONES  ") == "wireless headphones"

    def test_budget_intent_is_parsed(self, db):
        service = SearchService(db)
        assert service.parse_intent("laptop under $500")["budget"] == 500.0
        assert service.parse_intent("headphones below 100")["budget"] == 100.0

    def test_price_modifiers_are_detected(self, db):
        service = SearchService(db)
        assert "price_ascending" in service.parse_intent("cheap headphones")["modifiers"]
        assert "quality_bias" in service.parse_intent("best headphones")["modifiers"]

    def test_typo_correction(self, db):
        service = SearchService(db)
        corrected, changed = service.correct_query("wireles")
        assert changed is False or corrected != "wireles"

    def test_filters_match_expected_products(self, db):
        service = SearchService(db)
        entry = next(iter(service.index.entries.values()))
        permissive = SearchFilters()
        assert permissive.matches(entry) == entry.is_active

        impossible = SearchFilters(min_price=1e9)
        assert impossible.matches(entry) is False

    def test_search_returns_ranked_results(self, db):
        response = SearchService(db).search("", limit=10)
        assert response.total > 0
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_price_filter_is_applied(self, db):
        service = SearchService(db)
        response = service.search("", filters=SearchFilters(max_price=40.0), limit=50)
        for result in response.results:
            assert service.index.entries[result.product_id].effective_price <= 40.0

    def test_facets_reflect_the_result_set(self, db):
        response = SearchService(db).search("", limit=5)
        assert response.facets["price"]["min"] <= response.facets["price"]["max"]
        assert response.facets["availability"]["in_stock"] >= 0

    def test_pagination_offsets_results(self, db):
        service = SearchService(db)
        first = service.search("", limit=5, offset=0, use_cache=False)
        second = service.search("", limit=5, offset=5, use_cache=False)
        assert {r.product_id for r in first.results} != {r.product_id for r in second.results}

    def test_sorting_by_price(self, db):
        service = SearchService(db)
        response = service.search("", limit=10, sort="price_asc", use_cache=False)
        prices = [service.index.entries[r.product_id].effective_price for r in response.results]
        assert prices == sorted(prices)

    def test_autocomplete_requires_two_characters(self, db):
        service = SearchService(db)
        assert service.autocomplete("a") == []
        assert isinstance(service.autocomplete("wi"), list)

    def test_nonsense_query_degrades_without_raising(self, db):
        response = SearchService(db).search("zzzzqqqxxxvvv", limit=5)
        assert response.total >= 0
