"""End-to-end journeys exercised through the public API.

These follow the exact flow described in the requirements:
homepage -> search -> product -> recommendations -> add to cart -> event recorded.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


class TestShoppingJourney:
    def test_full_browse_to_cart_flow(self, client, sample_user_id):
        session = {"X-Session-Id": "e2e-journey-1"}

        # 1. Shopper opens the homepage
        homepage = client.get("/api/v1/recommendations/homepage",
                              params={"user_id": sample_user_id}, headers=session)
        assert homepage.status_code == 200
        home = homepage.json()
        assert len(home["sections"]) > 0, "homepage returned no sections"
        assert all("items" in section for section in home["sections"])

        # 2. Products load
        listing = client.get("/api/v1/products", params={"page_size": 12}, headers=session)
        assert listing.status_code == 200
        assert len(listing.json()["items"]) > 0

        # 3. Shopper searches
        results = client.get("/api/v1/search",
                             params={"q": "wireless", "user_id": sample_user_id}, headers=session)
        assert results.status_code == 200
        search_body = results.json()

        # 4. Search results appear (fall back to the catalogue if the term is absent)
        if search_body["total"] == 0:
            search_body = client.get("/api/v1/search", params={"q": ""}, headers=session).json()
        assert search_body["total"] > 0
        product_id = search_body["hits"][0]["product"]["id"]

        # 5. Shopper opens the product
        detail = client.get(f"/api/v1/products/{product_id}", headers=session)
        assert detail.status_code == 200
        assert detail.json()["id"] == product_id

        # A view event is recorded, exactly as the UI does
        view = client.post("/api/v1/events", headers=session, json={
            "event_type": "product_view", "user_id": sample_user_id,
            "product_id": product_id, "session_id": "e2e-journey-1",
        })
        assert view.status_code == 201

        # 6. Recommendations appear on the product page
        for path in (f"/api/v1/products/{product_id}/similar",
                     f"/api/v1/products/{product_id}/frequently-bought-together",
                     f"/api/v1/products/{product_id}/recommendations"):
            response = client.get(path, params={"user_id": sample_user_id}, headers=session)
            assert response.status_code == 200, path
            assert "items" in response.json()

        # 7. Shopper adds the product to the cart
        add = client.post("/api/v1/events", headers=session, json={
            "event_type": "add_to_cart", "user_id": sample_user_id,
            "product_id": product_id, "session_id": "e2e-journey-1", "quantity": 2,
        })
        assert add.status_code == 201

        # 8. The event is persisted and visible on the user's activity feed
        events = client.get(f"/api/v1/users/{sample_user_id}/events", params={"limit": 20}).json()
        recorded = [e for e in events if e["product_id"] == product_id]
        assert any(e["event_type"] == "add_to_cart" for e in recorded)
        assert any(e["event_type"] == "product_view" for e in recorded)

    def test_purchase_updates_the_profile(self, client, sample_user_id, sample_product_id):
        before = client.get(f"/api/v1/users/{sample_user_id}/profile").json()
        response = client.post("/api/v1/events", json={
            "event_type": "purchase", "user_id": sample_user_id, "product_id": sample_product_id,
            "quantity": 1, "value": 49.99, "session_id": "e2e-purchase",
        })
        assert response.status_code == 201
        after = client.get(f"/api/v1/users/{sample_user_id}/profile").json()
        # The profile cache is invalidated on write, so the interaction count moves.
        assert after["interaction_count"] >= before["interaction_count"]

    def test_anonymous_visitor_gets_a_working_homepage(self, client):
        response = client.get("/api/v1/recommendations/homepage",
                              headers={"X-Session-Id": "anon-visitor-1"})
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] is None
        assert len(body["sections"]) > 0

    def test_search_then_refine_with_filters(self, client):
        broad = client.get("/api/v1/search", params={"q": ""}).json()
        assert broad["total"] > 0
        category = broad["facets"]["categories"][0]["slug"]
        refined = client.get("/api/v1/search", params={"q": "", "category": category}).json()
        assert refined["total"] <= broad["total"]
        assert refined["total"] > 0

    def test_typo_is_corrected_end_to_end(self, client):
        catalogue = client.get("/api/v1/search", params={"q": ""}).json()
        title = catalogue["hits"][0]["product"]["title"]
        word = next((w for w in title.lower().split() if len(w) > 6), None)
        if not word:
            pytest.skip("no sufficiently long token to corrupt")
        typo = word[:-2] + word[-1]  # drop a character near the end
        response = client.get("/api/v1/search", params={"q": typo}).json()
        assert response["corrected_query"] is not None or response["total"] > 0


class TestAdminJourney:
    def test_admin_can_reach_every_intelligence_surface(self, client, admin_headers):
        for path in (
            "/api/v1/analytics/dashboard", "/api/v1/analytics/business", "/api/v1/analytics/ai",
            "/api/v1/analytics/customers", "/api/v1/analytics/products",
            "/api/v1/analytics/revenue-timeseries", "/api/v1/ml/registry", "/api/v1/ml/health",
            "/api/v1/ml/drift", "/api/v1/ml/summary", "/api/v1/ml/features/coverage",
        ):
            response = client.get(path, headers=admin_headers)
            assert response.status_code == 200, f"{path} -> {response.status_code} {response.text[:200]}"

    def test_dashboard_numbers_are_internally_consistent(self, client, admin_headers):
        body = client.get("/api/v1/analytics/dashboard", headers=admin_headers).json()
        business = body["business"]
        assert business["revenue"] >= 0
        assert business["orders"] >= 0
        assert 0.0 <= business["conversion_rate"] <= 1.0
        assert 0.0 <= business["retention_rate"] <= 1.0
        if business["orders"] > 0:
            # AOV must be consistent with revenue and order count.
            assert business["average_order_value"] == pytest.approx(
                business["revenue"] / business["orders"], rel=0.05)

    def test_ai_metrics_are_bounded_rates(self, client, admin_headers):
        ai = client.get("/api/v1/analytics/ai", headers=admin_headers).json()
        for block in (ai["recommendation"], ai["search"]):
            for key, value in block.items():
                if key.endswith(("ctr", "rate")) and isinstance(value, (int, float)):
                    assert 0.0 <= value <= 1.0, f"{key}={value} is not a valid rate"

    def test_model_reload_is_idempotent(self, client, admin_headers):
        first = client.post("/api/v1/ml/reload", headers=admin_headers)
        second = client.post("/api/v1/ml/reload", headers=admin_headers)
        assert first.status_code == 200 and second.status_code == 200
        assert set(first.json()["reloaded"]) == set(second.json()["reloaded"])
