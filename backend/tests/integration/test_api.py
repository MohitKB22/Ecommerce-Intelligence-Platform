"""API integration tests: routing, validation, auth, pagination and error shapes."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestHealth:
    def test_health_reports_dependencies(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ok", "degraded")
        assert body["checks"]["database"]["status"] == "ok"
        assert "cache" in body["checks"] and "models" in body["checks"]

    def test_liveness_is_cheap_and_always_ok(self, client):
        assert client.get("/api/v1/health/live").json()["status"] == "alive"

    def test_readiness_requires_the_database(self, client):
        body = client.get("/api/v1/health/ready").json()
        assert body["status"] == "ready"
        assert body["database"] == "ok"

    def test_request_id_header_is_returned(self, client):
        response = client.get("/api/v1/health/live")
        assert response.headers.get("X-Request-Id")
        assert response.headers.get("X-Response-Time-Ms")


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        headers = client.get("/api/v1/health/live").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "Referrer-Policy" in headers


class TestAuth:
    def test_login_succeeds_with_valid_credentials(self, client):
        response = client.post("/api/v1/auth/login",
                               json={"email": "admin@test.local", "password": "test-admin-password"})
        assert response.status_code == 200
        body = response.json()
        assert body["role"] == "admin" and body["access_token"]

    def test_login_rejects_a_bad_password(self, client):
        response = client.post("/api/v1/auth/login",
                               json={"email": "admin@test.local", "password": "wrong"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthenticated"

    def test_login_rejects_an_unknown_account(self, client):
        response = client.post("/api/v1/auth/login",
                               json={"email": "nobody@test.local", "password": "whatever"})
        assert response.status_code == 401

    def test_reserved_tld_addresses_are_accepted(self, client):
        """Seeded accounts use `.invalid`; deliverability validation must not block login."""
        response = client.post("/api/v1/auth/login",
                               json={"email": "user00001@example.invalid", "password": "demo-password"})
        assert response.status_code == 200

    def test_me_returns_the_token_subject(self, client, admin_headers):
        response = client.get("/api/v1/auth/me", headers=admin_headers)
        assert response.status_code == 200
        assert response.json()["role"] == "admin"

    def test_me_without_a_token_is_unauthenticated(self, client):
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_admin_endpoints_reject_anonymous_callers(self, client):
        for path in ("/api/v1/analytics/dashboard", "/api/v1/ml/registry", "/api/v1/ml/drift"):
            assert client.get(path).status_code == 401, path

    def test_admin_endpoints_reject_non_admin_tokens(self, client):
        token = client.post("/api/v1/auth/login",
                            json={"email": "user00001@example.invalid",
                                  "password": "demo-password"}).json()["access_token"]
        response = client.get("/api/v1/analytics/dashboard",
                              headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"

    def test_garbage_token_is_rejected(self, client):
        response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.token"})
        assert response.status_code == 401


class TestProducts:
    def test_list_is_paginated(self, client):
        response = client.get("/api/v1/products", params={"page": 1, "page_size": 5})
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) <= 5
        assert body["meta"]["page"] == 1 and body["meta"]["total"] > 0

    def test_pages_return_different_items(self, client):
        first = client.get("/api/v1/products", params={"page": 1, "page_size": 3}).json()
        second = client.get("/api/v1/products", params={"page": 2, "page_size": 3}).json()
        assert {i["id"] for i in first["items"]} != {i["id"] for i in second["items"]}

    def test_page_size_is_bounded(self, client):
        assert client.get("/api/v1/products", params={"page_size": 5000}).status_code == 422

    def test_filters_apply(self, client):
        body = client.get("/api/v1/products", params={"min_rating": 4, "in_stock": True}).json()
        for item in body["items"]:
            assert item["rating_avg"] >= 4
            assert item["inventory"] > 0

    def test_price_sort_is_ascending(self, client):
        items = client.get("/api/v1/products", params={"sort": "price_asc", "page_size": 10}).json()["items"]
        prices = [i["price"] for i in items]
        assert prices == sorted(prices)

    def test_detail_includes_relations(self, client, sample_product_id):
        body = client.get(f"/api/v1/products/{sample_product_id}").json()
        assert body["id"] == sample_product_id
        assert body["category"] is not None and body["brand"] is not None
        assert "effective_price" in body and "in_stock" in body

    def test_unknown_product_returns_a_clean_404(self, client):
        response = client.get("/api/v1/products/99999999")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert "request_id" in error
        assert "Traceback" not in response.text

    def test_invalid_id_is_a_validation_error(self, client):
        assert client.get("/api/v1/products/-5").status_code == 422

    def test_categories_and_brands(self, client):
        assert len(client.get("/api/v1/products/categories").json()) > 0
        assert len(client.get("/api/v1/products/brands").json()) > 0
        popular = client.get("/api/v1/products/categories/popular").json()
        assert all("product_count" in c for c in popular)

    def test_reviews_are_paginated(self, client, sample_product_id):
        body = client.get(f"/api/v1/products/{sample_product_id}/reviews",
                          params={"page_size": 3}).json()
        assert "items" in body and "meta" in body

    def test_review_creation_updates_the_aggregate(self, client, db, sample_product_id):
        from sqlalchemy import select

        from app.models import Review, User

        # Pick a customer who has not already reviewed this product.
        reviewed = {r for (r,) in db.execute(
            select(Review.user_id).where(Review.product_id == sample_product_id)).all()}
        user_id = db.execute(
            select(User.id).where(User.role == "user", User.id.notin_(reviewed or {-1})).limit(1)
        ).scalar_one()

        before = client.get(f"/api/v1/products/{sample_product_id}").json()["rating_count"]
        response = client.post(
            f"/api/v1/products/{sample_product_id}/reviews",
            json={"product_id": sample_product_id, "user_id": int(user_id), "rating": 5,
                  "title": "Excellent", "body": "The battery life is superb and setup was easy."},
        )
        assert response.status_code == 201, response.text
        after = client.get(f"/api/v1/products/{sample_product_id}").json()["rating_count"]
        assert after == before + 1

    def test_duplicate_review_is_a_conflict_not_a_crash(self, client, db):
        """A uniqueness violation must surface as 409, never as an unhandled 500."""
        from sqlalchemy import select

        from app.models import Review

        # Use any existing review rather than one for the first product, which
        # has no reviews on some generated datasets (the test used to skip).
        row = db.execute(select(Review.product_id, Review.user_id).order_by(Review.id).limit(1)).first()
        assert row is not None, "the seeded dataset should contain reviews"
        product_id, user_id = int(row[0]), int(row[1])
        response = client.post(
            f"/api/v1/products/{product_id}/reviews",
            json={"product_id": product_id, "user_id": user_id, "rating": 4,
                  "title": "Again", "body": "Trying to review the same product twice."},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "conflict"

    def test_review_validation_rejects_out_of_range_rating(self, client, sample_product_id, sample_user_id):
        response = client.post(
            f"/api/v1/products/{sample_product_id}/reviews",
            json={"product_id": sample_product_id, "user_id": sample_user_id, "rating": 9,
                  "title": "x", "body": "too many stars"},
        )
        assert response.status_code == 422


class TestSearch:
    def test_basic_search_returns_ranked_hits(self, client):
        body = client.get("/api/v1/search", params={"q": "wireless headphones"}).json()
        assert body["total"] >= 0
        assert body["strategy"] in ("hybrid", "semantic_fallback", "browse")
        scores = [h["score"] for h in body["hits"]]
        assert scores == sorted(scores, reverse=True)

    def test_every_hit_carries_signals_and_an_explanation(self, client):
        body = client.get("/api/v1/search", params={"q": "laptop"}).json()
        for hit in body["hits"][:3]:
            assert set(hit["signals"]) == {
                "text", "semantic", "popularity", "rating", "conversion", "personal", "availability"}
            assert hit["explanation"]

    def test_empty_query_browses_the_catalogue(self, client):
        body = client.get("/api/v1/search", params={"q": ""}).json()
        assert body["strategy"] == "browse"
        assert body["total"] > 0

    def test_filters_narrow_the_result_set(self, client):
        wide = client.get("/api/v1/search", params={"q": ""}).json()["total"]
        narrow = client.get("/api/v1/search", params={"q": "", "max_price": 50}).json()["total"]
        assert narrow <= wide

    def test_facets_are_returned(self, client):
        facets = client.get("/api/v1/search", params={"q": ""}).json()["facets"]
        assert "categories" in facets and "price" in facets and "availability" in facets

    def test_suggestions_require_a_prefix(self, client):
        assert client.get("/api/v1/search/suggestions", params={"q": "a"}).status_code == 200
        assert client.get("/api/v1/search/suggestions", params={"q": ""}).status_code == 422

    def test_popular_queries_endpoint(self, client):
        assert isinstance(client.get("/api/v1/search/popular").json(), list)

    def test_sort_options_are_validated(self, client):
        assert client.get("/api/v1/search", params={"q": "x", "sort": "bogus"}).status_code == 422


class TestEvents:
    def test_recording_an_event(self, client, sample_product_id, sample_user_id):
        response = client.post("/api/v1/events", json={
            "event_type": "product_view", "user_id": sample_user_id,
            "product_id": sample_product_id, "session_id": "test-session",
        })
        assert response.status_code == 201
        assert response.json()["event_type"] == "product_view"

    def test_unknown_event_type_is_rejected(self, client):
        response = client.post("/api/v1/events", json={"event_type": "teleport"})
        assert response.status_code == 422

    def test_unknown_product_is_rejected(self, client, sample_user_id):
        response = client.post("/api/v1/events", json={
            "event_type": "product_view", "user_id": sample_user_id, "product_id": 99999999})
        assert response.status_code == 404

    def test_batch_reports_partial_success(self, client, sample_product_id, sample_user_id):
        response = client.post("/api/v1/events/batch", json={"events": [
            {"event_type": "click", "user_id": sample_user_id, "product_id": sample_product_id},
            {"event_type": "not_a_real_event", "user_id": sample_user_id},
        ]})
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] == 1 and len(body["rejected"]) == 1

    def test_funnel_returns_counts(self, client):
        assert isinstance(client.get("/api/v1/events/funnel").json(), dict)


class TestUsers:
    def test_profile_exposes_personalization_signals(self, client, sample_user_id):
        body = client.get(f"/api/v1/users/{sample_user_id}/profile").json()
        assert body["user_id"] == sample_user_id
        assert "category_affinity" in body and "preference_vector" in body
        assert len(body["preference_vector"]) == 16

    def test_orders_are_paginated(self, client, sample_user_id):
        body = client.get(f"/api/v1/users/{sample_user_id}/orders").json()
        assert "items" in body and "meta" in body

    def test_unknown_user_returns_404(self, client):
        assert client.get("/api/v1/users/99999999/profile").status_code == 404

    def test_events_endpoint(self, client, sample_user_id):
        assert isinstance(client.get(f"/api/v1/users/{sample_user_id}/events").json(), list)


class TestOpenAPI:
    def test_schema_is_served(self, client):
        spec = client.get("/openapi.json").json()
        assert spec["info"]["title"]
        assert len(spec["paths"]) > 40

    def test_docs_render(self, client):
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200

    def test_root_redirects_to_docs(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (307, 302)
