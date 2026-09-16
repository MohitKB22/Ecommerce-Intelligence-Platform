"""Unit tests for configuration, cache, security and error handling."""
from __future__ import annotations

import time

import pytest

from app.core.cache import CacheService, InMemoryCache
from app.core.config import Settings
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.security import (
    create_access_token, decode_token, hash_password, verify_password,
)


class TestPasswordHashing:
    def test_round_trip(self):
        hashed = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", hashed)
        assert not verify_password("wrong password", hashed)

    def test_salt_makes_hashes_unique(self):
        assert hash_password("same") != hash_password("same")

    def test_rejects_empty_password(self):
        with pytest.raises(ValueError):
            hash_password("")

    def test_malformed_hash_is_rejected_not_crashed(self):
        assert verify_password("anything", "not-a-valid-hash") is False
        assert verify_password("anything", "") is False


class TestTokens:
    def test_round_trip_preserves_claims(self):
        token = create_access_token("42", role="admin", extra={"email": "a@b.co"})
        claims = decode_token(token)
        assert claims["sub"] == "42"
        assert claims["role"] == "admin"
        assert claims["email"] == "a@b.co"

    def test_expired_token_is_rejected(self):
        from app.core.errors import AuthenticationError

        token = create_access_token("1", expires_minutes=-1)
        with pytest.raises(AuthenticationError):
            decode_token(token)

    def test_tampered_token_is_rejected(self):
        from app.core.errors import AuthenticationError

        token = create_access_token("1")
        with pytest.raises(AuthenticationError):
            decode_token(token[:-3] + "xyz")


class TestInMemoryCache:
    def test_set_get_delete(self):
        cache = InMemoryCache()
        cache.set("k", "v", ttl=10)
        assert cache.get("k") == "v"
        cache.delete("k")
        assert cache.get("k") is None

    def test_ttl_expiry(self):
        cache = InMemoryCache()
        cache.set("k", "v", ttl=1)
        assert cache.get("k") == "v"
        time.sleep(1.05)
        assert cache.get("k") is None

    def test_lru_eviction_respects_max_entries(self):
        cache = InMemoryCache(max_entries=3)
        for i in range(5):
            cache.set(f"k{i}", "v", ttl=60)
        assert len(cache._data) <= 3  # noqa: SLF001 - asserting the eviction invariant

    def test_delete_prefix(self):
        cache = InMemoryCache()
        cache.set("a:1", "v", 60)
        cache.set("a:2", "v", 60)
        cache.set("b:1", "v", 60)
        assert cache.delete_prefix("a:") == 2
        assert cache.get("b:1") == "v"


class TestCacheService:
    def test_json_round_trip(self):
        service = CacheService(InMemoryCache(), enabled=True)
        service.set_json("key", {"a": [1, 2, 3]}, 30)
        assert service.get_json("key") == {"a": [1, 2, 3]}

    def test_get_or_set_calls_producer_once(self):
        service = CacheService(InMemoryCache(), enabled=True)
        calls = []

        def producer():
            calls.append(1)
            return {"value": 7}

        assert service.get_or_set("k", producer, 60) == {"value": 7}
        assert service.get_or_set("k", producer, 60) == {"value": 7}
        assert len(calls) == 1

    def test_disabled_cache_always_misses(self):
        service = CacheService(InMemoryCache(), enabled=False)
        service.set_json("k", {"a": 1}, 60)
        assert service.get_json("k") is None

    def test_rate_limit_blocks_after_threshold(self):
        service = CacheService(InMemoryCache(), enabled=True)
        allowed = [service.rate_limit("ip", limit=3, window_seconds=60)[0] for _ in range(5)]
        assert allowed == [True, True, True, False, False]

    def test_rate_limit_isolates_identities(self):
        service = CacheService(InMemoryCache(), enabled=True)
        for _ in range(3):
            service.rate_limit("a", 3, 60)
        assert service.rate_limit("b", 3, 60)[0] is True

    def test_stats_track_hit_rate(self):
        service = CacheService(InMemoryCache(), enabled=True)
        service.set_json("k", 1, 60)
        service.get_json("k")
        service.get_json("missing")
        stats = service.stats()
        assert stats["hits"] == 1 and stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestErrors:
    def test_payload_shape(self):
        payload = NotFoundError("gone").to_payload()
        assert payload["error"]["code"] == "not_found"
        assert payload["error"]["message"] == "gone"
        assert "request_id" in payload["error"]

    def test_status_codes(self):
        assert NotFoundError().status_code == 404
        assert ValidationError().status_code == 422
        assert AppError().status_code == 500

    def test_details_are_included_when_present(self):
        payload = ValidationError("bad", details=[{"field": "x"}]).to_payload()
        assert payload["error"]["details"] == [{"field": "x"}]


class TestSettings:
    def test_cors_origins_parsed_from_csv(self):
        settings = Settings(CORS_ORIGINS="http://a.com, http://b.com")
        assert settings.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_parsed_from_json_array(self):
        settings = Settings(CORS_ORIGINS='["http://a.com","http://b.com"]')
        assert settings.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_empty_is_empty_list(self):
        assert Settings(CORS_ORIGINS="").cors_origins == []

    def test_dotenv_style_csv_does_not_raise(self):
        """Regression: a comma-separated .env value used to crash at import."""
        assert len(Settings(CORS_ORIGINS="http://localhost:3000,http://localhost:5173").cors_origins) == 2

    def test_production_validation_flags_insecure_defaults(self):
        settings = Settings(ENVIRONMENT="production", SECRET_KEY="dev-only-insecure-key-change-me",
                            DATABASE_URL="sqlite:///x.db", DEBUG=True)
        problems = settings.validate_production()
        assert any("SECRET_KEY" in p for p in problems)
        assert any("SQLite" in p for p in problems)
        assert any("DEBUG" in p for p in problems)

    def test_development_never_reports_problems(self):
        assert Settings(ENVIRONMENT="development").validate_production() == []

    def test_weights_exposed_as_mappings(self):
        settings = Settings()
        assert set(settings.recommendation_weights) == {
            "collaborative", "content", "popularity", "personalization", "business"}
        assert abs(sum(settings.recommendation_weights.values()) - 1.0) < 1e-6
