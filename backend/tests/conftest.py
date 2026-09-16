"""Shared pytest fixtures.

Tests run against a small, deterministic synthetic dataset in an isolated SQLite
database so the suite is fast, hermetic and needs no external services.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Configure the environment BEFORE importing application modules, because
# settings are read at import time.
_TMP = Path(tempfile.mkdtemp(prefix="eci-tests-"))
os.environ.setdefault("ENVIRONMENT", "test")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["MODEL_PATH"] = str(_TMP / "models")
os.environ["REDIS_URL"] = ""          # force the in-memory cache backend
os.environ["CELERY_BROKER_URL"] = ""
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["ML_AUTOLOAD"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["ADMIN_EMAIL"] = "admin@test.local"
os.environ["ADMIN_PASSWORD"] = "test-admin-password"


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def seeded_db(tmp_root: Path):
    """Create the schema and load a compact synthetic dataset once per session."""
    from app.core.db import create_all, engine
    from ml.datasets.synthetic import GeneratorConfig
    from scripts.seed_database import seed

    create_all()
    cfg = GeneratorConfig(
        n_users=120, n_products=90, n_days=180, seed=4242,
        target_reviews=320, target_browse_events=3200, target_search_events=420,
    )
    seed(cfg, reset=True, export_csv=False)
    yield engine


@pytest.fixture()
def db(seeded_db):
    """A transactional session that is rolled back after each test."""
    from app.core.db import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def trained_models(seeded_db, tmp_root: Path):
    """Train every model once so inference paths can be exercised for real."""
    from ml.training.train_forecasting import ForecastingPipeline
    from ml.training.train_price_prediction import PricePredictionPipeline
    from ml.training.train_recommendation import RecommendationPipeline
    from ml.training.train_segmentation import SegmentationPipeline
    from ml.training.train_sentiment import SentimentPipeline

    registry_root = tmp_root / "models"
    results = {}
    for name, pipeline_cls in [
        ("sentiment", SentimentPipeline),
        ("segmentation", SegmentationPipeline),
        ("recommendation", RecommendationPipeline),
        ("forecasting", ForecastingPipeline),
        ("price_prediction", PricePredictionPipeline),
    ]:
        try:
            results[name] = pipeline_cls(registry_root=registry_root).run()
        except Exception as exc:  # noqa: BLE001 - recorded so tests can skip precisely
            results[name] = exc
    return results


@pytest.fixture()
def client(seeded_db):
    """FastAPI test client with cache and model state reset per test."""
    from fastapi.testclient import TestClient

    from app.core.cache import reset_cache
    from app.ml.model_store import reset_model_store
    from app.main import app

    reset_cache()
    reset_model_store()
    with TestClient(app) as test_client:
        yield test_client
    reset_cache()
    reset_model_store()


@pytest.fixture()
def admin_headers(client) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@test.local", "password": "test-admin-password"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def sample_product_id(db) -> int:
    from sqlalchemy import select

    from app.models import Product

    return int(db.execute(select(Product.id).limit(1)).scalar_one())


@pytest.fixture()
def sample_user_id(db) -> int:
    from sqlalchemy import select

    from app.models import User

    return int(db.execute(select(User.id).where(User.role == "user").limit(1)).scalar_one())
