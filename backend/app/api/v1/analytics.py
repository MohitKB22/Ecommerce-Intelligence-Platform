"""Analytics and ML monitoring endpoints (admin-authenticated)."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import AdminPrincipal, DbSession
from app.ml.model_store import get_model_store
from app.schemas.ml import ModelRegistryOut
from app.services.analytics_service import AnalyticsService
from app.services.feature_store import FeatureStore
from app.services.monitoring_service import MonitoringService

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard", summary="Complete admin dashboard payload (admin only)")
def dashboard(db: DbSession, principal: AdminPrincipal, days: int = Query(30, ge=1, le=365)):
    """Business, AI, customer, product and sentiment analytics in a single response.

    Every figure is computed from the operational tables.
    """
    return AnalyticsService(db).dashboard(days)


@router.get("/business", summary="Revenue, orders, AOV, conversion and retention")
def business(db: DbSession, principal: AdminPrincipal, days: int = Query(30, ge=1, le=365)):
    return AnalyticsService(db).business_metrics(days)


@router.get("/ai", summary="Recommendation and search effectiveness")
def ai(db: DbSession, principal: AdminPrincipal, days: int = Query(30, ge=1, le=365)):
    return AnalyticsService(db).ai_metrics(days)


@router.get("/customers", summary="Customer mix, segments and top spenders")
def customers(db: DbSession, principal: AdminPrincipal, days: int = Query(30, ge=1, le=365)):
    return AnalyticsService(db).customer_analytics(days)


@router.get("/products", summary="Best sellers, low stock and conversion outliers")
def products(db: DbSession, principal: AdminPrincipal, limit: int = Query(10, ge=1, le=50)):
    return AnalyticsService(db).product_analytics(limit)


@router.get("/revenue-timeseries", summary="Daily revenue and order counts")
def revenue_timeseries(db: DbSession, principal: AdminPrincipal, days: int = Query(30, ge=1, le=365)):
    return AnalyticsService(db).revenue_timeseries(days)


# ---- ML monitoring -----------------------------------------------------
ml_router = APIRouter(prefix="/ml", tags=["ml-monitoring"])


@ml_router.get("/registry", response_model=list[ModelRegistryOut], summary="Model registry (admin only)")
def registry(db: DbSession, principal: AdminPrincipal) -> list[ModelRegistryOut]:
    """Every registered model version with training metrics and live serving stats."""
    return [ModelRegistryOut.model_validate(row) for row in MonitoringService(db).registry()]


@ml_router.get("/health", summary="Per-model availability, latency and error rate")
def model_health(db: DbSession, principal: AdminPrincipal, days: int = Query(7, ge=1, le=90)):
    return MonitoringService(db).model_health(days)


@ml_router.get("/drift", summary="Data and prediction drift report")
def drift(db: DbSession, principal: AdminPrincipal):
    """Compares live feature distributions against the statistics captured at training time."""
    return MonitoringService(db).drift_report()


@ml_router.get("/summary", summary="One-line ML system status")
def summary(db: DbSession, principal: AdminPrincipal):
    return MonitoringService(db).summary()


@ml_router.post("/reload", summary="Hot-reload model artifacts from the registry")
def reload_models(principal: AdminPrincipal):
    """Picks up newly trained versions without restarting the API."""
    return {"reloaded": get_model_store().reload_all()}


@ml_router.get("/features/coverage", summary="Feature store coverage")
def feature_coverage(db: DbSession, principal: AdminPrincipal, version: str = Query("v1")):
    return FeatureStore(db, version).coverage(version)


@ml_router.get("/features/user/{user_id}", summary="Feature vector for a user")
def user_features(db: DbSession, principal: AdminPrincipal, user_id: int, version: str = Query("v1")):
    return FeatureStore(db, version).get_user_features(user_id).as_dict()


@ml_router.get("/features/product/{product_id}", summary="Feature vector for a product")
def product_features(db: DbSession, principal: AdminPrincipal, product_id: int, version: str = Query("v1")):
    return FeatureStore(db, version).get_product_features(product_id).as_dict()
