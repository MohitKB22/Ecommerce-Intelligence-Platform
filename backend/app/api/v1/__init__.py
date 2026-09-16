"""API v1 router aggregation."""
from fastapi import APIRouter

from app.api.v1 import (
    analytics,
    auth,
    events,
    health,
    intelligence,
    products,
    recommendations,
    search,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(products.router)
api_router.include_router(search.router)
api_router.include_router(recommendations.router)
api_router.include_router(events.router)
api_router.include_router(users.router)
api_router.include_router(intelligence.router)
api_router.include_router(analytics.router)
api_router.include_router(analytics.ml_router)

__all__ = ["api_router"]
