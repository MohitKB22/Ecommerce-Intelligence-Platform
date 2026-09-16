"""ORM model registry - importing this module registers every table."""
from app.models.catalog import Brand, Category, Product
from app.models.commerce import ORDER_STATUSES, Order, OrderItem, Review
from app.models.events import EVENT_TYPES, EVENT_WEIGHTS, RecommendationLog, SearchEvent, UserEvent
from app.models.mixins import PKMixin, TimestampMixin, utcnow
from app.models.ml import (
    Forecast,
    ModelPrediction,
    ModelRegistryEntry,
    PricePrediction,
    ProductFeature,
    UserFeature,
)
from app.models.user import CustomerSegment, User

__all__ = [
    "Brand", "Category", "Product",
    "Order", "OrderItem", "Review", "ORDER_STATUSES",
    "UserEvent", "SearchEvent", "RecommendationLog", "EVENT_TYPES", "EVENT_WEIGHTS",
    "ProductFeature", "UserFeature", "ModelRegistryEntry", "ModelPrediction", "Forecast", "PricePrediction",
    "User", "CustomerSegment",
    "PKMixin", "TimestampMixin", "utcnow",
]
