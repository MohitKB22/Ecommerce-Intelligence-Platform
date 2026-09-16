"""User, auth and profile schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class UserOut(ORMModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    country: str
    signup_source: str
    created_at: datetime
    last_active_at: datetime | None = None


class LoginRequest(BaseModel):
    """Login accepts any syntactically plausible address.

    Deliverability-style validation (as `EmailStr` performs) is wrong here: it
    rejects reserved TLDs such as `.local` and `.invalid`, which are exactly what
    synthetic and internal accounts use. Shape is checked; identity is proven by
    the password.
    """

    email: str = Field(..., min_length=3, max_length=255, examples=["admin@ecommerce-intelligence.local"])
    # No minimum length: a login endpoint must not enforce password *policy*.
    # Doing so returns 422 for a short wrong guess instead of a uniform 401,
    # which both leaks the policy and muddies the failure signal.
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _looks_like_email(cls, value: str) -> str:
        value = value.strip().lower()
        local, _, domain = value.partition("@")
        if not local or not domain or "." not in domain or " " in value:
            raise ValueError("must be a valid email address")
        return value


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str
    user_id: int


class UserProfileOut(BaseModel):
    user_id: int | None
    is_known: bool
    is_cold_start: bool
    interaction_count: int
    category_affinity: dict[str, float]
    brand_affinity: dict[str, float]
    recently_viewed: list[int]
    purchased: list[int]
    cart: list[int]
    wishlist: list[int]
    search_terms: list[str]
    preferred_price_range: list[float]
    avg_order_value: float
    order_count: int
    total_spend: float
    session_count: int
    review_count: int
    segment: str | None
    price_sensitivity: float
    last_active: str | None
    preference_vector: list[float]


class OrderItemOut(ORMModel):
    id: int
    product_id: int
    quantity: int
    unit_price: float
    discount_pct: float
    line_total: float


class OrderOut(ORMModel):
    id: int
    order_number: str
    user_id: int
    status: str
    total_amount: float
    discount_amount: float
    item_count: int
    currency: str
    channel: str
    placed_at: datetime
    items: list[OrderItemOut] = Field(default_factory=list)
