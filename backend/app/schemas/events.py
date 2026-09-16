"""Event tracking schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models import EVENT_TYPES
from app.schemas.common import ORMModel

EventTypeLiteral = str


class EventCreate(BaseModel):
    event_type: str = Field(..., description=f"One of: {', '.join(EVENT_TYPES)}", examples=["product_view"])
    user_id: int | None = Field(None, ge=1)
    product_id: int | None = Field(None, ge=1)
    session_id: str = Field("anonymous", max_length=64)
    quantity: int = Field(1, ge=1, le=999)
    value: float = Field(0.0, ge=0)
    source: str = Field("web", max_length=40)
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime | None = None


class EventBatch(BaseModel):
    events: list[EventCreate] = Field(..., min_length=1, max_length=500)


class EventOut(ORMModel):
    id: int
    user_id: int | None
    session_id: str
    event_type: str
    product_id: int | None
    quantity: int
    value: float
    source: str
    occurred_at: datetime


class BatchResult(BaseModel):
    accepted: int
    rejected: list[dict[str, Any]] = Field(default_factory=list)
