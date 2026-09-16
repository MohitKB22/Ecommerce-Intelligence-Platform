"""Behavioural event ingestion."""
from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, SessionId
from app.schemas.events import BatchResult, EventBatch, EventCreate, EventOut
from app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["events"])


@router.post("", response_model=EventOut, status_code=status.HTTP_201_CREATED,
             summary="Record a behavioural event",
             responses={404: {"description": "Unknown user or product"},
                        422: {"description": "Unknown event_type or invalid payload"}})
def record_event(db: DbSession, payload: EventCreate, session_id: SessionId) -> EventOut:
    """Ingest a single event.

    Events are the training signal for the recommendation, segmentation and
    forecasting models, and clicks/purchases are automatically attributed back to
    the recommendation impression that produced them.
    """
    event = EventService(db).record(
        event_type=payload.event_type, user_id=payload.user_id, product_id=payload.product_id,
        session_id=payload.session_id or session_id, quantity=payload.quantity, value=payload.value,
        source=payload.source, metadata=payload.metadata, occurred_at=payload.occurred_at,
    )
    return EventOut.model_validate(event)


@router.post("/batch", response_model=BatchResult, summary="Record up to 500 events in one call")
def record_batch(db: DbSession, payload: EventBatch, session_id: SessionId) -> BatchResult:
    """Partial success: valid events are stored and invalid ones reported per index."""
    events = [
        {**event.model_dump(), "session_id": event.session_id or session_id}
        for event in payload.events
    ]
    return BatchResult(**EventService(db).record_batch(events))


@router.get("/funnel", summary="Event counts by type over a window")
def funnel(db: DbSession, days: int = Query(30, ge=1, le=365)):
    return EventService(db).funnel(days)
