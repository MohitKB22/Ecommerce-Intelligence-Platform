"""Event ingestion - the training signal for every behavioural model."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.core.logging_config import get_logger
from app.models import EVENT_TYPES, Product, User, UserEvent
from app.services.personalization import PersonalizationService

logger = get_logger("events")


class EventService:
    def __init__(self, db: Session):
        self.db = db
        self.personalization = PersonalizationService(db)

    def record(self, *, event_type: str, user_id: int | None = None, product_id: int | None = None,
               session_id: str = "anonymous", quantity: int = 1, value: float = 0.0,
               source: str = "web", metadata: dict[str, Any] | None = None,
               occurred_at: datetime | None = None) -> UserEvent:
        if event_type not in EVENT_TYPES:
            raise ValidationError(
                f"Unknown event_type '{event_type}'.", details={"allowed": list(EVENT_TYPES)}
            )
        if product_id is not None and self.db.get(Product, product_id) is None:
            raise NotFoundError(f"Product {product_id} was not found.")
        if user_id is not None and self.db.get(User, user_id) is None:
            raise NotFoundError(f"User {user_id} was not found.")
        if quantity < 1:
            raise ValidationError("quantity must be at least 1.")

        event = UserEvent(
            user_id=user_id, session_id=session_id or "anonymous", event_type=event_type,
            product_id=product_id, quantity=quantity, value=float(value), source=source,
            event_metadata=metadata or {}, occurred_at=occurred_at or datetime.now(timezone.utc),
        )
        self.db.add(event)

        if user_id is not None:
            user = self.db.get(User, user_id)
            if user is not None:
                user.last_active_at = event.occurred_at
        self.db.commit()
        self.db.refresh(event)

        # The profile is now stale.
        if user_id is not None:
            self.personalization.invalidate(user_id)

        # Attribute recommendation clicks/conversions when they follow an impression.
        if event_type in ("click", "purchase") and product_id is not None:
            try:
                from app.recommendation.service import RecommendationService

                service = RecommendationService(self.db)
                if event_type == "click":
                    service.mark_clicked(user_id, product_id)
                else:
                    service.mark_converted(user_id, product_id)
            except Exception as exc:  # noqa: BLE001 - attribution is best-effort
                logger.warning("attribution_failed", error=str(exc))

        logger.info("event_recorded", event_type=event_type, user_id=user_id, product_id=product_id)
        return event

    def record_batch(self, events: Iterable[dict[str, Any]]) -> dict[str, Any]:
        accepted, rejected = 0, []
        for i, payload in enumerate(events):
            try:
                self.record(**payload)
                accepted += 1
            except (ValidationError, NotFoundError) as exc:
                rejected.append({"index": i, "reason": exc.message})
        return {"accepted": accepted, "rejected": rejected}

    def recent_for_user(self, user_id: int, limit: int = 50) -> list[UserEvent]:
        return list(self.db.execute(
            select(UserEvent).where(UserEvent.user_id == user_id)
            .order_by(UserEvent.occurred_at.desc()).limit(limit)
        ).scalars().all())

    def funnel(self, days: int = 30) -> dict[str, int]:
        from datetime import timedelta

        from sqlalchemy import func

        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(UserEvent.event_type, func.count(UserEvent.id))
            .where(UserEvent.occurred_at >= since)
            .group_by(UserEvent.event_type)
        ).all()
        return {event_type: int(count) for event_type, count in rows}
