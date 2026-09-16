"""User, profile and order endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, PaginationDep, SessionId
from app.core.errors import NotFoundError
from app.models import Order, User
from app.schemas.common import Page, PageMeta
from app.schemas.events import EventOut
from app.schemas.users import OrderOut, UserOut, UserProfileOut
from app.segmentation.service import SegmentationService
from app.services.event_service import EventService
from app.services.personalization import PersonalizationService

router = APIRouter(prefix="/users", tags=["users"])
UserId = Annotated[int, Path(..., ge=1, description="User identifier")]


def _require_user(db, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError(f"User {user_id} was not found.")
    return user


@router.get("/{user_id}", response_model=UserOut, summary="User record",
            responses={404: {"description": "User not found"}})
def get_user(db: DbSession, user_id: UserId) -> UserOut:
    return UserOut.model_validate(_require_user(db, user_id))


@router.get("/{user_id}/profile", response_model=UserProfileOut,
            summary="Behavioural preference profile")
def get_profile(db: DbSession, user_id: UserId, session_id: SessionId) -> UserProfileOut:
    """The derived profile that drives homepage, search and recommendation personalization.

    Includes time-decayed category and brand affinities, the inferred preferred
    price range, and a compact dense preference vector.
    """
    _require_user(db, user_id)
    profile = PersonalizationService(db).get_profile(user_id, session_id)
    return UserProfileOut.model_validate(profile.as_dict())


@router.get("/{user_id}/orders", response_model=Page[OrderOut], summary="Order history")
def get_orders(db: DbSession, user_id: UserId, pagination: PaginationDep) -> Page[OrderOut]:
    _require_user(db, user_id)
    stmt = (
        select(Order).options(selectinload(Order.items))
        .where(Order.user_id == user_id).order_by(Order.placed_at.desc())
    )
    total = db.query(Order).filter(Order.user_id == user_id).count()
    rows = db.execute(stmt.offset(pagination.offset).limit(pagination.page_size)).scalars().all()
    return Page[OrderOut](
        items=[OrderOut.model_validate(o) for o in rows],
        meta=PageMeta.build(pagination.page, pagination.page_size, total),
    )


@router.get("/{user_id}/events", response_model=list[EventOut], summary="Recent activity")
def get_events(db: DbSession, user_id: UserId, limit: int = Query(50, ge=1, le=200)) -> list[EventOut]:
    _require_user(db, user_id)
    return [EventOut.model_validate(e) for e in EventService(db).recent_for_user(user_id, limit)]


@router.get("/{user_id}/segment", summary="ML segment assignment for a customer")
def get_segment(db: DbSession, user_id: UserId):
    return SegmentationService(db).for_user(user_id)
