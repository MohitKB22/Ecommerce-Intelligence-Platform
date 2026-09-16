"""Recommendation endpoints for every storefront surface."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.deps import ActingUser, DbSession, SessionId
from app.recommendation.service import RecommendationService
from app.schemas.recommendations import HomepageResponse, RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/homepage", response_model=HomepageResponse,
            summary="Fully assembled personalized homepage")
def homepage(db: DbSession, user_id: ActingUser, session_id: SessionId,
             per_section: int = Query(8, ge=1, le=24)) -> HomepageResponse:
    """Returns every homepage rail in one round trip.

    The section mix itself is personalized: known shoppers get "Recommended for
    you", "Continue shopping" and "Recently viewed"; new visitors get popularity
    and trending rails instead.
    """
    return HomepageResponse.model_validate(
        RecommendationService(db).personalized_homepage(user_id, session_id, per_section)
    )


@router.get("/trending", response_model=RecommendationResponse, summary="Trending products")
def trending(db: DbSession, limit: int = Query(10, ge=1, le=50),
             category_id: int | None = Query(None, ge=1)) -> RecommendationResponse:
    return RecommendationResponse.model_validate(RecommendationService(db).trending(limit, category_id))


@router.get("/deals", response_model=RecommendationResponse, summary="Best current discounts")
def deals(db: DbSession, limit: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    return RecommendationResponse.model_validate(RecommendationService(db).deals(limit))


@router.get("/recently-viewed", response_model=RecommendationResponse, summary="Recently viewed items")
def recently_viewed(db: DbSession, user_id: ActingUser, session_id: SessionId,
                    limit: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    return RecommendationResponse.model_validate(
        RecommendationService(db).recently_viewed(user_id, session_id, limit)
    )


@router.get("/continue-shopping", response_model=RecommendationResponse, summary="Resume where you left off")
def continue_shopping(db: DbSession, user_id: ActingUser, session_id: SessionId,
                      limit: int = Query(8, ge=1, le=50)) -> RecommendationResponse:
    return RecommendationResponse.model_validate(
        RecommendationService(db).continue_shopping(user_id, session_id, limit)
    )


@router.get("/{user_id}", response_model=RecommendationResponse,
            summary="Personalized recommendations for a specific user")
def for_user(
    db: DbSession,
    session_id: SessionId,
    user_id: Annotated[int, Path(..., ge=1)],
    limit: int = Query(10, ge=1, le=50),
    surface: str = Query("homepage", max_length=40),
    w_collaborative: float | None = Query(None, ge=0, le=1,
                                             description="Override the collaborative weight"),
    w_content: float | None = Query(None, ge=0, le=1),
    w_popularity: float | None = Query(None, ge=0, le=1),
    w_personalization: float | None = Query(None, ge=0, le=1),
    w_business: float | None = Query(None, ge=0, le=1),
) -> RecommendationResponse:
    """Hybrid recommendations with optional per-request weight overrides.

    Unknown or low-signal users are routed to the cold-start path automatically;
    the response `strategy` field reports which path produced the results.
    """
    overrides = {
        key: value for key, value in {
            "collaborative": w_collaborative, "content": w_content, "popularity": w_popularity,
            "personalization": w_personalization, "business": w_business,
        }.items() if value is not None
    }
    return RecommendationResponse.model_validate(
        RecommendationService(db).for_user(
            user_id, limit, surface=surface, weights=overrides or None, session_id=session_id
        )
    )
