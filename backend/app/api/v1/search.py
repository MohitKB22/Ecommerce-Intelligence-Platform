"""Search endpoints: hybrid retrieval, suggestions, history."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import ActingUser, DbSession, SessionId
from app.repositories.product_repository import ProductRepository
from app.schemas.catalog import ProductSummary
from app.schemas.search import SearchFacets, SearchHit, SearchResults, SuggestionOut
from app.search.service import SearchFilters, SearchService
from app.services.personalization import PersonalizationService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResults, summary="Hybrid product search")
def search(
    db: DbSession,
    user_id: ActingUser,
    session_id: SessionId,
    q: str = Query("", max_length=200, description="Free-text query; supports typos and natural language"),
    page: int = Query(1, ge=1, le=1000),
    page_size: int = Query(20, ge=1, le=100),
    category: list[str] | None = Query(None, description="Repeatable category slug filter"),
    brand_id: list[int] | None = Query(None, description="Repeatable brand id filter"),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    min_rating: float | None = Query(None, ge=0, le=5),
    in_stock: bool = Query(False),
    on_sale: bool = Query(False),
    sort: str = Query("relevance",
                      pattern="^(relevance|price_asc|price_desc|rating|newest|discount)$"),
    personalize: bool = Query(True, description="Apply the caller's preference profile to ranking"),
) -> SearchResults:
    """Blend BM25 keyword matching, dense-vector semantic similarity, popularity,
    rating, conversion, personalization and availability into a single ranking.

    Typos are corrected against the index vocabulary and natural-language
    constraints such as "under $100" are parsed into filters.
    """
    service = SearchService(db)
    filters = SearchFilters(
        category_slugs=list(category or []), brand_ids=list(brand_id or []),
        min_price=min_price, max_price=max_price, min_rating=min_rating,
        in_stock_only=in_stock, on_sale_only=on_sale,
    )
    profile = None
    if personalize and (user_id is not None or session_id):
        profile = PersonalizationService(db).get_profile(user_id, session_id)

    response = service.search(
        q, filters=filters, limit=page_size, offset=(page - 1) * page_size,
        sort=sort, user_profile=profile,
    )
    service.log_search(
        query=q, user_id=user_id, session_id=session_id, result_count=response.total,
        latency_ms=response.took_ms, filters={"category": category, "brand_id": brand_id, "sort": sort},
    )

    products = {p.id: p for p in ProductRepository(db).get_many(response.product_ids())}
    hits = [
        SearchHit(
            product=ProductSummary.model_validate(products[result.product_id]),
            score=result.score,
            signals=result.signals.as_dict(),  # type: ignore[arg-type]
            explanation=result.explanation,
        )
        for result in response.results if result.product_id in products
    ]
    return SearchResults(
        query=response.query, normalized_query=response.normalized_query,
        corrected_query=response.corrected_query, strategy=response.strategy,
        total=response.total, took_ms=response.took_ms, weights=response.weights,
        hits=hits, facets=SearchFacets(**response.facets),
    )


@router.get("/suggestions", response_model=list[SuggestionOut], summary="Autocomplete suggestions")
def suggestions(db: DbSession, q: str = Query(..., min_length=1, max_length=100),
                limit: int = Query(8, ge=1, le=20)) -> list[SuggestionOut]:
    """Prefix suggestions drawn from product titles, historical queries and index terms."""
    return [SuggestionOut(**s) for s in SearchService(db).autocomplete(q, limit)]


@router.get("/popular", summary="Most frequent queries with their click-through rate")
def popular(db: DbSession, limit: int = Query(10, ge=1, le=50), days: int = Query(30, ge=1, le=365)):
    return SearchService(db).popular_queries(limit, days)


@router.get("/history", summary="A user's recent searches")
def history(db: DbSession, user_id: ActingUser, limit: int = Query(20, ge=1, le=100)):
    if user_id is None:
        return []
    return SearchService(db).history(user_id, limit)
