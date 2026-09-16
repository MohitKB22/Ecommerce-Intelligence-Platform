"""Product catalogue endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.deps import ActingUser, DbSession, PaginationDep, SessionId
from app.core.errors import ConflictError, NotFoundError
from app.recommendation.service import RecommendationService
from app.repositories.product_repository import SORT_OPTIONS, ProductRepository
from app.schemas.catalog import (
    BrandOut,
    CategoryOut,
    CategorySummary,
    ProductDetail,
    ProductSummary,
    ReviewCreate,
    ReviewOut,
)
from app.schemas.common import Page, PageMeta
from app.schemas.recommendations import RecommendationResponse
from app.sentiment.service import SentimentService

router = APIRouter(prefix="/products", tags=["products"])

ProductId = Annotated[int, Path(..., ge=1, description="Product identifier")]


@router.get("", response_model=Page[ProductSummary], summary="List products with filters and sorting")
def list_products(
    db: DbSession,
    pagination: PaginationDep,
    category: str | None = Query(None, description="Category slug, e.g. `audio`"),
    category_id: int | None = Query(None, ge=1),
    brand_id: int | None = Query(None, ge=1),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    min_rating: float | None = Query(None, ge=0, le=5),
    in_stock: bool = Query(False, description="Only products with inventory > 0"),
    on_sale: bool = Query(False, description="Only discounted products"),
    sort: str = Query("relevance", description=f"One of: {', '.join(SORT_OPTIONS)}"),
) -> Page[ProductSummary]:
    """Paginated catalogue browse. All filters are optional and combine with AND."""
    repo = ProductRepository(db)
    products, total = repo.list_products(
        page=pagination.page, page_size=pagination.page_size, category_slug=category,
        category_id=category_id, brand_id=brand_id, min_price=min_price, max_price=max_price,
        min_rating=min_rating, in_stock_only=in_stock, on_sale_only=on_sale, sort=sort,
    )
    return Page[ProductSummary](
        items=[ProductSummary.model_validate(p) for p in products],
        meta=PageMeta.build(pagination.page, pagination.page_size, total),
    )


@router.get("/categories", response_model=list[CategoryOut], summary="All categories")
def list_categories(db: DbSession) -> list[CategoryOut]:
    return [CategoryOut.model_validate(c) for c in ProductRepository(db).categories()]


@router.get("/categories/popular", response_model=list[CategorySummary],
            summary="Categories ranked by catalogue size")
def popular_categories(db: DbSession, limit: int = Query(8, ge=1, le=50)) -> list[CategorySummary]:
    return [CategorySummary.model_validate(c) for c in ProductRepository(db).popular_categories(limit)]


@router.get("/brands", response_model=list[BrandOut], summary="All brands")
def list_brands(db: DbSession, limit: int = Query(200, ge=1, le=500)) -> list[BrandOut]:
    return [BrandOut.model_validate(b) for b in ProductRepository(db).brands(limit)]


@router.get("/{product_id}", response_model=ProductDetail, summary="Product detail",
            responses={404: {"description": "Product not found"}})
def get_product(db: DbSession, product_id: ProductId) -> ProductDetail:
    product = ProductRepository(db).with_relations(product_id)
    if product is None:
        raise NotFoundError(f"Product {product_id} was not found.")
    return ProductDetail.model_validate(product)


@router.get("/{product_id}/reviews", response_model=Page[ReviewOut], summary="Reviews for a product")
def product_reviews(
    db: DbSession,
    product_id: ProductId,
    pagination: PaginationDep,
    sentiment: str | None = Query(None, pattern="^(positive|neutral|negative)$"),
) -> Page[ReviewOut]:
    repo = ProductRepository(db)
    if repo.get(product_id) is None:
        raise NotFoundError(f"Product {product_id} was not found.")
    reviews, total = repo.reviews_for(product_id, pagination.page, pagination.page_size, sentiment)
    return Page[ReviewOut](
        items=[ReviewOut.model_validate(r) for r in reviews],
        meta=PageMeta.build(pagination.page, pagination.page_size, total),
    )


@router.post("/{product_id}/reviews", response_model=ReviewOut, status_code=201,
             summary="Submit a review (scored by the sentiment model on write)",
             responses={404: {"description": "Unknown product or user"},
                        409: {"description": "This customer has already reviewed this product"}})
def create_review(db: DbSession, product_id: ProductId, payload: ReviewCreate) -> ReviewOut:
    """Persist a review, immediately classify its sentiment, and refresh product ratings."""
    from sqlalchemy import select

    from app.models import Review, User

    repo = ProductRepository(db)
    if repo.get(product_id) is None:
        raise NotFoundError(f"Product {product_id} was not found.")
    if db.get(User, payload.user_id) is None:
        raise NotFoundError(f"User {payload.user_id} was not found.")

    # One review per customer per product (enforced by a unique index).
    existing = db.execute(
        select(Review.id).where(Review.product_id == product_id, Review.user_id == payload.user_id)
    ).first()
    if existing is not None:
        raise ConflictError("This customer has already reviewed this product.")

    review = Review(
        product_id=product_id, user_id=payload.user_id, rating=payload.rating,
        title=payload.title, body=payload.body, verified_purchase=False,
    )
    sentiment = SentimentService(db).analyse_text(f"{payload.title}. {payload.body}")
    if sentiment["label"] != "unknown":
        review.sentiment_label = sentiment["label"]
        review.sentiment_score = sentiment["score"]
        review.sentiment_model_version = sentiment["model_version"]
        review.aspects = sentiment["aspects"]

    db.add(review)
    db.commit()
    db.refresh(review)
    repo.refresh_rating_aggregate(product_id)
    return ReviewOut.model_validate(review)


@router.get("/{product_id}/similar", response_model=RecommendationResponse, summary="Visually/semantically similar")
def similar(db: DbSession, product_id: ProductId,
            limit: int = Query(8, ge=1, le=50)) -> RecommendationResponse:
    return RecommendationResponse.model_validate(RecommendationService(db).similar_products(product_id, limit))


@router.get("/{product_id}/frequently-bought-together", response_model=RecommendationResponse,
            summary="Co-purchase recommendations")
def fbt(db: DbSession, product_id: ProductId, limit: int = Query(4, ge=1, le=20)) -> RecommendationResponse:
    return RecommendationResponse.model_validate(
        RecommendationService(db).frequently_bought_together(product_id, limit)
    )


@router.get("/{product_id}/recommendations", response_model=RecommendationResponse,
            summary="Personalized recommendations in the context of this product")
def product_recommendations(db: DbSession, product_id: ProductId, user_id: ActingUser,
                            session_id: SessionId,
                            limit: int = Query(8, ge=1, le=50)) -> RecommendationResponse:
    if ProductRepository(db).get(product_id) is None:
        raise NotFoundError(f"Product {product_id} was not found.")
    service = RecommendationService(db)
    return RecommendationResponse.model_validate(
        service.for_user(user_id, limit, surface="product_detail", exclude=[product_id],
                         session_id=session_id)
    )
