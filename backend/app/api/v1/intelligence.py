"""ML-facing endpoints: sentiment, forecasting, pricing and segmentation."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.deps import DbSession
from app.forecasting.service import ForecastingService
from app.schemas.ml import (
    ForecastResponse,
    PricePredictionResponse,
    ProductSentiment,
    SegmentOverview,
    SentimentRequest,
    SentimentResult,
)
from app.segmentation.service import SegmentationService
from app.sentiment.service import SentimentService
from app.services.pricing_service import PricingService

router = APIRouter(tags=["intelligence"])
ProductId = Annotated[int, Path(..., ge=1, description="Product identifier")]

MODEL_UNAVAILABLE = {503: {"description": "The required model has not been trained yet"}}


@router.get("/products/{product_id}/sentiment", response_model=ProductSentiment,
            summary="Aggregated review sentiment for a product",
            responses={404: {"description": "Product not found"}})
def product_sentiment(db: DbSession, product_id: ProductId) -> ProductSentiment:
    """Sentiment distribution, per-aspect breakdown, praise/complaints and keywords.

    Reviews that the batch pipeline has not yet scored are classified on demand.
    """
    return ProductSentiment.model_validate(SentimentService(db).product_sentiment(product_id))


@router.post("/sentiment/analyze", response_model=SentimentResult,
             summary="Classify arbitrary review text", responses=MODEL_UNAVAILABLE)
def analyze_sentiment(db: DbSession, payload: SentimentRequest) -> SentimentResult:
    """Returns an overall label plus per-aspect polarity with negation handling."""
    return SentimentResult.model_validate(SentimentService(db).analyse_text(payload.text))


@router.get("/forecast/{product_id}", response_model=ForecastResponse,
            summary="Demand forecast (7/14/30-day horizons)",
            responses={**MODEL_UNAVAILABLE, 404: {"description": "Product not found"}})
def forecast(db: DbSession, product_id: ProductId,
             horizon: int = Query(30, ge=1, le=120, description="Forecast horizon in days"),
             include_history: bool = Query(True)) -> ForecastResponse:
    """Recursive weekly forecast expanded to daily points, with 95% intervals,
    cumulative horizon totals and a stock-out risk assessment against inventory."""
    return ForecastResponse.model_validate(
        ForecastingService(db).forecast(product_id, horizon, include_history)
    )


@router.get("/forecast/accuracy/summary", summary="Hold-out accuracy of the forecasting model")
def forecast_accuracy(db: DbSession):
    return ForecastingService(db).accuracy()


@router.get("/forecast/demand/top", summary="Highest projected 30-day demand")
def top_demand(db: DbSession, limit: int = Query(10, ge=1, le=50)):
    return ForecastingService(db).top_demand(limit)


@router.get("/price-prediction/{product_id}", response_model=PricePredictionResponse,
            summary="Model-implied price with explainability",
            responses={**MODEL_UNAVAILABLE, 404: {"description": "Product not found"}})
def price_prediction(db: DbSession, product_id: ProductId) -> PricePredictionResponse:
    """Predicts a market-consistent price and explains it two ways: global
    permutation importance and per-product feature contributions."""
    return PricePredictionResponse.model_validate(PricingService(db).predict(product_id))


@router.get("/price-prediction/opportunities/list", summary="Largest price/model divergences")
def price_opportunities(db: DbSession, limit: int = Query(10, ge=1, le=50),
                        direction: str = Query("up", pattern="^(up|down|stable)$")):
    return PricingService(db).opportunities(limit, direction)


@router.get("/segments", response_model=SegmentOverview, summary="Customer segment overview")
def segments(db: DbSession) -> SegmentOverview:
    """K-Means + RFM segments with size, share and centroid characteristics."""
    return SegmentOverview.model_validate(SegmentationService(db).overview())


@router.get("/segments/{label}/members", summary="Customers in a segment")
def segment_members(db: DbSession, label: str, limit: int = Query(50, ge=1, le=200)):
    return SegmentationService(db).members(label, limit)
