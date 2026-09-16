# API reference

Base URL `http://localhost:8000/api/v1` · interactive docs at `/docs` (Swagger)
and `/redoc`. The machine-readable schema is at `/openapi.json`.

56 operations across 55 paths.

## Conventions

**Pagination.** List endpoints take `page` (≥1) and `page_size` (1-100, default 24)
and return:

```json
{"items": [...], "meta": {"page": 1, "page_size": 24, "total": 600,
                          "total_pages": 25, "has_next": true, "has_previous": false}}
```

**Errors.** Every failure uses one envelope. Stack traces are never exposed.

```json
{"error": {"code": "not_found",
           "message": "Product 42 was not found.",
           "request_id": "a1b2c3d4e5f6a7b8",
           "details": [{"field": "rating", "issue": "Input should be less than or equal to 5"}]}}
```

| Code | HTTP | Meaning |
|---|---|---|
| `validation_error` | 422 | Request failed schema validation |
| `unauthenticated` | 401 | Missing or invalid bearer token |
| `forbidden` | 403 | Authenticated but not an administrator |
| `not_found` | 404 | Unknown resource |
| `conflict` | 409 | Uniqueness or referential constraint violated |
| `rate_limited` | 429 | Rate limit exceeded (`Retry-After` header set) |
| `model_unavailable` | 503 | Model not trained — run `make train` |
| `dependency_unavailable` | 503 | Database or cache unreachable |
| `internal_error` | 500 | Unexpected fault (logged with the request id) |

**Headers.** Responses carry `X-Request-Id`, `X-Response-Time-Ms` and
`X-RateLimit-Remaining`. Send `X-Session-Id` to personalise anonymous traffic.

**Auth.** `POST /auth/login` returns a JWT. Send `Authorization: Bearer <token>`.
Admin endpoints require `role=admin`.

---

## Health

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Aggregate status of database, cache and models |
| GET | `/health/live` | Liveness probe (never touches dependencies) |
| GET | `/health/ready` | Readiness — 503 when the database is unreachable |

`/health` returns `ok`, `degraded` (optional dependency down — still HTTP 200) or
`error` (HTTP 503).

## Auth

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/login` | `{email, password}` → token. 401 on bad credentials |
| GET | `/auth/me` | Current user |

Login validates address *shape* only, not deliverability — reserved TLDs such as
`.invalid` and `.local` are used by seeded and internal accounts and must work.

## Products

| Method | Path | Notes |
|---|---|---|
| GET | `/products` | Filters: `category`, `brand_id`, `min_price`, `max_price`, `min_rating`, `in_stock`, `on_sale`, `sort` |
| GET | `/products/{id}` | Detail with category and brand |
| GET | `/products/categories` | All categories |
| GET | `/products/categories/popular` | Ranked by catalogue size |
| GET | `/products/brands` | All brands |
| GET | `/products/{id}/reviews` | Paginated, filter by `sentiment` |
| POST | `/products/{id}/reviews` | Creates a review, scores sentiment on write, refreshes rating aggregate. 409 if the customer already reviewed it |
| GET | `/products/{id}/similar` | Content + behavioural similarity |
| GET | `/products/{id}/frequently-bought-together` | Lift-weighted co-purchase |
| GET | `/products/{id}/recommendations` | Personalized, excluding this product |
| GET | `/products/{id}/sentiment` | Aggregated review sentiment + aspects |

`sort` ∈ `relevance | price_asc | price_desc | rating | newest | discount`.

## Search

| Method | Path | Notes |
|---|---|---|
| GET | `/search` | Hybrid retrieval + ranking |
| GET | `/search/suggestions` | Autocomplete from titles, past queries and vocabulary |
| GET | `/search/popular` | Top queries with CTR |
| GET | `/search/history` | A user's recent searches |

Example:

```bash
curl "http://localhost:8000/api/v1/search?q=wireless+headphones+for+gym&min_rating=4&page_size=5"
```

```json
{
  "query": "wireless headphones for gym",
  "corrected_query": null,
  "strategy": "hybrid",
  "total": 293,
  "took_ms": 3.25,
  "weights": {"text": 0.30, "semantic": 0.25, "popularity": 0.12, "rating": 0.10,
              "conversion": 0.10, "personal": 0.08, "availability": 0.05},
  "hits": [{
    "product": {"id": 191, "title": "Tundrix Headphones Elite 538", "effective_price": 214.5, "...": "..."},
    "score": 0.7555,
    "signals": {"text": 1.0, "semantic": 0.87, "popularity": 0.41, "rating": 0.78,
                "conversion": 0.12, "personal": 0.0, "availability": 1.0},
    "explanation": "Ranked by keyword match and semantic similarity"
  }],
  "facets": {"categories": [...], "brands": [...], "price": {"min": 19.0, "max": 480.0}}
}
```

Typos are corrected against the index vocabulary (`headphnes` → `headphones`) and
reported in `corrected_query`. Natural-language constraints such as
`laptop under $500` are parsed into filters.

## Recommendations

| Method | Path | Notes |
|---|---|---|
| GET | `/recommendations/homepage` | Every rail in one call; section mix is itself personalized |
| GET | `/recommendations/{user_id}` | Hybrid, with optional `w_*` weight overrides |
| GET | `/recommendations/trending` | Trending over 14 days |
| GET | `/recommendations/deals` | Largest active discounts |
| GET | `/recommendations/recently-viewed` | From the event stream |
| GET | `/recommendations/continue-shopping` | Cart remnants, then similar items |

Weight overrides accept `w_collaborative`, `w_content`, `w_popularity`,
`w_personalization`, `w_business` (each 0-1).

```json
{
  "items": [{
    "product": {"id": 88, "title": "Norsel Sleep Band Plus 759", "...": "..."},
    "score": 0.6421, "rank": 1,
    "components": {"collaborative": 0.52, "content": 0.31, "popularity": 0.88,
                   "personalization": 0.44, "business": 0.61},
    "explanation": "Recommended because it is trending across the store and customers with similar taste bought this"
  }],
  "strategy": "hybrid",
  "model_version": "v20260814041922",
  "count": 10
}
```

`strategy` reports which path ran: `hybrid`, `cold_start`, `trending_events`,
`co_purchase_model`, `popularity_fallback`.

## Events

| Method | Path | Notes |
|---|---|---|
| POST | `/events` | Record one event |
| POST | `/events/batch` | Up to 500; partial success reported per index |
| GET | `/events/funnel` | Counts by type |

Types: `product_view`, `search`, `click`, `add_to_cart`, `remove_from_cart`,
`wishlist`, `purchase`, `review`. Clicks and purchases are automatically
attributed back to the recommendation impression that produced them, which is
what powers the CTR and conversion figures on the dashboard.

## Users

| Method | Path | Notes |
|---|---|---|
| GET | `/users/{id}` | User record |
| GET | `/users/{id}/profile` | Preference profile driving all personalization |
| GET | `/users/{id}/orders` | Paginated order history |
| GET | `/users/{id}/events` | Recent activity |
| GET | `/users/{id}/segment` | ML segment assignment |

## Intelligence

| Method | Path | Notes |
|---|---|---|
| GET | `/products/{id}/sentiment` | Distribution, aspects, praise, complaints, keywords |
| POST | `/sentiment/analyze` | Score arbitrary text |
| GET | `/forecast/{product_id}` | 7/14/30-day demand with intervals and stock-out risk |
| GET | `/forecast/accuracy/summary` | Hold-out metrics of the live model |
| GET | `/forecast/demand/top` | Highest projected 30-day demand |
| GET | `/price-prediction/{product_id}` | Model-implied price with explanations |
| GET | `/price-prediction/opportunities/list` | Largest price/model divergences |
| GET | `/segments` | Segment overview |
| GET | `/segments/{label}/members` | Customers in a segment |

```bash
curl -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"Battery life is excellent but the ear cushions are uncomfortable."}'
```

```json
{"label": "positive", "score": 0.62,
 "aspects": {"battery_life": "positive", "comfort": "negative"},
 "positive_aspects": ["battery_life"], "negative_aspects": ["comfort"],
 "model_version": "v20260814041905"}
```

## Analytics and ML monitoring (admin only)

| Method | Path | Notes |
|---|---|---|
| GET | `/analytics/dashboard` | Complete dashboard payload |
| GET | `/analytics/business` | Revenue, orders, AOV, conversion, retention |
| GET | `/analytics/ai` | Recommendation and search effectiveness |
| GET | `/analytics/customers` | Segments, new/returning, top spenders |
| GET | `/analytics/products` | Best sellers, low stock, conversion outliers |
| GET | `/analytics/revenue-timeseries` | Daily revenue and orders |
| GET | `/ml/registry` | Every model version with metrics |
| GET | `/ml/health` | Availability, latency, error rate |
| GET | `/ml/drift` | Drift report |
| GET | `/ml/summary` | One-line ML status |
| POST | `/ml/reload` | Hot-reload artifacts |
| GET | `/ml/features/coverage` | Feature store coverage |
| GET | `/ml/features/user/{id}` | A user's feature vector |
| GET | `/ml/features/product/{id}` | A product's feature vector |

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@ecommerce-intelligence.local","password":"admin-change-me"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/analytics/dashboard
```

## Rate limiting

Fixed-window, default 300 requests per 60s per client, keyed on
`X-Forwarded-For` or the peer address. Health and documentation routes are
exempt. Disable with `RATE_LIMIT_ENABLED=false`.
