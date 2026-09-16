# BUILD REPORT

**Project:** E-Commerce Intelligence
**Build date:** 2026-08-14
**Package:** `ecommerce-intelligence-production.zip` (455 KB, 262 entries)
**Status:** Production-ready local deployment. Docker Compose and browser-based
E2E were **not executable in the build environment** and are marked UNVERIFIED
below rather than claimed as passing.

---

## Component status

| Component | Result | How it was verified |
|---|---|---|
| Backend tests | **PASS** | `pytest backend/tests` — 173 passed |
| Frontend tests | **PASS** | `vitest run` — 27 passed |
| Frontend build | **PASS** | `vite build` — 21 chunks, largest 394 kB (107 kB gzip) |
| Frontend lint | **PASS** | `eslint --max-warnings 0` — 0 errors, 0 warnings |
| Frontend type check | **PASS** | `tsc --noEmit` — clean |
| Backend lint | **PASS** | `ruff check backend/app ml scripts` — All checks passed |
| ML validation | **PASS** | All 5 pipelines trained; each beats its baseline |
| Database migrations | **PASS** | `alembic upgrade head` → 17 tables; full downgrade/upgrade round-trip |
| Seed data | **PASS** | 1,200 users · 600 products · 6,000 reviews · 136,000 events · 16,743 orders |
| API | **PASS** | 56 operations; every endpoint exercised via TestClient |
| Project validator | **PASS** | `validate_project.py` — 36 passed, 0 required failures, 1 warning |
| Final health check | **PASS** | `/api/v1/health` → `status: ok` |
| Docker build | **UNVERIFIED** | Docker unavailable in the build sandbox |
| Docker Compose | **UNVERIFIED** | Docker unavailable in the build sandbox |
| Browser E2E | **UNVERIFIED** | No browser available; API-level E2E journeys pass instead |
| Redis path | **UNVERIFIED** | No Redis available; in-process cache fallback exercised |
| PostgreSQL path | **UNVERIFIED** | No PostgreSQL available; SQLite path exercised |

> The Dockerfiles, Compose files and CI workflows are complete and syntactically
> validated (`yaml.safe_load` on all five). They have not been *executed*.

---

## Tests

```
Backend    173 passed
Frontend    27 passed
Total      200 passed, 0 failed
```

Breakdown:

| Suite | Tests | Covers |
|---|---|---|
| `tests/unit/test_core.py` | 28 | Config, cache, TTL/LRU, rate limiting, password hashing, JWT, error envelopes |
| `tests/unit/test_ml_components.py` | 45 | Ranking/regression/classification/drift metrics, BM25, edit distance, embeddings, registry |
| `tests/unit/test_search_service.py` | 23 | Ranking math, vector store, intent parsing, filters, facets, pagination |
| `tests/integration/test_api.py` | 47 | Every route, auth, pagination bounds, validation, error shapes, OpenAPI |
| `tests/integration/test_ml_pipelines.py` | 22 | Trains all 5 models, asserts baselines beaten, all inference paths, graceful degradation |
| `tests/e2e/test_user_journey.py` | 9 | Full shopper journey, admin journey, metric consistency |
| `frontend/src/__tests__` | 27 | Formatting, catalog service, API client error handling, ProductCard, UI primitives |

---

## ML validation

Every model is compared against an explicit baseline, and the comparison is a
test that fails CI if a model regresses.

| Model | Metric | Model | Baseline | Improvement |
|---|---|---|---|---|
| Recommendation | NDCG@10 | **0.0638** | 0.0610 (popularity) | **+4.6%** |
| Recommendation | NDCG@10 | 0.0638 | 0.0065 (random) | **+880%** |
| Sentiment | Accuracy | **0.9147** | 0.7687 (majority class) | **+19.0%** |
| Forecasting | MAE | **1.0886** | 1.1902 (seasonal naive) | **−8.5%** |
| Price prediction | MAE | **$32.36** | $66.13 (category median) | **−51.1%** |
| Segmentation | Silhouette | **0.3371** | — | 6 named segments |

Supporting figures: recommendation precision@10 0.0237, recall@20 0.1109,
catalogue coverage@10 0.278; sentiment F1-macro 0.8384, aspect detection F1
0.7730, aspect polarity accuracy 0.9956; forecasting RMSE 1.6800 across 441
forecastable products; price R² 0.9680, MAPE 11.90%, 82.5% within 20%.

All five pipelines complete in **26 seconds** total on the full dataset.

---

## Bugs found and fixed

**20 bugs were found and all 20 were fixed.** The substantive ones:

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | SQLite `PRAGMA journal_mode=WAL` raised on filesystems without shared-memory locking | Every DB connection failed | Probe WAL → TRUNCATE → DELETE and degrade silently |
| 2 | `vars()` on a slots dataclass | Seed script crashed | Explicit field access |
| 3 | Starlette renamed `HTTP_422_UNPROCESSABLE_ENTITY` | Deprecation warnings, version-fragile | Use numeric status codes |
| 4 | Baskets were single-item; no co-purchase signal | "Frequently bought together" had nothing to learn from | Complementary-category basket expansion |
| 5 | **Hybrid recommender scored 58% *worse* than a popularity baseline** | Core feature was actively harmful | Popularity-residualise CF and content components; BM25-weight the implicit matrix; 32 factors |
| 6 | Structured logs written to stdout | Corrupted machine-readable CLI output | Log to stderr |
| 7 | Baseline comparison excluded history for the model but not the baseline | Evaluation was invalid | Identical exclusion rules for every arm |
| 8 | `EmailStr` rejects reserved TLDs | **No seeded account could log in**; every admin endpoint 401'd | Validate address shape, not deliverability |
| 9 | Segment cluster profiles keyed by `int` | `/segments` returned 500 | Coerce keys to `str` for JSON |
| 10 | `func.case(...)` is not valid SQLAlchemy | `/ml/health` returned 500 | Use top-level `case()` |
| 11 | Weekly demand reindex misaligned (`to_period` vs `date_range` anchor) | **Every forecast was zero** | Monday-anchored week start + an assertion guard |
| 12 | `base_cost` leaked the price target (r=0.985) | Price model R²=0.988 while learning nothing | Feature removed; regression test added |
| 13 | `category_price_rank` was the target in disguise | Same | Feature removed; leave-one-out category stats |
| 14 | Review *titles* generated deterministically from rating | Sentiment scored a meaningless 100% | Titles drawn from a noisy tone variable; test asserts accuracy < 1.0 |
| 15 | GBM on raw units lost to the naive forecaster | "Advanced" model added nothing | Predict the residual against the baseline instead |
| 16 | Every event got its own session id | 73% session conversion, view→cart > 1.0 | Realistic multi-event sessions, higher browse volume |
| 17 | Duplicate review raised a raw `IntegrityError` | **Unhandled 500** | Global `IntegrityError` → 409 plus an explicit pre-check |
| 18 | Login enforced a password *minimum length* | Short wrong guesses got 422 instead of 401, leaking policy | Validate presence only |
| 19 | Weight fitting ranked without history exclusion; evaluation ranked with it | Weights fitted under conditions they were never scored in | Both paths use the same ranking function |

| 20 | `CORS_ORIGINS` declared as `list[str]`; pydantic-settings JSON-decodes complex types from dotenv *before* validators run | **`cp .env.example .env` — the documented first step — crashed the app on startup.** Caught only by testing the extracted package | Store as `str`, parse in a `cors_origins` property; 4 regression tests added |

Also fixed during the build: composite `tsconfig` with `noEmit`, `vite.config.ts`
importing `defineConfig` from `vite` instead of `vitest/config`, naive/aware
datetime subtraction in the feature-store worker, a positive-biased clause
classifier that collapsed aspect polarity accuracy to 0.35, and 254 lint
violations auto-corrected with the remainder resolved by hand.

---

## What was built

- **17-table schema** with foreign keys, check constraints, composite indexes and
  a reversible Alembic migration.
- **56 API operations** across 55 paths with Pydantic validation, pagination,
  JWT auth, rate limiting, security headers and a single error envelope.
- **5 ML pipelines**, each with training, baseline-compared evaluation, a
  registered artifact, inference, API integration and UI integration.
- **11 frontend routes** — storefront plus admin analytics, ML monitoring and
  system health — every one wired to a real endpoint.
- **Dual-mode runtime**: PostgreSQL/Redis/Celery/pgvector in production,
  SQLite/in-process cache/thread scheduler/numpy index with zero setup.
- **Graceful degradation**: missing models return a clean `503 model_unavailable`
  and the storefront falls back to popularity ranking rather than failing.

## Known limitations

- Docker, PostgreSQL, Redis and browser E2E paths are written but unexecuted here.
- Recommendation lift over popularity is modest (+4.6%). This is honest for data
  where global demand dominates individual taste by construction; the +880% lift
  over random is the more meaningful signal.
- Aspect polarity accuracy (0.996) is flattered by synthetic reviews being
  composed from the same lexicon the extractor consults.
- The search index is per-replica in-memory; catalogue edits propagate on refresh.


---

## Final verification (from the extracted ZIP)

The ZIP was extracted to a clean directory and run end to end:

```
alembic upgrade head          -> 17 tables created
scripts/seed_database.py      -> 600 products / 1,200 users / 6,000 reviews
                                 136,000 events / 16,743 orders
scripts/train_all.py          -> all 5 models trained
pytest backend/tests          -> 173 passed
validate_project.py           -> 34 passed, 0 required failures, 3 warnings
```

Live API checks against the extracted copy:

| Check | Result |
|---|---|
| `/api/v1/health` | `ok` — database ok, 5/5 models loaded |
| `/api/v1/search?q=wireless headphones for gym` | 293 results in 3.07 ms |
| Typo correction | `wireles headphnes` → `wireless headphones` |
| `/api/v1/recommendations/homepage` | 6 sections, personalized |
| `/api/v1/analytics/dashboard` | revenue $1,170,312 · 1,780 orders · rec CTR 5.4% · search CTR 42.1% |
| forecast / price / sentiment / segments / FBT / similar | all HTTP 200 |
| OpenAPI | 55 paths |

The three warnings are expected for a source package: Redis not running,
`frontend/dist` not built, `node_modules` not installed.
