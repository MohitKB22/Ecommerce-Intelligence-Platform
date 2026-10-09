# Troubleshooting

## Installation

**`error: externally-managed-environment`** (macOS Homebrew Python, recent Linux)
The system interpreter refuses `pip install` (PEP 668). `make install` creates
`.venv/` and installs there; every other `make` target then uses `.venv/bin/python`.
If you prefer your own environment, activate it first and `make` will use it.

**`No matching distribution found for numpy==2.1.1`** (or pandas/scipy)
The interpreter is too old. Apple's `/usr/bin/python3` is 3.9; this project
needs 3.10 - 3.14. Install one (`brew install python@3.12`), delete any
half-built `.venv/`, and run `make install` again.

**`.venv` was created with the wrong Python**
`rm -rf .venv && make install PYTHON_CANDIDATES=python3.12`

## Startup

**`RuntimeError: Refusing to start in production with insecure configuration`**
Working as designed. Set `SECRET_KEY`, change `ADMIN_PASSWORD`, point
`DATABASE_URL` at PostgreSQL and set `DEBUG=false` — or use
`ENVIRONMENT=development` locally.

**`ModuleNotFoundError: No module named 'app'` / `'ml'`**
Both the repository root and `backend/` must be importable:

```bash
export PYTHONPATH="$PWD:$PWD/backend"
```

The Makefile and Docker images set this already.

**`sqlite3.OperationalError: disk I/O error`**
The filesystem does not support the locking SQLite needs — common on network
shares, some FUSE mounts and certain container bind mounts. The engine already
degrades WAL → TRUNCATE → DELETE automatically; if it still fails, move the
database to a local disk:

```bash
DATABASE_URL=sqlite:////tmp/eci/ecommerce.db
```

**Port already allocated**
Change `FRONTEND_PORT` / `BACKEND_PORT` / `POSTGRES_PORT` in `.env`.

---

## Data

**Storefront is empty / `total: 0`**

```bash
make seed          # or: make reset  to rebuild from scratch
```

**"Database already seeded, skipping"**
Intentional idempotence. Use `python scripts/seed_database.py --reset`.

**Seeding is slow**
Default volume is ~136k events. Reduce it:

```bash
python scripts/seed_database.py --reset --users 300 --products 150 --events 20000
```

---

## Models

**`503 model_unavailable`**
The model has not been trained. This is a clean, expected state — the storefront
keeps working with popularity-based fallbacks.

```bash
make train                              # all five
python -m ml.training.train_forecasting # or just one
```

**`No product has enough demand history to train a forecaster`**
The forecaster needs SKUs with ≥25 units across ≥20 weeks. A small or short
dataset will not qualify. Seed a longer window:

```bash
python scripts/seed_database.py --reset --days 365
```

**Training finishes but the API still says unavailable**
`MODEL_PATH` differs between the training process and the API, or the artifact is
cached. Confirm both use the same path, then hot-reload:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/ml/reload
```

**Recommendation quality looks poor**
Check `lift_vs_popularity_ndcg@10` in `/api/v1/ml/registry`. If it is negative,
the fitted weights did not generalise — usually too few post-cutoff interactions.
More event data fixes it; the pipeline falls back to popularity-only weights when
the validation window is too small to fit on.

---

## API

**401 on every admin endpoint**
Obtain a token first and check the account has `role=admin`:

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@ecommerce-intelligence.local","password":"admin-change-me"}'
```

**403 with a valid token**
The token belongs to a non-admin user. Admin analytics require `role=admin`.

**422 on login**
The address must contain `@` and a dot. Note that deliverability is *not*
checked — `.invalid` and `.local` addresses are accepted by design.

**429 rate limited**
Default is 300 requests / 60s per client. Raise `RATE_LIMIT_REQUESTS` or set
`RATE_LIMIT_ENABLED=false` for load testing.

**409 conflict when creating a review**
One review per customer per product, enforced by a unique index.

---

## Frontend

**"Could not reach the API. Is the backend running?"**
Start the backend on :8000. In dev the Vite proxy forwards `/api`; if you set
`VITE_API_BASE_URL` to a different origin, that origin must appear in
`CORS_ORIGINS`.

**`error TS6310: Referenced project may not disable emit`**
`tsconfig.node.json` must not set `noEmit` while `composite: true`. The shipped
config is already correct — this appears only if it was edited.

**`'test' does not exist in type 'UserConfigExport'`**
`vite.config.ts` must import `defineConfig` from `vitest/config`, not `vite`.

**Build succeeds but the page is blank**
Check the browser console. When served under a sub-path, set Vite's `base`.

---

## Docker

**`migrate` exits non-zero**
Inspect it directly — usually PostgreSQL was not ready or credentials disagree:

```bash
docker compose logs migrate
```

**Backend restarts continuously**
`docker compose logs backend`. Common causes: unreachable database, a
production-config refusal, or a port conflict.

**Changes not reflected**
Images cache aggressively:

```bash
docker compose build --no-cache && docker compose up -d
```

---

## Diagnostics

```bash
make validate                        # full project audit
python scripts/validate_project.py --json
curl http://localhost:8000/api/v1/health | python3 -m json.tool
python -m app.workers.runner --list  # available background jobs
python -m pytest backend/tests -v    # full test suite
```

`make validate` checks structure, configuration, imports, database and cache
connectivity, model availability and live API behaviour, and exits non-zero on
any required failure.
