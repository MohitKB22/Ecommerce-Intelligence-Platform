# Deployment

## Docker Compose (recommended)

```bash
cp .env.example .env
# edit .env: set SECRET_KEY and ADMIN_PASSWORD at minimum
docker compose up --build
```

Services: `postgres`, `redis`, `migrate` (one-shot), `backend`, `worker`,
`frontend`, plus optional `vector-db` and `gateway` profiles.

Startup order is enforced by health checks and
`depends_on: {migrate: {condition: service_completed_successfully}}`, so the API
only accepts traffic after migrations, seeding and training have finished.

```bash
docker compose ps                    # status
docker compose logs -f backend       # follow logs
docker compose down                  # stop
docker compose down -v               # stop and delete volumes
```

### Optional profiles

```bash
docker compose --profile vector up -d      # pgvector service
docker compose --profile gateway up -d     # single entrypoint on :8080
```

Set `VECTOR_DB_URL` to activate pgvector; without it the exact numpy index is used.

### Development overlay

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Mounts source, enables uvicorn `--reload` and Vite HMR, switches to
human-readable logs and disables rate limiting.

---

## Production checklist

**Required before going live.** The application refuses to start in
`ENVIRONMENT=production` if any of these are wrong:

- [ ] `SECRET_KEY` set to a strong random value — `openssl rand -hex 32`
- [ ] `ADMIN_PASSWORD` changed from the default
- [ ] `DATABASE_URL` points at PostgreSQL (SQLite is rejected)
- [ ] `DEBUG=false`

**Strongly recommended:**

- [ ] `CORS_ORIGINS` restricted to your real origins
- [ ] TLS terminated at a load balancer or the gateway
- [ ] `REDIS_URL` configured — the in-process cache does not survive multiple replicas
- [ ] Database backups scheduled
- [ ] `LOG_JSON=true` and logs shipped to your aggregator
- [ ] Health checks wired to `/api/v1/health/ready`
- [ ] Resource limits set on containers

---

## Scaling

**Backend** is stateless — scale horizontally behind a load balancer.

```bash
docker compose up -d --scale backend=4
```

Two caveats when running multiple replicas:

1. **Redis becomes mandatory.** The in-process cache is per-replica, so rate
   limits and cached profiles would diverge.
2. **The search index is per-replica.** Each holds its own copy, rebuilt on
   startup and on schedule. That is fine (it is derived state), but a catalogue
   edit propagates only after each replica refreshes. Call the worker task
   `refresh_search_index` or restart replicas for an immediate rebuild.

**Workers** — with `CELERY_BROKER_URL` set, run Celery workers and a beat
scheduler. Without it, run exactly one `worker` container; the in-process
scheduler is not distributed and duplicates would double-write features.

**Database** — the indexes needed for the hot paths ship in the initial
migration. Add read replicas before sharding; analytics queries are the heaviest
consumers and are cached for 120s.

---

## Zero-downtime model updates

Models are versioned artifacts, not code:

```bash
docker compose exec backend python scripts/train_all.py
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/ml/reload
```

`/ml/reload` swaps artifacts in place. The worker also polls for new versions on
a 15-minute cadence. Roll back by promoting an earlier version:

```python
from ml.registry import get_registry
get_registry().promote("recommendation", "v20260814041922")
```

---

## Kubernetes sketch

The containers are Kubernetes-ready: non-root user, health endpoints, no local
state beyond mounted volumes.

- `backend` → Deployment, `readinessProbe: /api/v1/health/ready`,
  `livenessProbe: /api/v1/health/live`
- `worker` → Deployment with `replicas: 1` (or Celery + a separate beat pod)
- `migrate` → Job or initContainer
- `frontend` → Deployment + Service, or ship `dist/` to a CDN
- Config via ConfigMap; `SECRET_KEY`, database and admin credentials via Secret
- Model artifacts on a ReadWriteMany PVC, or bake them into the image at build

---

## Monitoring

- `/api/v1/health` — dependency status, suitable for an uptime check
- `/api/v1/ml/health` — per-model latency, error rate, call volume
- `/api/v1/ml/drift` — feature drift against training baselines
- Structured JSON logs include `request_id`, `method`, `path`, `status`,
  `duration_ms`, so p95 latency and error rate are derivable from logs alone

`infrastructure/monitoring/` is reserved for Prometheus/Grafana configuration.

---

## Backups

```bash
# backup
docker compose exec postgres pg_dump -U eci ecommerce_intelligence | gzip > backup.sql.gz

# restore
gunzip -c backup.sql.gz | docker compose exec -T postgres psql -U eci ecommerce_intelligence
```

Model artifacts live in the `model_data` volume. They are reproducible from the
database via `scripts/train_all.py`, so backing them up is an optimisation rather
than a necessity — but doing so makes rollbacks instant.
