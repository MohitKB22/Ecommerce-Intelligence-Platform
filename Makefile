# E-Commerce Intelligence - developer entry points.
# Every target below is expected to work from a clean checkout.

SHELL := /bin/bash
PY    ?= python3
PIP   ?= $(PY) -m pip
NPM   ?= npm
export PYTHONPATH := $(CURDIR):$(CURDIR)/backend

.DEFAULT_GOAL := help
.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend \
        migrate migration seed train train-% test test-backend test-frontend test-e2e \
        lint lint-backend lint-frontend format typecheck build validate worker \
        docker-up docker-down docker-dev docker-logs docker-rebuild clean reset

help: ## Show this help
	@grep -hE '^[a-zA-Z_%-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- setup ---------------------------------------------------------------
install: install-backend install-frontend ## Install all dependencies

install-backend: ## Install Python dependencies (including dev tools)
	$(PIP) install -r backend/requirements-dev.txt

install-frontend: ## Install frontend dependencies
	cd frontend && $(NPM) install

# ---- database ------------------------------------------------------------
migrate: ## Apply database migrations
	$(PY) -m alembic upgrade head

migration: ## Autogenerate a migration: make migration m="add x"
	$(PY) -m alembic revision --autogenerate -m "$(m)"

seed: ## Load the synthetic dataset (use ARGS="--reset" to rebuild)
	$(PY) scripts/seed_database.py $(ARGS)

reset: ## Drop, recreate and reseed the database
	$(PY) scripts/seed_database.py --reset

# ---- machine learning ----------------------------------------------------
train: ## Train every model
	$(PY) scripts/train_all.py --continue-on-error

train-%: ## Train one model, e.g. make train-recommendation
	$(PY) -m ml.training.train_$*

# ---- run -----------------------------------------------------------------
dev: ## Print instructions for running both services
	@echo "Run these in two terminals:"
	@echo "  make dev-backend    -> http://localhost:8000/docs"
	@echo "  make dev-frontend   -> http://localhost:3000"

dev-backend: ## Start the API with auto-reload
	$(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend

dev-frontend: ## Start the Vite dev server
	cd frontend && $(NPM) run dev

worker: ## Run background jobs once and exit
	$(PY) -m app.workers.runner --once

# ---- quality -------------------------------------------------------------
test: test-backend test-frontend ## Run every test suite

test-backend: ## Run the Python test suite
	$(PY) -m pytest backend/tests -v

test-e2e: ## Run only the end-to-end journeys
	$(PY) -m pytest backend/tests/e2e -v

test-frontend: ## Run the frontend test suite
	cd frontend && $(NPM) run test

lint: lint-backend lint-frontend ## Lint everything

lint-backend: ## Lint Python
	$(PY) -m ruff check backend/app ml scripts

lint-frontend: ## Lint TypeScript
	cd frontend && $(NPM) run lint

format: ## Auto-format Python
	$(PY) -m ruff check --fix backend/app ml scripts
	$(PY) -m ruff format backend/app ml scripts

typecheck: ## Type-check both sides
	$(PY) -m mypy backend/app --ignore-missing-imports || true
	cd frontend && $(NPM) run typecheck

build: ## Build the frontend for production
	cd frontend && $(NPM) run build

validate: ## Run the full project audit
	$(PY) scripts/validate_project.py

# ---- docker --------------------------------------------------------------
docker-up: ## Start the whole stack
	docker compose up --build -d
	@echo "Frontend: http://localhost:3000   API docs: http://localhost:8000/docs"

docker-dev: ## Start the stack with hot reload
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

docker-down: ## Stop the stack
	docker compose down

docker-rebuild: ## Rebuild images from scratch and restart
	docker compose down -v
	docker compose build --no-cache
	docker compose up -d

docker-logs: ## Tail service logs
	docker compose logs -f --tail=100

# ---- housekeeping --------------------------------------------------------
clean: ## Remove caches and build output
	find . -type d -name __pycache__ -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage frontend/dist
	@echo "Cleaned."
