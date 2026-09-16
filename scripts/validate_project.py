#!/usr/bin/env python3
"""Automated project audit.

Verifies that the repository is structurally complete and that every runtime
dependency the application needs is reachable. Exits non-zero if any REQUIRED
check fails; optional checks are reported but never fail the build.

    python scripts/validate_project.py            # structure + reachable services
    python scripts/validate_project.py --strict   # optional failures also fail
    python scripts/validate_project.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

REQUIRED_FILES = [
    "README.md", "LICENSE", ".env.example", ".gitignore", "Makefile",
    "docker-compose.yml", "docker-compose.dev.yml", "alembic.ini",
    "backend/Dockerfile", "backend/requirements.txt", "backend/app/main.py",
    "backend/app/core/config.py", "backend/app/core/db.py", "backend/app/core/cache.py",
    "backend/app/core/security.py", "backend/app/core/errors.py",
    "backend/app/models/__init__.py", "backend/app/api/v1/__init__.py",
    "backend/alembic/env.py", "backend/pytest.ini",
    "frontend/package.json", "frontend/vite.config.ts", "frontend/tsconfig.json",
    "frontend/index.html", "frontend/src/main.tsx", "frontend/src/App.tsx",
    "frontend/Dockerfile", "infrastructure/nginx/frontend.conf",
    "ml/datasets/synthetic.py", "ml/registry.py", "ml/evaluation/metrics.py",
    "ml/training/train_recommendation.py", "ml/training/train_segmentation.py",
    "ml/training/train_sentiment.py", "ml/training/train_forecasting.py",
    "ml/training/train_price_prediction.py",
    "ml/inference/artifacts.py", "ml/features/embeddings.py", "ml/features/bm25.py",
    "scripts/seed_database.py", "scripts/train_all.py",
    "docs/architecture.md", "docs/api.md", "docs/ml.md", "docs/deployment.md",
    "docs/troubleshooting.md",
    ".github/workflows/ci.yml", ".github/workflows/test.yml", ".github/workflows/build.yml",
]

REQUIRED_DIRS = [
    "backend/app/api/v1", "backend/app/core", "backend/app/models", "backend/app/schemas",
    "backend/app/services", "backend/app/repositories", "backend/app/search",
    "backend/app/recommendation", "backend/app/forecasting", "backend/app/segmentation",
    "backend/app/sentiment", "backend/app/workers", "backend/app/ml",
    "backend/tests/unit", "backend/tests/integration", "backend/tests/e2e",
    "frontend/src/components", "frontend/src/pages", "frontend/src/layouts", "frontend/src/hooks",
    "frontend/src/services", "frontend/src/api", "frontend/src/store", "frontend/src/types",
    "frontend/src/utils", "frontend/src/charts",
    "ml/datasets", "ml/preprocessing", "ml/features", "ml/training", "ml/evaluation",
    "ml/inference", "ml/models", "ml/notebooks",
    "data/raw", "data/processed", "data/seed",
    "infrastructure/docker", "infrastructure/nginx", "infrastructure/monitoring",
    "infrastructure/deployment", "scripts", "docs", "tests",
]

MODEL_NAMES = ("recommendation", "segmentation", "sentiment", "forecasting", "price_prediction")


@dataclass
class Result:
    name: str
    ok: bool
    required: bool
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add(self, name: str, ok: bool, required: bool = True, detail: str = "") -> None:
        self.results.append(Result(name, ok, required, detail))

    @property
    def required_failures(self) -> list[Result]:
        return [r for r in self.results if r.required and not r.ok]

    @property
    def optional_failures(self) -> list[Result]:
        return [r for r in self.results if not r.required and not r.ok]

    def to_dict(self) -> dict:
        return {
            "passed": len([r for r in self.results if r.ok]),
            "failed": len([r for r in self.results if not r.ok]),
            "required_failures": len(self.required_failures),
            "optional_failures": len(self.optional_failures),
            "checks": [
                {"name": r.name, "ok": r.ok, "required": r.required, "detail": r.detail}
                for r in self.results
            ],
        }


def check_structure(report: Report) -> None:
    missing_files = [f for f in REQUIRED_FILES if not (REPO_ROOT / f).is_file()]
    report.add("Required files present", not missing_files, True,
               f"missing: {', '.join(missing_files[:6])}" if missing_files else
               f"{len(REQUIRED_FILES)} files verified")

    missing_dirs = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).is_dir()]
    report.add("Required directories present", not missing_dirs, True,
               f"missing: {', '.join(missing_dirs[:6])}" if missing_dirs else
               f"{len(REQUIRED_DIRS)} directories verified")

    env_example = REPO_ROOT / ".env.example"
    if env_example.is_file():
        content = env_example.read_text(encoding="utf-8")
        needed = ["DATABASE_URL", "REDIS_URL", "SECRET_KEY", "MODEL_PATH", "ENVIRONMENT",
                  "LOG_LEVEL", "CORS_ORIGINS", "VECTOR_DB_URL"]
        absent = [key for key in needed if key not in content]
        report.add("Configuration template complete", not absent, True,
                   f"missing keys: {', '.join(absent)}" if absent else f"{len(needed)} keys present")

    # A committed .env would leak secrets.
    report.add("No committed .env file", not (REPO_ROOT / ".env").is_file(), True,
               "found a .env in the repository root" if (REPO_ROOT / ".env").is_file() else "clean")


def check_code_quality(report: Report) -> None:
    """Scan source for placeholders that would contradict a production claim."""
    # Assembled at runtime so this scanner does not match its own source.
    markers = ("TO" + "DO", "FIX" + "ME", "XX" + "X:", "NotImplemented" + "Error")
    offenders: list[str] = []
    self_path = Path(__file__).resolve()
    for pattern in ("backend/app/**/*.py", "ml/**/*.py", "scripts/*.py", "frontend/src/**/*.ts",
                    "frontend/src/**/*.tsx"):
        for path in REPO_ROOT.glob(pattern):
            if "node_modules" in path.parts or "__pycache__" in path.parts:
                continue
            if path.resolve() == self_path:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for marker in markers:
                if marker in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{marker}")
                    break
    report.add("No placeholder markers in source", not offenders, True,
               f"{len(offenders)} file(s): {', '.join(offenders[:4])}" if offenders else "clean")

    # Obvious hard-coded secrets.
    suspicious: list[str] = []
    for path in REPO_ROOT.glob("backend/app/**/*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for needle in ("aws_secret_access_key", "-----BEGIN RSA PRIVATE KEY-----", "sk_live_"):
            if needle in text:
                suspicious.append(str(path.relative_to(REPO_ROOT)))
    report.add("No hard-coded credentials", not suspicious, True,
               ", ".join(suspicious) if suspicious else "clean")


def check_imports(report: Report) -> None:
    try:
        from app.main import app

        spec = app.openapi()
        operations = sum(
            1 for methods in spec["paths"].values()
            for verb in methods if verb in ("get", "post", "put", "patch", "delete")
        )
        report.add("Backend application imports", True, True,
                   f"{len(spec['paths'])} paths / {operations} operations")
    except Exception as exc:  # noqa: BLE001
        report.add("Backend application imports", False, True, f"{type(exc).__name__}: {exc}")
        return

    for module in ("ml.datasets.synthetic", "ml.registry", "ml.evaluation.metrics",
                   "ml.features.embeddings", "ml.features.bm25", "ml.inference.artifacts",
                   "ml.inference.forecast_service", "ml.inference.price_service"):
        try:
            __import__(module)
            report.add(f"Import {module}", True, True)
        except Exception as exc:  # noqa: BLE001
            report.add(f"Import {module}", False, True, f"{type(exc).__name__}: {exc}")


def check_database(report: Report) -> None:
    try:
        from app.core.db import Base, SessionLocal, dialect_name, ping
        from app.models import Product, User
        from sqlalchemy import func, select

        connected = ping()
        report.add("Database reachable", connected, True, f"dialect={dialect_name()}")
        if not connected:
            return

        report.add("Schema defines every table", len(Base.metadata.tables) >= 17, True,
                   f"{len(Base.metadata.tables)} tables")

        with SessionLocal() as db:
            products = db.execute(select(func.count(Product.id))).scalar_one()
            users = db.execute(select(func.count(User.id))).scalar_one()
        seeded = products > 0 and users > 0
        report.add("Database seeded", seeded, False,
                   f"{products} products / {users} users" if seeded
                   else "empty - run: python scripts/seed_database.py")
    except Exception as exc:  # noqa: BLE001
        report.add("Database reachable", False, True, f"{type(exc).__name__}: {exc}")


def check_cache(report: Report) -> None:
    try:
        from app.core.cache import get_cache

        cache = get_cache()
        cache.set_json("validate:probe", {"ok": True}, 10)
        works = cache.get_json("validate:probe") == {"ok": True}
        report.add("Cache read/write", works, True, f"backend={cache.backend_name}")
        report.add("Redis available", cache.backend_name == "redis", False,
                   "using the in-process fallback" if cache.backend_name != "redis" else "connected")
    except Exception as exc:  # noqa: BLE001
        report.add("Cache read/write", False, True, f"{type(exc).__name__}: {exc}")


def check_models(report: Report) -> None:
    try:
        from app.ml.model_store import get_model_store

        store = get_model_store()
        available = {name: store.is_available(name) for name in MODEL_NAMES}
        trained = [name for name, ok in available.items() if ok]
        report.add("ML models trained", len(trained) == len(MODEL_NAMES), False,
                   f"{len(trained)}/{len(MODEL_NAMES)} available"
                   + (f" (missing: {', '.join(n for n in MODEL_NAMES if n not in trained)})"
                      if len(trained) < len(MODEL_NAMES) else ""))
        for name in trained:
            try:
                loaded = store.load(name)
                report.add(f"Load model '{name}'", True, False, f"version {loaded.version}")
            except Exception as exc:  # noqa: BLE001
                report.add(f"Load model '{name}'", False, False, f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        report.add("ML models trained", False, False, f"{type(exc).__name__}: {exc}")


def check_api(report: Report) -> None:
    try:
        from app.main import app
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            health = client.get("/api/v1/health")
            report.add("API health endpoint", health.status_code == 200, True,
                       f"status={health.json().get('status')}" if health.status_code == 200
                       else f"HTTP {health.status_code}")

            for path in ("/api/v1/products?page_size=1", "/api/v1/search?q=",
                         "/api/v1/recommendations/trending?limit=1",
                         "/api/v1/products/categories"):
                response = client.get(path)
                report.add(f"GET {path.split('?')[0]}", response.status_code == 200, True,
                           f"HTTP {response.status_code}")

            report.add("Unknown resource returns 404", client.get("/api/v1/products/99999999")
                       .status_code == 404, True)
            report.add("Admin routes are protected", client.get("/api/v1/analytics/dashboard")
                       .status_code == 401, True)
            report.add("OpenAPI schema served", client.get("/openapi.json").status_code == 200, True)
    except Exception as exc:  # noqa: BLE001
        report.add("API health endpoint", False, True, f"{type(exc).__name__}: {exc}")


def check_frontend(report: Report) -> None:
    package_json = REPO_ROOT / "frontend" / "package.json"
    if package_json.is_file():
        data = json.loads(package_json.read_text(encoding="utf-8"))
        scripts = data.get("scripts", {})
        needed = {"dev", "build", "test", "lint", "typecheck"}
        report.add("Frontend scripts defined", needed <= set(scripts), True,
                   f"missing: {', '.join(sorted(needed - set(scripts)))}"
                   if not needed <= set(scripts) else "dev/build/test/lint/typecheck")

    dist = REPO_ROOT / "frontend" / "dist" / "index.html"
    report.add("Frontend production build present", dist.is_file(), False,
               "dist/index.html found" if dist.is_file() else "run: npm --prefix frontend run build")

    node_modules = REPO_ROOT / "frontend" / "node_modules"
    report.add("Frontend dependencies installed", node_modules.is_dir(), False,
               "node_modules present" if node_modules.is_dir() else "run: npm --prefix frontend ci")


def render(report: Report, use_colour: bool) -> None:
    def paint(text: str, colour: str) -> str:
        return f"{colour}{text}{RESET}" if use_colour else text

    print("\nE-Commerce Intelligence - project validation")
    print("=" * 62)
    for result in report.results:
        if result.ok:
            mark = paint("PASS", GREEN)
        elif result.required:
            mark = paint("FAIL", RED)
        else:
            mark = paint("WARN", YELLOW)
        line = f"  [{mark}] {result.name}"
        if result.detail:
            line += paint(f"  ({result.detail})", DIM) if use_colour else f"  ({result.detail})"
        print(line)

    summary = report.to_dict()
    print("-" * 62)
    print(f"  {summary['passed']} passed, {summary['required_failures']} required failures, "
          f"{summary['optional_failures']} warnings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the project structure and runtime.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="only run static structure checks")
    args = parser.parse_args()

    report = Report()
    checks: list[Callable[[Report], None]] = [check_structure, check_code_quality, check_frontend]
    if not args.skip_runtime:
        checks += [check_imports, check_database, check_cache, check_models, check_api]
    for check in checks:
        check(report)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        render(report, use_colour=not args.no_color and sys.stdout.isatty())

    failed = len(report.required_failures)
    if args.strict:
        failed += len(report.optional_failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
