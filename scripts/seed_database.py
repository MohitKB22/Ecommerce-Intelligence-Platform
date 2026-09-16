#!/usr/bin/env python3
"""Populate the database with the deterministic synthetic dataset.

    python scripts/seed_database.py [--reset] [--users N] [--products N] [--days N]

All generated records are SYNTHETIC and are produced by the simulation in
`ml/datasets/synthetic.py`. No real customer or retailer data is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.core.db import Base, SessionLocal, create_all, engine  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models import (  # noqa: E402
    Brand,
    Category,
    Order,
    OrderItem,
    Product,
    RecommendationLog,
    Review,
    SearchEvent,
    User,
    UserEvent,
)

from ml.datasets.synthetic import GeneratorConfig, generate_dataset  # noqa: E402

logger = get_logger("seed")
CHUNK = 2000


def _to_dt(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.to_pydatetime()


def _bulk(session, model, rows: list[dict], label: str) -> int:
    total = 0
    for start in range(0, len(rows), CHUNK):
        session.bulk_insert_mappings(model, rows[start:start + CHUNK])
        session.flush()
        total += len(rows[start:start + CHUNK])
    session.commit()
    logger.info("seed_table_loaded", table=label, rows=total)
    return total


def reset_schema() -> None:
    logger.warning("dropping_all_tables")
    Base.metadata.drop_all(bind=engine)
    create_all()


def seed(cfg: GeneratorConfig, reset: bool, export_csv: bool = True) -> dict:
    create_all()
    if reset:
        reset_schema()

    session = SessionLocal()
    try:
        if session.query(Product).count() > 0 and not reset:
            logger.warning("database_already_seeded_skipping", hint="pass --reset to rebuild")
            return {"skipped": True}

        logger.info("generating_synthetic_dataset", users=cfg.n_users, products=cfg.n_products,
                    days=cfg.n_days, seed=cfg.seed, dataset_version=cfg.dataset_version)
        ds = generate_dataset(cfg)
        summary = ds.summary()

        _bulk(session, Category, [
            {"id": int(r.id), "slug": r.slug, "name": r.name,
             "parent_id": int(r.parent_id) if pd.notna(r.parent_id) else None,
             "description": r.description}
            for r in ds.categories.itertuples()
        ], "categories")

        _bulk(session, Brand, [
            {"id": int(r.id), "slug": r.slug, "name": r.name, "reputation_score": float(r.reputation_score)}
            for r in ds.brands.itertuples()
        ], "brands")

        _bulk(session, Product, [
            {"id": int(r.id), "sku": r.sku, "title": r.title, "description": r.description,
             "category_id": int(r.category_id), "brand_id": int(r.brand_id), "price": float(r.price),
             "base_cost": float(r.base_cost), "discount_pct": float(r.discount_pct), "currency": r.currency,
             "rating_avg": float(r.rating_avg), "rating_count": int(r.rating_count),
             "inventory": int(r.inventory), "is_active": bool(r.is_active), "image_url": r.image_url,
             "specifications": r.specifications, "tags": r.tags, "launched_at": _to_dt(r.launched_at)}
            for r in ds.products.itertuples()
        ], "products")

        admin_hash = hash_password(settings.ADMIN_PASSWORD)
        demo_hash = hash_password("demo-password")
        users = [
            {"id": int(r.id), "email": r.email, "full_name": r.full_name, "role": "user",
             "password_hash": demo_hash, "is_active": bool(r.is_active), "country": r.country,
             "signup_source": r.signup_source, "created_at": _to_dt(r.created_at),
             "updated_at": _to_dt(r.created_at),
             "last_active_at": _to_dt(r.last_active_at) if pd.notna(r.last_active_at) else None,
             "preferred_categories": r.preferred_categories, "preferred_brands": r.preferred_brands,
             "price_sensitivity": float(r.price_sensitivity)}
            for r in ds.users.itertuples()
        ]
        admin_id = len(users) + 1
        users.append({
            "id": admin_id, "email": settings.ADMIN_EMAIL, "full_name": "Platform Administrator",
            "role": "admin", "password_hash": admin_hash, "is_active": True, "country": "US",
            "signup_source": "internal", "preferred_categories": [], "preferred_brands": [],
            "price_sensitivity": 0.5,
        })
        _bulk(session, User, users, "users")

        _bulk(session, Order, [
            {"id": int(r.id), "user_id": int(r.user_id), "order_number": r.order_number, "status": r.status,
             "total_amount": float(r.total_amount), "discount_amount": float(r.discount_amount),
             "item_count": int(r.item_count), "currency": r.currency, "channel": r.channel,
             "placed_at": _to_dt(r.placed_at), "created_at": _to_dt(r.placed_at),
             "updated_at": _to_dt(r.placed_at)}
            for r in ds.orders.itertuples()
        ], "orders")

        _bulk(session, OrderItem, [
            {"id": int(r.id), "order_id": int(r.order_id), "product_id": int(r.product_id),
             "quantity": int(r.quantity), "unit_price": float(r.unit_price),
             "discount_pct": float(r.discount_pct), "line_total": float(r.line_total)}
            for r in ds.order_items.itertuples()
        ], "order_items")

        _bulk(session, Review, [
            {"id": int(r.id), "product_id": int(r.product_id), "user_id": int(r.user_id), "rating": int(r.rating),
             "title": r.title, "body": r.body, "verified_purchase": bool(r.verified_purchase),
             "helpful_votes": int(r.helpful_votes), "created_at": _to_dt(r.created_at),
             "updated_at": _to_dt(r.created_at)}
            for r in ds.reviews.itertuples()
        ], "reviews")

        _bulk(session, UserEvent, [
            {"id": int(r.id), "user_id": int(r.user_id), "session_id": r.session_id, "event_type": r.event_type,
             "product_id": int(r.product_id) if pd.notna(r.product_id) else None, "quantity": int(r.quantity),
             "value": float(r.value), "source": r.source, "event_metadata": r.metadata,
             "occurred_at": _to_dt(r.occurred_at)}
            for r in ds.events.itertuples()
        ], "user_events")

        _bulk(session, SearchEvent, [
            {"id": int(r.id), "user_id": int(r.user_id), "session_id": r.session_id, "query": r.query,
             "normalized_query": r.normalized_query, "result_count": int(r.result_count),
             "clicked_product_id": int(r.clicked_product_id) if pd.notna(r.clicked_product_id) else None,
             "clicked_position": int(r.clicked_position) if pd.notna(r.clicked_position) else None,
             "converted": bool(r.converted), "latency_ms": float(r.latency_ms), "filters": r.filters,
             "occurred_at": _to_dt(r.occurred_at)}
            for r in ds.search_events.itertuples()
        ], "search_events")

        _bulk(session, RecommendationLog, [
            {"id": int(r.id), "user_id": int(r.user_id), "product_id": int(r.product_id), "surface": r.surface,
             "strategy": r.strategy, "model_version": r.model_version, "score": float(r.score),
             "rank": int(r.rank), "explanation": r.explanation, "clicked": bool(r.clicked),
             "converted": bool(r.converted), "served_at": _to_dt(r.served_at)}
            for r in ds.impressions.itertuples()
        ], "recommendations")

        if export_csv:
            seed_dir = REPO_ROOT / "data" / "seed"
            seed_dir.mkdir(parents=True, exist_ok=True)
            ds.price_history.to_csv(seed_dir / "price_history.csv", index=False)
            ds.daily_demand.to_csv(seed_dir / "daily_demand.csv", index=False)
            ds.reviews[["id", "product_id", "rating", "aspects_truth"]].to_json(
                seed_dir / "review_aspect_truth.json", orient="records"
            )
            (seed_dir / "dataset_manifest.json").write_text(
                json.dumps({"synthetic": True, "generator": "ml/datasets/synthetic.py", **summary}, indent=2),
                encoding="utf-8",
            )
            logger.info("seed_artifacts_written", path=str(seed_dir))

        summary["admin_email"] = settings.ADMIN_EMAIL
        logger.info("seed_complete", **{k: v for k, v in summary.items() if isinstance(v, (int, str))})
        return summary
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the E-Commerce Intelligence database with synthetic data.")
    parser.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    parser.add_argument("--users", type=int, default=1200)
    parser.add_argument("--products", type=int, default=600)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--reviews", type=int, default=6000)
    parser.add_argument("--events", type=int, default=130000)
    parser.add_argument("--searches", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=20240613)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    configure_logging(settings.LOG_LEVEL, json_output=False)
    cfg = GeneratorConfig(
        n_users=args.users, n_products=args.products, n_days=args.days, seed=args.seed,
        target_reviews=args.reviews, target_browse_events=args.events, target_search_events=args.searches,
    )
    result = seed(cfg, reset=args.reset, export_csv=not args.no_csv)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
