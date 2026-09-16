"""Load training frames out of the operational database.

Training always reads from the same database the API serves from, so a model is
never trained on data the application cannot reproduce at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def _read(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def load_products(engine: Engine) -> pd.DataFrame:
    df = _read(engine, """
        SELECT p.id, p.sku, p.title, p.description, p.category_id, p.brand_id, p.price, p.base_cost,
               p.discount_pct, p.rating_avg, p.rating_count, p.inventory, p.is_active, p.tags,
               p.specifications, p.launched_at, c.slug AS category_slug, c.name AS category_name,
               b.name AS brand_name, b.reputation_score AS brand_reputation
        FROM products p
        JOIN categories c ON c.id = p.category_id
        JOIN brands b ON b.id = p.brand_id
    """)
    for col in ("tags", "specifications"):
        if col in df.columns:
            df[col] = df[col].apply(_coerce_json)
    if "launched_at" in df.columns:
        df["launched_at"] = pd.to_datetime(df["launched_at"], errors="coerce", utc=True)
    return df


def _coerce_json(value):
    if isinstance(value, (list, dict)) or value is None:
        return value
    import json

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def load_users(engine: Engine) -> pd.DataFrame:
    df = _read(engine, """
        SELECT id, email, full_name, role, is_active, country, signup_source, created_at, last_active_at,
               preferred_categories, preferred_brands, price_sensitivity
        FROM users
    """)
    for col in ("preferred_categories", "preferred_brands"):
        if col in df.columns:
            df[col] = df[col].apply(_coerce_json)
    for col in ("created_at", "last_active_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_events(engine: Engine, since: datetime | None = None) -> pd.DataFrame:
    sql = """
        SELECT id, user_id, session_id, event_type, product_id, quantity, value, source, occurred_at
        FROM user_events
    """
    params: dict = {}
    if since is not None:
        sql += " WHERE occurred_at >= :since"
        params["since"] = since
    df = _read(engine, sql, params)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce", utc=True)
    return df.dropna(subset=["occurred_at"])


def load_orders(engine: Engine) -> pd.DataFrame:
    df = _read(engine, """
        SELECT o.id, o.user_id, o.status, o.total_amount, o.discount_amount, o.item_count,
               o.channel, o.placed_at
        FROM orders o
    """)
    df["placed_at"] = pd.to_datetime(df["placed_at"], errors="coerce", utc=True)
    return df


def load_order_items(engine: Engine) -> pd.DataFrame:
    df = _read(engine, """
        SELECT oi.id, oi.order_id, oi.product_id, oi.quantity, oi.unit_price, oi.discount_pct, oi.line_total,
               o.user_id, o.status, o.placed_at, p.category_id, p.brand_id
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        JOIN products p ON p.id = oi.product_id
    """)
    df["placed_at"] = pd.to_datetime(df["placed_at"], errors="coerce", utc=True)
    return df


def load_reviews(engine: Engine) -> pd.DataFrame:
    df = _read(engine, """
        SELECT id, product_id, user_id, rating, title, body, verified_purchase, helpful_votes,
               sentiment_label, sentiment_score, created_at
        FROM reviews
    """)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df


def load_search_events(engine: Engine) -> pd.DataFrame:
    df = _read(engine, """
        SELECT id, user_id, query, normalized_query, result_count, clicked_product_id, clicked_position,
               converted, latency_ms, occurred_at
        FROM search_events
    """)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce", utc=True)
    return df


def load_daily_demand(engine: Engine) -> pd.DataFrame:
    """Units sold per product per day, derived from revenue-bearing orders."""
    df = _read(engine, """
        SELECT oi.product_id AS product_id, o.placed_at AS placed_at, oi.quantity AS quantity,
               oi.unit_price AS unit_price, oi.discount_pct AS discount_pct
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        WHERE o.status IN ('paid','shipped','delivered')
    """)
    if df.empty:
        return pd.DataFrame(columns=["product_id", "date", "units", "revenue", "avg_price"])
    df["placed_at"] = pd.to_datetime(df["placed_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["placed_at"])
    df["date"] = df["placed_at"].dt.floor("D")
    df["revenue"] = df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100.0)
    grouped = df.groupby(["product_id", "date"], as_index=False).agg(
        units=("quantity", "sum"), revenue=("revenue", "sum"), avg_price=("unit_price", "mean")
    )
    return grouped


@dataclass(slots=True)
class TrainingFrames:
    products: pd.DataFrame
    users: pd.DataFrame
    events: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    reviews: pd.DataFrame
    search_events: pd.DataFrame
    daily_demand: pd.DataFrame
    loaded_at: datetime

    def validate(self, require: tuple[str, ...] = ()) -> list[str]:
        """Return a list of data-quality problems. Empty means the data is usable."""
        problems: list[str] = []
        checks = {
            "products": (self.products, 10),
            "users": (self.users, 5),
            "events": (self.events, 50),
            "order_items": (self.order_items, 20),
            "reviews": (self.reviews, 20),
            "daily_demand": (self.daily_demand, 20),
        }
        for name in require:
            frame, minimum = checks.get(name, (pd.DataFrame(), 0))
            if len(frame) < minimum:
                problems.append(f"{name}: only {len(frame)} rows, need at least {minimum}")
        if "products" in require and not self.products.empty:
            if self.products["id"].duplicated().any():
                problems.append("products: duplicate ids")
            if (self.products["price"] < 0).any():
                problems.append("products: negative prices")
        if "reviews" in require and not self.reviews.empty:
            bad = ~self.reviews["rating"].between(1, 5)
            if bad.any():
                problems.append(f"reviews: {int(bad.sum())} ratings outside 1-5")
        return problems


def load_all(engine: Engine) -> TrainingFrames:
    return TrainingFrames(
        products=load_products(engine),
        users=load_users(engine),
        events=load_events(engine),
        orders=load_orders(engine),
        order_items=load_order_items(engine),
        reviews=load_reviews(engine),
        search_events=load_search_events(engine),
        daily_demand=load_daily_demand(engine),
        loaded_at=datetime.now(timezone.utc),
    )
