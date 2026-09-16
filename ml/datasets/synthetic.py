"""Deterministic synthetic e-commerce data generator.

IMPORTANT: every record produced here is SYNTHETIC. It is generated from a
latent-factor simulation written for this project and contains no real customer,
product, pricing or transaction data from any company.

The simulation is deliberately *learnable*: user tastes, product attributes,
seasonality, price elasticity and review sentiment are all driven by explicit
latent structure, so the downstream models have genuine signal to recover rather
than fitting noise.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ml.datasets.taxonomy import (
    ASPECT_LEXICON,
    BRANDS,
    CATEGORIES,
    CATEGORY_ASPECTS,
    COLOURS,
    COMPLEMENTARY,
    LEAF_CATEGORIES,
    MATERIALS,
    QUALIFIERS,
    REVIEW_CLOSERS_NEG,
    REVIEW_CLOSERS_NEU,
    REVIEW_CLOSERS_POS,
    REVIEW_OPENERS_NEG,
    REVIEW_OPENERS_NEU,
    REVIEW_OPENERS_POS,
    SEARCH_INTENT_TEMPLATES,
    USE_CASES,
)

LATENT_DIM = 12
SEGMENT_ARCHETYPES = [
    # (name, activity multiplier, price sensitivity, basket size lambda, review propensity)
    ("high_value", 1.6, 0.20, 2.6, 0.30),
    ("loyal", 1.4, 0.40, 2.0, 0.34),
    ("frequent", 1.5, 0.55, 1.5, 0.22),
    ("discount_seeker", 1.1, 0.90, 1.9, 0.18),
    ("occasional", 0.6, 0.55, 1.4, 0.16),
    ("at_risk", 0.35, 0.60, 1.3, 0.10),
    ("new", 0.8, 0.50, 1.2, 0.12),
]


@dataclass(slots=True)
class GeneratorConfig:
    n_users: int = 1200
    n_products: int = 600
    n_days: int = 365
    seed: int = 20240613
    target_reviews: int = 6000
    target_browse_events: int = 130000
    target_search_events: int = 9000
    end_date: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    )

    @property
    def start_date(self) -> datetime:
        return self.end_date - timedelta(days=self.n_days - 1)

    @property
    def dataset_version(self) -> str:
        payload = f"{self.n_users}-{self.n_products}-{self.n_days}-{self.seed}"
        return "syn-" + hashlib.sha256(payload.encode()).hexdigest()[:10]


@dataclass(slots=True)
class SyntheticDataset:
    """Container for every generated table."""

    categories: pd.DataFrame
    brands: pd.DataFrame
    products: pd.DataFrame
    users: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    reviews: pd.DataFrame
    events: pd.DataFrame
    search_events: pd.DataFrame
    price_history: pd.DataFrame
    daily_demand: pd.DataFrame
    impressions: pd.DataFrame
    dataset_version: str
    config: GeneratorConfig

    def summary(self) -> dict[str, int | str]:
        return {
            "dataset_version": self.dataset_version,
            "categories": len(self.categories),
            "brands": len(self.brands),
            "products": len(self.products),
            "users": len(self.users),
            "orders": len(self.orders),
            "order_items": len(self.order_items),
            "reviews": len(self.reviews),
            "events": len(self.events),
            "search_events": len(self.search_events),
            "price_history_rows": len(self.price_history),
            "daily_demand_rows": len(self.daily_demand),
            "impressions": len(self.impressions),
        }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _build_categories() -> pd.DataFrame:
    rows = []
    slugs = list(CATEGORIES)
    index = {slug: i + 1 for i, slug in enumerate(slugs)}
    for slug in slugs:
        meta = CATEGORIES[slug]
        rows.append(
            {
                "id": index[slug],
                "slug": slug,
                "name": meta["name"],
                "parent_id": index.get(meta["parent"]) if meta["parent"] else None,
                "description": f"{meta['name']} products across the catalogue.",
            }
        )
    return pd.DataFrame(rows)


def _build_brands() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": i + 1, "slug": name.lower().replace(" ", "-"), "name": name, "reputation_score": rep}
            for i, (name, rep) in enumerate(BRANDS)
        ]
    )


def _build_products(rng: np.random.Generator, cfg: GeneratorConfig, categories: pd.DataFrame,
                    brands: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cat_index = {row.slug: row.id for row in categories.itertuples()}
    leaf = LEAF_CATEGORIES
    n = cfg.n_products

    cat_slugs = rng.choice(leaf, size=n, p=_softmax(rng.normal(0, 0.35, len(leaf))))
    brand_ids = rng.integers(1, len(brands) + 1, size=n)

    # Latent product vectors: category centroid + brand offset + product noise.
    cat_centroids = {slug: rng.normal(0, 1.0, LATENT_DIM) for slug in leaf}
    brand_offsets = rng.normal(0, 0.35, (len(brands) + 1, LATENT_DIM))
    latent = np.vstack(
        [cat_centroids[s] + brand_offsets[b] + rng.normal(0, 0.45, LATENT_DIM) for s, b in zip(cat_slugs, brand_ids)]
    )
    latent /= np.linalg.norm(latent, axis=1, keepdims=True) + 1e-9

    # Intrinsic quality drives both ratings and repeat demand.
    brand_rep = brands.set_index("id")["reputation_score"].to_dict()
    quality = np.clip(
        rng.beta(5, 2.2, n) * 0.7 + np.array([brand_rep[b] for b in brand_ids]) * 0.3 + rng.normal(0, 0.04, n),
        0.05,
        0.995,
    )

    # Popularity is heavy-tailed (a few hero SKUs carry most volume).
    popularity = rng.lognormal(mean=0.0, sigma=1.05, size=n)
    popularity *= 0.6 + 0.8 * quality
    popularity /= popularity.mean()

    rows = []
    for i in range(n):
        slug = str(cat_slugs[i])
        meta = CATEGORIES[slug]
        keyword = str(rng.choice(meta["keywords"]))
        brand_name = brands.loc[brands["id"] == brand_ids[i], "name"].iloc[0]
        qualifier = str(rng.choice(QUALIFIERS))
        model_code = f"{rng.integers(100, 999)}"
        title = f"{brand_name} {keyword.title()} {qualifier} {model_code}"

        lo, hi = meta["price"]
        # price correlates with quality and brand reputation, plus noise
        price_pos = np.clip(0.55 * quality[i] + 0.25 * brand_rep[brand_ids[i]] + rng.normal(0, 0.13), 0.02, 0.99)
        price = float(np.round(lo + (hi - lo) * (price_pos ** 1.35), 2))
        base_cost = float(np.round(price * rng.uniform(0.42, 0.68), 2))
        discount = float(np.round(max(0.0, rng.normal(6, 9)), 1)) if rng.random() < 0.42 else 0.0
        discount = min(discount, 55.0)

        material = str(rng.choice(MATERIALS))
        colour = str(rng.choice(COLOURS))
        use_case = str(rng.choice(USE_CASES))
        specs = {
            "material": material,
            "colour": colour,
            "warranty_months": int(rng.choice([6, 12, 24, 36])),
            "weight_g": int(rng.integers(80, 4200)),
            "model": model_code,
        }
        if slug in ("audio", "wearables", "cameras", "laptops"):
            specs["battery_hours"] = int(rng.integers(4, 60))
            specs["wireless"] = bool(rng.random() < 0.75)
        description = (
            f"The {title} is a {colour} {keyword} built from {material}, designed {use_case}. "
            f"It ships with a {specs['warranty_months']} month warranty and is part of the "
            f"{meta['name']} range from {brand_name}. "
            f"{'Wireless connectivity included. ' if specs.get('wireless') else ''}"
            f"Engineered for reliable everyday performance with a focus on "
            f"{'premium materials' if quality[i] > 0.7 else 'practical value'}."
        )
        tags = [keyword, colour, material, meta["name"].lower(), qualifier.lower()]
        if use_case.startswith("for "):
            tags.append(use_case[4:])

        launch_offset = int(rng.integers(0, cfg.n_days + 400))
        launched_at = cfg.end_date - timedelta(days=launch_offset)

        rows.append(
            {
                "id": i + 1,
                "sku": f"ECI-{100000 + i}",
                "title": title,
                "description": description,
                "category_id": cat_index[slug],
                "category_slug": slug,
                "brand_id": int(brand_ids[i]),
                "price": price,
                "base_cost": base_cost,
                "discount_pct": discount,
                "currency": "USD",
                "inventory": int(max(0, rng.negative_binomial(4, 0.02) if rng.random() > 0.06 else 0)),
                "is_active": bool(rng.random() > 0.02),
                "image_url": f"/static/catalog/{slug}/{100000 + i}.jpg",
                "specifications": specs,
                "tags": tags,
                "launched_at": launched_at,
                "quality": float(quality[i]),
                "popularity_base": float(popularity[i]),
                "keyword": keyword,
            }
        )
    return pd.DataFrame(rows), latent


def _build_users(rng: np.random.Generator, cfg: GeneratorConfig, products: pd.DataFrame,
                 product_latent: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    n = cfg.n_users
    archetype_p = np.array([0.08, 0.14, 0.16, 0.14, 0.24, 0.13, 0.11])
    archetype_p /= archetype_p.sum()
    arche_idx = rng.choice(len(SEGMENT_ARCHETYPES), size=n, p=archetype_p)

    latent = rng.normal(0, 1.0, (n, LATENT_DIM))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True) + 1e-9

    first = ["Ada", "Bo", "Cai", "Dara", "Eli", "Fen", "Gia", "Hal", "Ira", "Jun", "Kit", "Lior", "Mira", "Nils",
             "Oona", "Pax", "Quin", "Rhea", "Sol", "Tam", "Uma", "Vik", "Wren", "Xan", "Yara", "Zeke"]
    last = ["Alvarez", "Brennan", "Castille", "Duarte", "Eriksen", "Falk", "Grimaldi", "Haas", "Ibarra", "Jansen",
            "Kowal", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quiroga", "Rosales", "Sandoval",
            "Tanaka", "Ustinov", "Vasquez", "Whitlock", "Xiong", "Yilmaz", "Zubiri"]
    countries = ["US", "GB", "DE", "CA", "AU", "FR", "IN", "JP", "BR", "NL"]
    country_p = np.array([0.34, 0.12, 0.09, 0.08, 0.06, 0.06, 0.09, 0.05, 0.06, 0.05])
    sources = ["organic", "paid_search", "social", "referral", "email"]

    cat_slugs = sorted(products["category_slug"].unique())
    cat_ids = {row.category_slug: row.category_id for row in
               products.drop_duplicates("category_slug").itertuples()}

    # Category affinity derived from the user's latent vector vs category centroids.
    cat_centroid = np.vstack([
        product_latent[(products["category_slug"] == s).to_numpy()].mean(axis=0) for s in cat_slugs
    ])
    cat_affinity = _softmax(latent @ cat_centroid.T * 2.4, axis=1)

    brand_ids = sorted(products["brand_id"].unique())

    rows = []
    for i in range(n):
        name_i = f"{first[rng.integers(0, len(first))]} {last[rng.integers(0, len(last))]}"
        arche = SEGMENT_ARCHETYPES[arche_idx[i]]
        # 'new' users sign up late in the window; 'at_risk' signed up early then went quiet
        if arche[0] == "new":
            signup_day = int(rng.integers(max(0, cfg.n_days - 45), cfg.n_days))
        elif arche[0] == "at_risk":
            signup_day = int(rng.integers(0, max(1, cfg.n_days // 3)))
        else:
            signup_day = int(rng.integers(0, cfg.n_days))
        top_cats = [cat_slugs[j] for j in np.argsort(-cat_affinity[i])[:3]]
        pref_brands = [int(b) for b in rng.choice(brand_ids, size=int(rng.integers(1, 4)), replace=False)]
        rows.append(
            {
                "id": i + 1,
                "email": f"user{i + 1:05d}@example.invalid",
                "full_name": name_i,
                "role": "user",
                "is_active": bool(rng.random() > 0.03),
                "country": str(rng.choice(countries, p=country_p)),
                "signup_source": str(rng.choice(sources, p=[0.4, 0.2, 0.16, 0.12, 0.12])),
                "signup_day": signup_day,
                "created_at": cfg.start_date + timedelta(days=signup_day),
                "preferred_categories": top_cats,
                "preferred_brands": pref_brands,
                "price_sensitivity": float(np.clip(arche[2] + rng.normal(0, 0.09), 0.02, 0.98)),
                "archetype": arche[0],
                "activity": float(max(0.05, arche[1] * rng.lognormal(0, 0.32))),
                "basket_lambda": float(arche[3]),
                "review_propensity": float(arche[4]),
            }
        )
    users = pd.DataFrame(rows)
    users["preferred_category_ids"] = users["preferred_categories"].apply(lambda cs: [cat_ids[c] for c in cs])
    return users, latent


def _affinity_matrix(user_latent: np.ndarray, product_latent: np.ndarray, users: pd.DataFrame,
                     products: pd.DataFrame) -> np.ndarray:
    """P(user buys product | product chosen) proportional weights, shape (n_products, n_users)."""
    taste = user_latent @ product_latent.T  # (n_users, n_products)
    price = products["price"].to_numpy()
    price_z = (np.log1p(price) - np.log1p(price).mean()) / (np.log1p(price).std() + 1e-9)
    sensitivity = users["price_sensitivity"].to_numpy()[:, None]
    # price-sensitive users are penalised for expensive items
    utility = 2.6 * taste - 1.5 * sensitivity * price_z[None, :]
    # brand loyalty bonus
    brand_of_product = products["brand_id"].to_numpy()
    for ui, prefs in enumerate(users["preferred_brands"]):
        if prefs:
            utility[ui, np.isin(brand_of_product, prefs)] += 0.55
    weights = np.exp(utility - utility.max(axis=0, keepdims=True))
    return weights.T  # (n_products, n_users)


def _seasonality(cfg: GeneratorConfig) -> pd.DataFrame:
    days = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D", tz=timezone.utc)
    idx = np.arange(cfg.n_days)
    dow = days.dayofweek.to_numpy()
    # weekend uplift, weekly + annual seasonality, gentle growth trend
    weekly = 1.0 + 0.22 * np.isin(dow, [4, 5]) + 0.08 * (dow == 6) - 0.10 * (dow == 1)
    annual = 1.0 + 0.28 * np.sin(2 * np.pi * (idx / 365.25) - 0.9)
    month = days.month.to_numpy()
    holiday = 1.0 + 0.45 * (month == 11) + 0.30 * (month == 12) - 0.12 * (month == 2)
    trend = 1.0 + 0.30 * (idx / max(cfg.n_days - 1, 1))
    return pd.DataFrame(
        {"date": days, "day_index": idx, "dow": dow, "month": month,
         "weekly": weekly, "annual": annual, "holiday": holiday, "trend": trend,
         "factor": weekly * annual * holiday * trend}
    )


def _price_history(rng: np.random.Generator, cfg: GeneratorConfig, products: pd.DataFrame,
                   season: pd.DataFrame) -> pd.DataFrame:
    """Weekly price/competitor/promotion series per product."""
    weeks = season.iloc[::7].reset_index(drop=True)
    frames = []
    for row in products.itertuples():
        n_w = len(weeks)
        drift = np.cumsum(rng.normal(0, 0.006, n_w))
        noise = rng.normal(0, 0.012, n_w)
        promo = (rng.random(n_w) < 0.14).astype(float)
        promo_depth = promo * rng.uniform(0.06, 0.28, n_w)
        price = row.price * (1 + drift + noise) * (1 - promo_depth)
        competitor = row.price * (1 + drift * 0.7 + rng.normal(0, 0.03, n_w) + rng.normal(0.01, 0.02))
        frames.append(
            pd.DataFrame(
                {
                    "product_id": row.id,
                    "date": weeks["date"].to_numpy(),
                    "day_index": weeks["day_index"].to_numpy(),
                    "price": np.round(np.clip(price, row.base_cost * 1.03, row.price * 1.6), 2),
                    "competitor_price": np.round(np.clip(competitor, row.base_cost, row.price * 1.8), 2),
                    "promo": promo,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _simulate_demand(rng: np.random.Generator, cfg: GeneratorConfig, products: pd.DataFrame,
                     season: pd.DataFrame, price_history: pd.DataFrame) -> np.ndarray:
    """Return an (n_products, n_days) integer matrix of units sold."""
    n_d = cfg.n_days
    base = products["popularity_base"].to_numpy()[:, None]
    factor = season["factor"].to_numpy()[None, :]

    # Product lifecycle: ramp up after launch, slow decay for older SKUs.
    launch_day = np.array([
        (pd.Timestamp(ts).to_pydatetime() - cfg.start_date).days for ts in products["launched_at"]
    ])[:, None]
    day_idx = np.arange(n_d)[None, :]
    age = day_idx - launch_day
    lifecycle = np.where(age < 0, 0.0, 1.0 - np.exp(-np.clip(age, 0, None) / 21.0)) * np.exp(
        -np.clip(age, 0, None) / 900.0
    )

    # Price elasticity: cheaper-than-usual weeks sell more.
    price_pivot = price_history.pivot_table(index="product_id", columns="day_index", values="price")
    price_pivot = price_pivot.reindex(index=products["id"]).reindex(columns=range(n_d)).ffill(axis=1).bfill(axis=1)
    rel_price = price_pivot.to_numpy() / products["price"].to_numpy()[:, None]
    elasticity = np.clip(rel_price, 0.4, 1.8) ** (-1.7)

    # Scaled so the head of the catalogue carries enough daily volume to be
    # forecastable; the long tail stays realistically sparse.
    lam = 0.17 * base * factor * lifecycle * elasticity
    lam = np.clip(lam, 0, 40)
    return rng.poisson(lam)


def _generate_transactions(rng: np.random.Generator, cfg: GeneratorConfig, products: pd.DataFrame,
                           users: pd.DataFrame, demand: np.ndarray, affinity: np.ndarray,
                           price_history: pd.DataFrame, season: pd.DataFrame):
    """Turn the demand matrix into concrete purchases, orders and order items."""
    _, n_d = demand.shape
    signup_day = users["signup_day"].to_numpy()
    activity = users["activity"].to_numpy()

    weights = affinity * activity[None, :]
    weights /= weights.sum(axis=1, keepdims=True)
    cdf = np.cumsum(weights, axis=1)

    price_lookup = (
        price_history.pivot_table(index="product_id", columns="day_index", values="price")
        .reindex(index=products["id"]).reindex(columns=range(n_d)).ffill(axis=1).bfill(axis=1).to_numpy()
    )
    discount_arr = products["discount_pct"].to_numpy()

    purchases: list[tuple[int, int, int, float, float]] = []  # (user_idx, product_idx, day, unit_price, discount)
    p_idx, d_idx = np.nonzero(demand)
    for pi, di in zip(p_idx, d_idx):
        units = int(demand[pi, di])
        draws = rng.random(units)
        buyers = np.searchsorted(cdf[pi], draws)
        buyers = np.clip(buyers, 0, len(users) - 1)
        for b in buyers:
            if signup_day[b] > di:  # not yet a customer - one resample, else drop
                b = int(np.clip(np.searchsorted(cdf[pi], rng.random()), 0, len(users) - 1))
                if signup_day[b] > di:
                    continue
            purchases.append((int(b), int(pi), int(di), float(price_lookup[pi, di]), float(discount_arr[pi])))

    if not purchases:  # pragma: no cover - defensive
        raise RuntimeError("demand simulation produced no purchases")

    pur = pd.DataFrame(purchases, columns=["user_idx", "product_idx", "day", "unit_price", "discount_pct"])
    pur["user_id"] = pur["user_idx"] + 1
    pur["product_id"] = pur["product_idx"] + 1

    # Group same-user same-day purchases into a single order (basket behaviour).
    dates = season["date"].to_numpy()
    order_rows, item_rows = [], []
    order_id = 0
    status_choices = np.array(["paid", "delivered", "shipped", "cancelled", "returned"])
    status_p = np.array([0.18, 0.62, 0.12, 0.05, 0.03])

    # Pre-index products by category so complementary picks are cheap.
    cat_of_product = products["category_slug"].to_numpy()
    products_by_cat: dict[str, np.ndarray] = {
        slug: np.nonzero(cat_of_product == slug)[0] for slug in np.unique(cat_of_product)
    }
    basket_lambda = users["basket_lambda"].to_numpy()
    extra_rows: list[dict] = []

    for (uid, day), grp in pur.groupby(["user_id", "day"], sort=True):
        order_id += 1
        placed = pd.Timestamp(dates[day]).to_pydatetime().replace(
            hour=int(rng.integers(6, 23)), minute=int(rng.integers(0, 60))
        )
        total = 0.0
        discount_total = 0.0
        items_in_order = []
        # Basket expansion: shoppers often add complementary items in the same
        # session. This is what creates co-purchase (frequently-bought-together)
        # structure in the data rather than a catalogue of single-item orders.
        u_idx = int(grp["user_idx"].iloc[0])
        anchor_idx = int(grp["product_idx"].iloc[0])
        n_extra = int(rng.poisson(max(basket_lambda[u_idx] - 1.0, 0.15)))
        if n_extra > 0:
            anchor_cat = str(cat_of_product[anchor_idx])
            pool_cats = [anchor_cat] + COMPLEMENTARY.get(anchor_cat, [])
            pool = np.concatenate([products_by_cat[c] for c in pool_cats if c in products_by_cat])
            already = set(grp["product_idx"].tolist())
            if pool.size:
                w = affinity[pool, u_idx].astype(float)
                w = w / w.sum() if w.sum() > 0 else None
                picks = rng.choice(pool, size=min(n_extra, 3), replace=False,
                                   p=w) if w is not None and pool.size > 3 else pool[:0]
                for pk in np.atleast_1d(picks):
                    pk = int(pk)
                    if pk in already:
                        continue
                    already.add(pk)
                    extra_rows.append({
                        "user_idx": u_idx, "product_idx": pk, "day": int(day),
                        "unit_price": float(price_lookup[pk, day]), "discount_pct": float(discount_arr[pk]),
                        "user_id": int(uid), "product_id": pk + 1, "_order_id": order_id,
                    })
        for row in grp.itertuples():
            qty = 1 + int(rng.random() < 0.18) + int(rng.random() < 0.05)
            unit = round(row.unit_price, 2)
            line = round(unit * qty * (1 - row.discount_pct / 100.0), 2)
            total += line
            discount_total += round(unit * qty - line, 2)
            items_in_order.append(
                {
                    "order_id": order_id,
                    "product_id": int(row.product_id),
                    "quantity": qty,
                    "unit_price": unit,
                    "discount_pct": float(row.discount_pct),
                    "line_total": line,
                }
            )
        for extra in [e for e in extra_rows if e.get("_order_id") == order_id]:
            qty = 1 + int(rng.random() < 0.12)
            unit = round(extra["unit_price"], 2)
            line = round(unit * qty * (1 - extra["discount_pct"] / 100.0), 2)
            total += line
            discount_total += round(unit * qty - line, 2)
            items_in_order.append({
                "order_id": order_id, "product_id": int(extra["product_id"]), "quantity": qty,
                "unit_price": unit, "discount_pct": float(extra["discount_pct"]), "line_total": line,
            })
        status = str(rng.choice(status_choices, p=status_p))
        order_rows.append(
            {
                "id": order_id,
                "user_id": int(uid),
                "order_number": f"ECI-ORD-{order_id:07d}",
                "status": status,
                "total_amount": round(total, 2),
                "discount_amount": round(discount_total, 2),
                "item_count": sum(i["quantity"] for i in items_in_order),
                "currency": "USD",
                "channel": str(rng.choice(["web", "mobile", "app"], p=[0.52, 0.33, 0.15])),
                "placed_at": placed,
                "day": int(day),
            }
        )
        item_rows.extend(items_in_order)

    if extra_rows:
        extras_df = pd.DataFrame(extra_rows).drop(columns=["_order_id"])
        pur = pd.concat([pur, extras_df], ignore_index=True).sort_values(["user_id", "day"]).reset_index(drop=True)
    orders = pd.DataFrame(order_rows)
    order_items = pd.DataFrame(item_rows)
    for i, row in enumerate(item_rows):
        row["id"] = i + 1
    order_items["id"] = np.arange(1, len(order_items) + 1)
    return pur, orders, order_items


def _render_review(rng: np.random.Generator, rating: int, category_slug: str) -> tuple[str, str, dict]:
    """Compose review text from aspect phrases; returns (title, body, aspect ground truth)."""
    candidates = CATEGORY_ASPECTS.get(category_slug, ["value", "build_quality", "ease_of_use"])
    n_aspects = int(rng.integers(1, min(4, len(candidates)) + 1))
    chosen = list(rng.choice(candidates, size=n_aspects, replace=False))

    # Real reviewers are inconsistent: a 4-star review can read like a rant and a
    # 2-star one can open warmly. Without this ambiguity the text->rating mapping
    # is deterministic and any classifier scores a meaningless 100%.
    tone = rating
    if rng.random() < 0.22:
        tone = int(np.clip(rating + rng.choice([-1, 1]), 1, 5))

    if tone >= 4:
        pos_prob = 0.82
        openers, closers = REVIEW_OPENERS_POS, REVIEW_CLOSERS_POS
    elif tone == 3:
        pos_prob = 0.5
        openers, closers = REVIEW_OPENERS_NEU, REVIEW_CLOSERS_NEU
    else:
        pos_prob = 0.18
        openers, closers = REVIEW_OPENERS_NEG, REVIEW_CLOSERS_NEG

    # Neutral filler shared across every rating blurs the class boundaries further.
    filler = ["Arrived as described.", "Packaging was standard.", "Used it for a few weeks now.",
              "Bought it on a whim.", "Comparable to my previous one.", "Nothing surprising here."]

    parts = [str(rng.choice(openers))]
    aspects: dict[str, str] = {}
    for a in chosen:
        polarity = "positive" if rng.random() < pos_prob else "negative"
        phrase = str(rng.choice(ASPECT_LEXICON[a][polarity]))
        aspects[a] = polarity
        parts.append(phrase.capitalize() + ".")
    if len(chosen) > 1 and len(set(aspects.values())) > 1:
        parts.insert(2, "That said,")
    if rng.random() < 0.45:
        parts.insert(int(rng.integers(1, len(parts) + 1)), str(rng.choice(filler)))
    parts.append(str(rng.choice(closers)))
    body = " ".join(parts).replace("That said, ", "That said, ")
    # Titles are drawn from the (noisy) tone, not the rating, and overlap across
    # adjacent tones. A rating-derived title would hand the label straight to the
    # classifier and produce a meaningless 100% accuracy.
    title_pool = {
        "pos": ["Excellent", "Very good", "Happy with it", "Does the job well", "Better than expected"],
        "neu": ["Decent but flawed", "Mixed", "It is okay", "Fine for the price", "Some good, some bad"],
        "neg": ["Underwhelming", "Would not recommend", "Disappointing", "Not for me", "Expected more"],
    }
    bucket = "pos" if tone >= 4 else ("neu" if tone == 3 else "neg")
    if rng.random() < 0.15:  # occasionally a title that does not match the body
        bucket = str(rng.choice(["pos", "neu", "neg"]))
    return str(rng.choice(title_pool[bucket])), body, aspects


def _build_reviews(rng: np.random.Generator, cfg: GeneratorConfig, purchases: pd.DataFrame,
                   products: pd.DataFrame, users: pd.DataFrame, season: pd.DataFrame) -> pd.DataFrame:
    quality = products["quality"].to_numpy()
    cat_slug = products["category_slug"].to_numpy()
    propensity = users["review_propensity"].to_numpy()
    dates = season["date"].to_numpy()

    # one review per (user, product) - dedupe purchases first
    uniq = purchases.drop_duplicates(subset=["user_id", "product_id"], keep="first")
    draw = rng.random(len(uniq))
    keep = draw < propensity[uniq["user_idx"].to_numpy()]
    uniq = uniq[keep]
    if len(uniq) > cfg.target_reviews:
        uniq = uniq.sample(n=cfg.target_reviews, random_state=cfg.seed % (2**31))
    uniq = uniq.sort_values("day")

    rows = []
    verified_pairs = set(zip(uniq["user_id"], uniq["product_id"]))
    for rid, row in enumerate(uniq.itertuples(), start=1):
        q = quality[row.product_idx]
        # rating = product quality + user noise, mapped to 1..5
        latent_score = np.clip(rng.normal(1.6 + 3.7 * q, 0.85), 1.0, 5.49)
        rating = int(np.clip(round(latent_score), 1, 5))
        title, body, aspects = _render_review(rng, rating, str(cat_slug[row.product_idx]))
        lag = int(rng.integers(2, 21))
        day = min(row.day + lag, cfg.n_days - 1)
        rows.append(
            {
                "id": rid,
                "product_id": int(row.product_id),
                "user_id": int(row.user_id),
                "rating": rating,
                "title": title,
                "body": body,
                "verified_purchase": True,
                "helpful_votes": int(rng.negative_binomial(2, 0.25)),
                "aspects_truth": aspects,
                "created_at": pd.Timestamp(dates[day]).to_pydatetime(),
                "day": day,
            }
        )

    # Real catalogues also carry unverified reviews from browsers who never
    # completed a purchase. Top up to the configured target with those.
    deficit = cfg.target_reviews - len(rows)
    if deficit > 0:
        n_u, n_p = len(users), len(products)
        attempts = 0
        rid = len(rows)
        while deficit > 0 and attempts < deficit * 12:
            attempts += 1
            u = int(rng.integers(0, n_u))
            p = int(rng.integers(0, n_p))
            pair = (u + 1, p + 1)
            if pair in verified_pairs:
                continue
            verified_pairs.add(pair)
            q = quality[p]
            rating = int(np.clip(round(np.clip(rng.normal(1.5 + 3.7 * q, 1.0), 1.0, 5.49)), 1, 5))
            title, body, aspects = _render_review(rng, rating, str(cat_slug[p]))
            day = int(rng.integers(max(users["signup_day"].iloc[u], 0), cfg.n_days))
            rid += 1
            deficit -= 1
            rows.append({
                "id": rid, "product_id": p + 1, "user_id": u + 1, "rating": rating, "title": title,
                "body": body, "verified_purchase": False,
                "helpful_votes": int(rng.negative_binomial(2, 0.35)), "aspects_truth": aspects,
                "created_at": pd.Timestamp(dates[day]).to_pydatetime(), "day": day,
            })
    df = pd.DataFrame(rows)
    df["id"] = np.arange(1, len(df) + 1)
    return df


def _build_events(rng: np.random.Generator, cfg: GeneratorConfig, purchases: pd.DataFrame, products: pd.DataFrame,
                  users: pd.DataFrame, affinity: np.ndarray, season: pd.DataFrame,
                  reviews: pd.DataFrame) -> pd.DataFrame:
    """Purchase-anchored funnel events plus organic browsing sessions."""
    dates = season["date"].to_numpy()
    n_users, n_products = len(users), len(products)
    rows: list[dict] = []

    def ts(day: int, hour_jitter: bool = True) -> datetime:
        base = pd.Timestamp(dates[min(max(day, 0), cfg.n_days - 1)]).to_pydatetime()
        if hour_jitter:
            base = base.replace(hour=int(rng.integers(6, 23)), minute=int(rng.integers(0, 60)))
        return base

    # Sessions must group multiple events. Giving every event its own session id
    # makes session counts meaningless and inflates session-level conversion.
    session_cache: dict[tuple[int, int, int], str] = {}

    def session_for(user_id: int, day: int, slot: int = 0) -> str:
        key = (int(user_id), int(day), int(slot))
        if key not in session_cache:
            session_cache[key] = f"s{user_id}-{day}-{slot}-{rng.integers(10000, 99999)}"
        return session_cache[key]

    # 1. Funnel leading to every purchase: view -> click -> add_to_cart -> purchase
    for row in purchases.itertuples():
        day = int(row.day)
        session = session_for(int(row.user_id), day, 0)
        for etype, offset in (("product_view", -1), ("click", 0), ("add_to_cart", 0), ("purchase", 0)):
            if etype == "click" and rng.random() < 0.25:
                continue
            rows.append(
                {
                    "user_id": int(row.user_id),
                    "session_id": session,
                    "event_type": etype,
                    "product_id": int(row.product_id),
                    "quantity": 1,
                    "value": float(row.unit_price) if etype == "purchase" else 0.0,
                    "source": "web",
                    "metadata": {"funnel": True},
                    "occurred_at": ts(day + offset),
                    "day": max(day + offset, 0),
                }
            )

    # 2. Organic browsing weighted by affinity (creates the implicit-feedback signal)
    remaining = max(0, cfg.target_browse_events - len(rows))
    activity = users["activity"].to_numpy()
    # sqrt-weighting stops the handful of hyper-active users from absorbing all
    # browsing traffic, which would collapse the distinct-session count
    user_p = np.sqrt(activity) / np.sqrt(activity).sum()
    signup_day = users["signup_day"].to_numpy()
    aff_by_user = affinity.T  # (n_users, n_products)
    aff_cdf = np.cumsum(aff_by_user / aff_by_user.sum(axis=1, keepdims=True), axis=1)

    browse_users = rng.choice(n_users, size=remaining, p=user_p)
    browse_days = rng.integers(0, cfg.n_days, size=remaining)
    draws = rng.random(remaining)
    etype_choices = np.array(["product_view", "click", "wishlist", "add_to_cart", "remove_from_cart"])
    # Browsing is overwhelmingly view traffic; only a small share reaches the cart.
    etype_p = np.array([0.795, 0.155, 0.022, 0.021, 0.007])
    etypes = rng.choice(etype_choices, size=remaining, p=etype_p)

    for k in range(remaining):
        u = int(browse_users[k])
        d = int(browse_days[k])
        if signup_day[u] > d:  # shift the event into the user's active window
            span = cfg.n_days - signup_day[u]
            if span <= 0:
                continue
            d = int(signup_day[u] + rng.integers(0, span))
        pi = int(np.clip(np.searchsorted(aff_cdf[u], draws[k]), 0, n_products - 1))
        rows.append(
            {
                "user_id": u + 1,
                # 1-3 browsing sessions per user-day, each holding several events
                "session_id": session_for(u + 1, d, int(rng.integers(0, 6))),
                "event_type": str(etypes[k]),
                "product_id": pi + 1,
                "quantity": 1,
                "value": 0.0,
                "source": str(rng.choice(["web", "mobile", "app"], p=[0.5, 0.33, 0.17])),
                "metadata": {"organic": True},
                "occurred_at": ts(d),
                "day": d,
            }
        )

    # 3. Review events mirror the reviews table
    for row in reviews.itertuples():
        rows.append(
            {
                "user_id": int(row.user_id),
                "session_id": session_for(int(row.user_id), int(row.day), 0),
                "event_type": "review",
                "product_id": int(row.product_id),
                "quantity": 1,
                "value": float(row.rating),
                "source": "web",
                "metadata": {"rating": int(row.rating)},
                "occurred_at": ts(int(row.day)),
                "day": int(row.day),
            }
        )

    events = pd.DataFrame(rows).sort_values("occurred_at").reset_index(drop=True)
    events["id"] = np.arange(1, len(events) + 1)
    return events


def _build_search_events(rng: np.random.Generator, cfg: GeneratorConfig, products: pd.DataFrame, users: pd.DataFrame,
                         affinity: np.ndarray, season: pd.DataFrame, brands: pd.DataFrame) -> pd.DataFrame:
    dates = season["date"].to_numpy()
    n_users, n_products = len(users), len(products)
    activity = users["activity"].to_numpy()
    user_p = activity / activity.sum()
    signup_day = users["signup_day"].to_numpy()
    aff_by_user = affinity.T
    aff_cdf = np.cumsum(aff_by_user / aff_by_user.sum(axis=1, keepdims=True), axis=1)
    brand_names = brands["name"].tolist()
    keywords = products["keyword"].to_numpy()

    rows = []
    n = cfg.target_search_events
    us = rng.choice(n_users, size=n, p=user_p)
    ds = rng.integers(0, cfg.n_days, size=n)
    draws = rng.random(n)
    for k in range(n):
        u, d = int(us[k]), int(ds[k])
        if signup_day[u] > d:
            span = cfg.n_days - signup_day[u]
            if span <= 0:
                continue
            d = int(signup_day[u] + rng.integers(0, span))
        pi = int(np.clip(np.searchsorted(aff_cdf[u], draws[k]), 0, n_products - 1))
        kw = str(keywords[pi])
        template = str(rng.choice(SEARCH_INTENT_TEMPLATES))
        query = template.format(
            kw=kw,
            use_case=str(rng.choice(USE_CASES)),
            brand=str(rng.choice(brand_names)),
            price=int(rng.choice([25, 50, 100, 200, 500])),
            colour=str(rng.choice(COLOURS)),
        )
        # inject realistic typos into ~8% of queries
        if rng.random() < 0.08 and len(query) > 6:
            pos = int(rng.integers(1, len(query) - 1))
            query = query[:pos] + query[pos + 1:]
        result_count = int(rng.integers(0, 48))
        clicked = result_count > 0 and rng.random() < 0.42
        position = int(rng.integers(1, min(result_count, 20) + 1)) if clicked else None
        rows.append(
            {
                "user_id": u + 1,
                "session_id": f"s{u + 1}-{d}-{rng.integers(1000, 9999)}",
                "query": query,
                "normalized_query": query.lower().strip(),
                "result_count": result_count,
                "clicked_product_id": pi + 1 if clicked else None,
                "clicked_position": position,
                "converted": bool(clicked and rng.random() < 0.19),
                "latency_ms": float(np.round(rng.gamma(3.2, 9.0), 2)),
                "filters": {"category": str(products["category_slug"].iloc[pi])} if rng.random() < 0.3 else {},
                "occurred_at": pd.Timestamp(dates[d]).to_pydatetime().replace(
                    hour=int(rng.integers(6, 23)), minute=int(rng.integers(0, 60))
                ),
                "day": d,
            }
        )
    df = pd.DataFrame(rows)
    df["id"] = np.arange(1, len(df) + 1)
    return df


def _build_impressions(rng: np.random.Generator, cfg: GeneratorConfig, purchases: pd.DataFrame,
                       users: pd.DataFrame, products: pd.DataFrame, affinity: np.ndarray,
                       season: pd.DataFrame) -> pd.DataFrame:
    """Historical recommendation impressions with realistic click/convert outcomes.

    Without these the AI-effectiveness panel of the dashboard has nothing to
    report. Click probability rises with the user's affinity for the item, so
    CTR by strategy is meaningful rather than uniform noise.
    """
    dates = season["date"].to_numpy()
    n_users, n_products = len(users), len(products)
    signup_day = users["signup_day"].to_numpy()
    aff = affinity.T  # (users, products)
    aff_cdf = np.cumsum(aff / aff.sum(axis=1, keepdims=True), axis=1)
    purchased = {(int(r.user_id), int(r.product_id)) for r in purchases.itertuples()}

    surfaces = np.array(["homepage", "product_detail", "search", "cart"])
    surface_p = np.array([0.52, 0.26, 0.16, 0.06])
    strategies = np.array(["hybrid", "cold_start", "trending_events", "content_similarity"])
    strategy_p = np.array([0.58, 0.14, 0.18, 0.10])

    n = int(cfg.target_browse_events * 0.55)
    activity = users["activity"].to_numpy()
    us = rng.choice(n_users, size=n, p=activity / activity.sum())
    ds = rng.integers(0, cfg.n_days, size=n)
    draws = rng.random(n)
    rows = []
    for k in range(n):
        u, d = int(us[k]), int(ds[k])
        if signup_day[u] > d:
            span = cfg.n_days - signup_day[u]
            if span <= 0:
                continue
            d = int(signup_day[u] + rng.integers(0, span))
        pi = int(np.clip(np.searchsorted(aff_cdf[u], draws[k]), 0, n_products - 1))
        rank = int(rng.integers(1, 13))
        strategy = str(rng.choice(strategies, p=strategy_p))
        # position bias x affinity: higher slots and better-matched items get clicked
        position_factor = 1.0 / (1.0 + 0.35 * (rank - 1))
        affinity_factor = float(aff[u, pi] / (aff[u].max() or 1.0))
        base = 0.16 if strategy == "hybrid" else (0.11 if strategy == "content_similarity" else 0.08)
        click_p = float(np.clip(base * position_factor * (0.5 + 1.6 * affinity_factor), 0.005, 0.85))
        clicked = bool(rng.random() < click_p)
        converted = bool(clicked and (int(us[k]) + 1, pi + 1) in purchased and rng.random() < 0.34)
        rows.append({
            "user_id": u + 1, "product_id": pi + 1,
            "surface": str(rng.choice(surfaces, p=surface_p)), "strategy": strategy,
            "model_version": "seed-v1", "score": round(float(0.3 + 0.7 * affinity_factor), 5),
            "rank": rank, "explanation": {}, "clicked": clicked, "converted": converted,
            "served_at": pd.Timestamp(dates[d]).to_pydatetime().replace(
                hour=int(rng.integers(6, 23)), minute=int(rng.integers(0, 60))),
        })
    df = pd.DataFrame(rows)
    df["id"] = np.arange(1, len(df) + 1)
    return df


def _finalise_products(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Backfill rating aggregates so the catalogue is internally consistent."""
    agg = reviews.groupby("product_id")["rating"].agg(["mean", "count"])
    products = products.copy()
    products["rating_avg"] = products["id"].map(agg["mean"]).fillna(0.0).round(2)
    products["rating_count"] = products["id"].map(agg["count"]).fillna(0).astype(int)
    return products


def generate_dataset(config: GeneratorConfig | None = None) -> SyntheticDataset:
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    categories = _build_categories()
    brands = _build_brands()
    products, product_latent = _build_products(rng, cfg, categories, brands)
    users, user_latent = _build_users(rng, cfg, products, product_latent)
    affinity = _affinity_matrix(user_latent, product_latent, users, products)

    season = _seasonality(cfg)
    price_history = _price_history(rng, cfg, products, season)
    demand = _simulate_demand(rng, cfg, products, season, price_history)

    purchases, orders, order_items = _generate_transactions(
        rng, cfg, products, users, demand, affinity, price_history, season
    )
    reviews = _build_reviews(rng, cfg, purchases, products, users, season)
    events = _build_events(rng, cfg, purchases, products, users, affinity, season, reviews)
    search_events = _build_search_events(rng, cfg, products, users, affinity, season, brands)
    impressions = _build_impressions(rng, cfg, purchases, users, products, affinity, season)
    products = _finalise_products(products, reviews)

    # last_active_at from event stream
    last_event = events.groupby("user_id")["occurred_at"].max()
    users = users.copy()
    users["last_active_at"] = users["id"].map(last_event)

    dd_rows = []
    for pi in range(len(products)):
        units = demand[pi]
        nz = np.nonzero(units)[0]
        for d in nz:
            dd_rows.append({"product_id": pi + 1, "day_index": int(d),
                            "date": season["date"].iloc[int(d)], "units": int(units[d])})
    daily_demand = pd.DataFrame(dd_rows)

    return SyntheticDataset(
        categories=categories,
        brands=brands,
        products=products,
        users=users,
        orders=orders,
        order_items=order_items,
        reviews=reviews,
        events=events,
        search_events=search_events,
        price_history=price_history,
        daily_demand=daily_demand,
        impressions=impressions,
        dataset_version=cfg.dataset_version,
        config=cfg,
    )
