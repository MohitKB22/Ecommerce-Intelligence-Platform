"""Product / category / brand data access."""
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.orm import selectinload

from app.models import Brand, Category, OrderItem, Product, Review, UserEvent
from app.repositories.base import BaseRepository

SORT_OPTIONS = {
    "relevance": (Product.rating_count.desc(), Product.rating_avg.desc()),
    "price_asc": (Product.price.asc(),),
    "price_desc": (Product.price.desc(),),
    "rating": (Product.rating_avg.desc(), Product.rating_count.desc()),
    "newest": (Product.launched_at.desc().nullslast(), Product.id.desc()),
    "discount": (Product.discount_pct.desc(),),
    "popularity": (Product.rating_count.desc(),),
}


class ProductRepository(BaseRepository[Product]):
    model = Product

    def with_relations(self, product_id: int) -> Product | None:
        return self.db.execute(
            select(Product)
            .options(selectinload(Product.category), selectinload(Product.brand))
            .where(Product.id == product_id)
        ).scalars().first()

    def list_products(self, *, page: int = 1, page_size: int = 24, category_slug: str | None = None,
                      category_id: int | None = None, brand_id: int | None = None,
                      min_price: float | None = None, max_price: float | None = None,
                      min_rating: float | None = None, in_stock_only: bool = False,
                      on_sale_only: bool = False, sort: str = "relevance",
                      ids: Sequence[int] | None = None) -> tuple[list[Product], int]:
        stmt: Select = (
            select(Product)
            .options(selectinload(Product.category), selectinload(Product.brand))
            .where(Product.is_active.is_(True))
        )
        if category_slug:
            stmt = stmt.join(Category, Category.id == Product.category_id).where(Category.slug == category_slug)
        if category_id is not None:
            stmt = stmt.where(Product.category_id == category_id)
        if brand_id is not None:
            stmt = stmt.where(Product.brand_id == brand_id)
        if min_price is not None:
            stmt = stmt.where(Product.price >= min_price)
        if max_price is not None:
            stmt = stmt.where(Product.price <= max_price)
        if min_rating is not None:
            stmt = stmt.where(Product.rating_avg >= min_rating)
        if in_stock_only:
            stmt = stmt.where(Product.inventory > 0)
        if on_sale_only:
            stmt = stmt.where(Product.discount_pct > 0)
        if ids is not None:
            stmt = stmt.where(Product.id.in_(list(ids)))

        for clause in SORT_OPTIONS.get(sort, SORT_OPTIONS["relevance"]):
            stmt = stmt.order_by(clause)
        return self.paginate(stmt, page, page_size)

    def categories(self) -> list[Category]:
        return list(self.db.execute(select(Category).order_by(Category.name)).scalars().all())

    def category_by_slug(self, slug: str) -> Category | None:
        return self.db.execute(select(Category).where(Category.slug == slug)).scalars().first()

    def brands(self, limit: int = 200) -> list[Brand]:
        return list(self.db.execute(select(Brand).order_by(Brand.name).limit(limit)).scalars().all())

    def popular_categories(self, limit: int = 8) -> list[dict]:
        rows = self.db.execute(
            select(Category.id, Category.slug, Category.name, func.count(Product.id).label("products"),
                   func.coalesce(func.avg(Product.rating_avg), 0.0))
            .join(Product, Product.category_id == Category.id)
            .where(Product.is_active.is_(True))
            .group_by(Category.id, Category.slug, Category.name)
            .order_by(func.count(Product.id).desc())
            .limit(limit)
        ).all()
        return [
            {"id": int(cid), "slug": slug, "name": name, "product_count": int(count),
             "avg_rating": round(float(rating), 2)}
            for cid, slug, name, count, rating in rows
        ]

    def reviews_for(self, product_id: int, page: int = 1, page_size: int = 10,
                    sentiment: str | None = None) -> tuple[list[Review], int]:
        stmt = select(Review).where(Review.product_id == product_id)
        if sentiment:
            stmt = stmt.where(Review.sentiment_label == sentiment)
        stmt = stmt.order_by(Review.helpful_votes.desc(), Review.created_at.desc())
        return self.paginate(stmt, page, page_size)

    def refresh_rating_aggregate(self, product_id: int) -> None:
        """Recompute denormalised rating columns after a review changes."""
        row = self.db.execute(
            select(func.avg(Review.rating), func.count(Review.id)).where(Review.product_id == product_id)
        ).one()
        product = self.db.get(Product, product_id)
        if product is not None:
            product.rating_avg = round(float(row[0] or 0.0), 2)
            product.rating_count = int(row[1] or 0)
            self.db.commit()

    def low_inventory(self, threshold: int = 10, limit: int = 20) -> list[Product]:
        return list(self.db.execute(
            select(Product).where(Product.is_active.is_(True), Product.inventory <= threshold)
            .order_by(Product.inventory.asc()).limit(limit)
        ).scalars().all())

    def best_sellers(self, limit: int = 10, days: int = 90) -> list[dict]:
        from datetime import datetime, timedelta, timezone

        from app.models import Order

        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(Product, func.sum(OrderItem.quantity).label("units"),
                   func.sum(OrderItem.line_total).label("revenue"))
            .join(OrderItem, OrderItem.product_id == Product.id)
            .join(Order, Order.id == OrderItem.order_id)
            .where(Order.placed_at >= since, Order.status.in_(("paid", "shipped", "delivered")))
            .group_by(Product.id)
            .order_by(func.sum(OrderItem.quantity).desc())
            .limit(limit)
        ).all()
        return [{"product": p, "units_sold": int(u or 0), "revenue": round(float(r or 0.0), 2)}
                for p, u, r in rows]

    def conversion_stats(self, limit: int = 10, best: bool = True) -> list[dict]:
        """View -> purchase conversion per product, from the event stream."""
        views = (
            select(UserEvent.product_id, func.count(UserEvent.id).label("views"))
            .where(UserEvent.event_type.in_(("product_view", "click")))
            .group_by(UserEvent.product_id).subquery()
        )
        purchases = (
            select(UserEvent.product_id, func.count(UserEvent.id).label("purchases"))
            .where(UserEvent.event_type == "purchase")
            .group_by(UserEvent.product_id).subquery()
        )
        ratio = (func.coalesce(purchases.c.purchases, 0) * 1.0) / views.c.views
        stmt = (
            select(Product, views.c.views, func.coalesce(purchases.c.purchases, 0), ratio.label("cvr"))
            .join(views, views.c.product_id == Product.id)
            .outerjoin(purchases, purchases.c.product_id == Product.id)
            .where(views.c.views >= 10)
            .order_by(ratio.desc() if best else ratio.asc())
            .limit(limit)
        )
        return [
            {"product": p, "views": int(v), "purchases": int(pu), "conversion_rate": round(float(cvr), 5)}
            for p, v, pu, cvr in self.db.execute(stmt).all()
        ]
