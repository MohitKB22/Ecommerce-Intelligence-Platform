"""Search index: BM25 lexical postings + dense vectors + facet metadata.

The index is built once from the catalogue and rebuilt on demand (or by the
background worker). It is deliberately held in memory: for a catalogue of this
size an exact scan is sub-millisecond and avoids a second stateful service.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from ml.features.bm25 import BM25Index
from ml.features.embeddings import build_embedder, tokenize
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Brand, Category, Product
from app.search.vector_store import NumpyVectorStore, VectorStore, build_vector_store

logger = get_logger("search_index")


@dataclass
class ProductIndexEntry:
    id: int
    title: str
    sku: str
    category_id: int
    category_slug: str
    category_name: str
    brand_id: int
    brand_name: str
    price: float
    effective_price: float
    discount_pct: float
    rating_avg: float
    rating_count: int
    inventory: int
    is_active: bool
    tags: list[str] = field(default_factory=list)
    popularity: float = 0.0
    conversion_rate: float = 0.0


@dataclass
class SearchIndex:
    entries: dict[int, ProductIndexEntry]
    bm25: BM25Index
    vectors: VectorStore
    embedder: Any
    title_terms: dict[str, set[int]]
    suggestions: list[str]
    built_at: datetime
    build_ms: float

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": self.size,
            "vocabulary": len(self.bm25.vocabulary),
            "vector_backend": self.vectors.backend,
            "vector_count": self.vectors.size(),
            "embedding_backend": getattr(self.embedder, "backend_name", "unknown"),
            "suggestion_terms": len(self.suggestions),
            "built_at": self.built_at.isoformat(),
            "build_ms": round(self.build_ms, 2),
        }


class SearchIndexManager:
    """Thread-safe singleton holder with explicit rebuild."""

    def __init__(self) -> None:
        self._index: SearchIndex | None = None
        self._lock = threading.RLock()

    def get(self, db: Session) -> SearchIndex:
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = self._build(db)
        return self._index

    def rebuild(self, db: Session) -> SearchIndex:
        with self._lock:
            self._index = self._build(db)
            return self._index

    def peek(self) -> SearchIndex | None:
        return self._index

    def invalidate(self) -> None:
        with self._lock:
            self._index = None

    def _build(self, db: Session) -> SearchIndex:
        started = time.perf_counter()
        rows = db.execute(
            select(Product, Category.slug, Category.name, Brand.name)
            .join(Category, Category.id == Product.category_id)
            .join(Brand, Brand.id == Product.brand_id)
        ).all()

        entries: dict[int, ProductIndexEntry] = {}
        documents: list[tuple[int, str]] = []
        texts: list[str] = []
        ids: list[int] = []
        title_terms: dict[str, set[int]] = {}
        suggestion_counter: dict[str, int] = {}

        for product, category_slug, category_name, brand_name in rows:
            tags = product.tags or []
            specs = product.specifications or {}
            searchable = " ".join([
                product.title, product.title,   # title weighted twice in the lexical field
                category_name, brand_name, " ".join(tags),
                " ".join(f"{k} {v}" for k, v in specs.items()),
                product.description,
            ])
            entry = ProductIndexEntry(
                id=product.id, title=product.title, sku=product.sku,
                category_id=product.category_id, category_slug=category_slug, category_name=category_name,
                brand_id=product.brand_id, brand_name=brand_name,
                price=float(product.price), effective_price=float(product.effective_price),
                discount_pct=float(product.discount_pct), rating_avg=float(product.rating_avg),
                rating_count=int(product.rating_count), inventory=int(product.inventory),
                is_active=bool(product.is_active), tags=list(tags),
            )
            entries[product.id] = entry
            documents.append((product.id, searchable))
            texts.append(searchable)
            ids.append(product.id)

            for term in set(tokenize(f"{product.title} {' '.join(tags)} {brand_name} {category_name}")):
                title_terms.setdefault(term, set()).add(product.id)
                suggestion_counter[term] = suggestion_counter.get(term, 0) + 1

        bm25 = BM25Index.build(documents)

        embedder = build_embedder(dim=128)
        vectors_matrix = embedder.fit_transform(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        store = build_vector_store(dim=int(vectors_matrix.shape[1]) if vectors_matrix.size else 1)
        if vectors_matrix.size:
            try:
                store.upsert(ids, vectors_matrix)
            except Exception as exc:  # noqa: BLE001 - remote vector db problems -> local store
                logger.warning("vector_upsert_failed_using_numpy", error=str(exc))
                store = NumpyVectorStore()
                store.upsert(ids, vectors_matrix)

        suggestions = [
            term for term, _ in sorted(suggestion_counter.items(), key=lambda kv: -kv[1]) if len(term) > 2
        ][:4000]

        build_ms = (time.perf_counter() - started) * 1000.0
        logger.info("search_index_built", documents=len(entries), vocabulary=len(bm25.vocabulary),
                    build_ms=round(build_ms, 1))
        return SearchIndex(
            entries=entries, bm25=bm25, vectors=store, embedder=embedder,
            title_terms=title_terms, suggestions=suggestions,
            built_at=datetime.now(timezone.utc), build_ms=build_ms,
        )


_manager = SearchIndexManager()


def get_index_manager() -> SearchIndexManager:
    return _manager
