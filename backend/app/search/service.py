"""Hybrid search service: lexical + semantic + filters + ranking + personalization."""
from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from ml.features.embeddings import tokenize
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import get_cache
from app.core.logging_config import get_logger
from app.models import SearchEvent
from app.search.index import ProductIndexEntry, SearchIndex, get_index_manager
from app.search.ranking import (
    RankedResult,
    RankingSignals,
    default_weights,
    explain,
    minmax_normalise,
    normalise_weights,
    score_result,
)

logger = get_logger("search")

CANDIDATE_POOL = 300
SEARCH_CACHE_TTL = 120


@dataclass(slots=True)
class SearchFilters:
    category_slugs: list[str] = field(default_factory=list)
    category_ids: list[int] = field(default_factory=list)
    brand_ids: list[int] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None
    in_stock_only: bool = False
    on_sale_only: bool = False

    def cache_fragment(self) -> str:
        return "|".join([
            ",".join(sorted(self.category_slugs)), ",".join(str(c) for c in sorted(self.category_ids)),
            ",".join(str(b) for b in sorted(self.brand_ids)),
            str(self.min_price), str(self.max_price), str(self.min_rating),
            str(self.in_stock_only), str(self.on_sale_only),
        ])

    def matches(self, entry: ProductIndexEntry) -> bool:
        if not entry.is_active:
            return False
        if self.category_slugs and entry.category_slug not in self.category_slugs:
            return False
        if self.category_ids and entry.category_id not in self.category_ids:
            return False
        if self.brand_ids and entry.brand_id not in self.brand_ids:
            return False
        if self.min_price is not None and entry.effective_price < self.min_price:
            return False
        if self.max_price is not None and entry.effective_price > self.max_price:
            return False
        if self.min_rating is not None and entry.rating_avg < self.min_rating:
            return False
        if self.in_stock_only and entry.inventory <= 0:
            return False
        return not (self.on_sale_only and entry.discount_pct <= 0)


@dataclass
class SearchResponse:
    query: str
    normalized_query: str
    corrected_query: str | None
    total: int
    results: list[RankedResult]
    facets: dict[str, Any]
    took_ms: float
    strategy: str
    weights: dict[str, float]

    def product_ids(self) -> list[int]:
        return [r.product_id for r in self.results]


class SearchService:
    def __init__(self, db: Session):
        self.db = db
        self.index: SearchIndex = get_index_manager().get(db)
        self.cache = get_cache()

    # ---- query understanding -------------------------------------------
    @staticmethod
    def normalise(query: str) -> str:
        return re.sub(r"\s+", " ", (query or "").strip().lower())

    def correct_query(self, query: str) -> tuple[str, bool]:
        """Per-token typo correction against the index vocabulary."""
        tokens = tokenize(query, drop_stopwords=False)
        if not tokens:
            return query, False
        corrected: list[str] = []
        changed = False
        for token in tokens:
            if len(token) <= 3 or self.index.bm25.term_exists(token):
                corrected.append(token)
                continue
            candidates = self.index.bm25.closest_terms(token, max_distance=2, limit=1)
            if candidates and candidates[0] != token:
                corrected.append(candidates[0])
                changed = True
            else:
                corrected.append(token)
        return " ".join(corrected), changed

    def parse_intent(self, query: str) -> dict[str, Any]:
        """Pull structured constraints out of natural-language queries."""
        intent: dict[str, Any] = {"budget": None, "category_hint": None, "modifiers": []}
        lowered = query.lower()

        budget = re.search(r"under\s*\$?(\d+(?:\.\d+)?)", lowered) or re.search(r"below\s*\$?(\d+(?:\.\d+)?)", lowered)
        if budget:
            intent["budget"] = float(budget.group(1))
        cheap = any(word in lowered for word in ("cheap", "budget", "affordable", "inexpensive"))
        premium = any(word in lowered for word in ("premium", "best", "professional", "pro "))
        if cheap:
            intent["modifiers"].append("price_ascending")
        if premium:
            intent["modifiers"].append("quality_bias")
        if any(word in lowered for word in ("in stock", "available")):
            intent["modifiers"].append("in_stock")

        for entry in self.index.entries.values():
            if entry.category_name.lower() in lowered or entry.category_slug in lowered:
                intent["category_hint"] = entry.category_slug
                break
        return intent

    # ---- retrieval ------------------------------------------------------
    def _candidates(self, query: str, filters: SearchFilters) -> tuple[dict[int, float], dict[int, float]]:
        lexical = self.index.bm25.score(query, limit=CANDIDATE_POOL)

        semantic: dict[int, float] = {}
        if self.index.vectors.size():
            try:
                vector = self.index.embedder.encode_one(query)
                for pid, score in self.index.vectors.search(vector, k=CANDIDATE_POOL):
                    semantic[pid] = max(score, 0.0)
            except Exception as exc:  # noqa: BLE001 - semantic layer is best-effort
                logger.warning("semantic_search_failed", error=str(exc))
        return lexical, semantic

    def search(self, query: str, *, filters: SearchFilters | None = None, limit: int = 20, offset: int = 0,
               sort: str = "relevance", user_profile: Any = None, weights: dict[str, float] | None = None,
               use_cache: bool = True) -> SearchResponse:
        started = time.perf_counter()
        filters = filters or SearchFilters()
        normalized = self.normalise(query)
        active_weights = normalise_weights({**default_weights(), **(weights or {})})

        cache_key = None
        if use_cache and user_profile is None and sort == "relevance":
            cache_key = self.cache.key("search", normalized, filters.cache_fragment(), limit, offset)
            cached = self.cache.get_json(cache_key)
            if cached is not None:
                return SearchResponse(
                    query=query, normalized_query=normalized, corrected_query=cached.get("corrected_query"),
                    total=cached["total"],
                    results=[
                        RankedResult(r["product_id"], r["score"], RankingSignals(**r["signals"]), r["explanation"])
                        for r in cached["results"]
                    ],
                    facets=cached["facets"], took_ms=round((time.perf_counter() - started) * 1000, 3),
                    strategy=cached["strategy"] + "+cache", weights=active_weights,
                )

        corrected, changed = self.correct_query(normalized)
        effective_query = corrected if changed else normalized
        intent = self.parse_intent(normalized)
        if intent["budget"] is not None and filters.max_price is None:
            filters.max_price = intent["budget"]
        if "in_stock" in intent["modifiers"]:
            filters.in_stock_only = True

        if not effective_query:
            # Empty query: browse mode, ranked by popularity within the filters.
            eligible = [e for e in self.index.entries.values() if filters.matches(e)]
            lexical_raw, semantic_raw = {}, {}
            candidate_ids = [e.id for e in eligible]
            strategy = "browse"
        else:
            lexical_raw, semantic_raw = self._candidates(effective_query, filters)
            candidate_ids = [pid for pid in set(lexical_raw) | set(semantic_raw)
                             if pid in self.index.entries and filters.matches(self.index.entries[pid])]
            strategy = "hybrid"
            if not candidate_ids:
                # Nothing matched: fall back to semantic-only over the whole catalogue.
                eligible = {e.id for e in self.index.entries.values() if filters.matches(e)}
                try:
                    vector = self.index.embedder.encode_one(effective_query)
                    semantic_raw = {pid: max(s, 0.0)
                                    for pid, s in self.index.vectors.search(vector, k=limit + offset + 40,
                                                                            allowed_ids=eligible)}
                    candidate_ids = list(semantic_raw)
                    strategy = "semantic_fallback"
                except Exception:  # noqa: BLE001
                    candidate_ids = []

        lexical = minmax_normalise({k: v for k, v in lexical_raw.items() if k in candidate_ids})
        semantic = minmax_normalise({k: v for k, v in semantic_raw.items() if k in candidate_ids})

        popularity_raw = {pid: self.index.entries[pid].rating_count for pid in candidate_ids}
        popularity = minmax_normalise({k: float(np.log1p(v)) for k, v in popularity_raw.items()})

        results: list[RankedResult] = []
        for pid in candidate_ids:
            entry = self.index.entries[pid]
            signals = RankingSignals(
                text=lexical.get(pid, 0.0),
                semantic=semantic.get(pid, 0.0),
                popularity=popularity.get(pid, 0.0),
                rating=min(entry.rating_avg / 5.0, 1.0),
                conversion=entry.conversion_rate,
                personal=self._personal_score(entry, user_profile),
                availability=1.0 if entry.inventory > 0 else 0.0,
            )
            if "quality_bias" in intent["modifiers"]:
                signals.rating = min(signals.rating * 1.25, 1.0)
            score = score_result(signals, active_weights)
            results.append(RankedResult(pid, score, signals, explain(signals, active_weights)))

        results = self._sort(results, sort, intent)
        total = len(results)
        page = results[offset: offset + limit]
        facets = self._facets(candidate_ids)
        took_ms = round((time.perf_counter() - started) * 1000, 3)

        if cache_key is not None:
            self.cache.set_json(cache_key, {
                "total": total,
                "results": [r.as_dict() for r in page],
                "facets": facets,
                "strategy": strategy,
                "corrected_query": corrected if changed else None,
            }, SEARCH_CACHE_TTL)

        return SearchResponse(
            query=query, normalized_query=normalized, corrected_query=corrected if changed else None,
            total=total, results=page, facets=facets, took_ms=took_ms, strategy=strategy,
            weights=active_weights,
        )

    @staticmethod
    def _personal_score(entry: ProductIndexEntry, profile: Any) -> float:
        if profile is None:
            return 0.0
        score = 0.0
        category_affinity = getattr(profile, "category_affinity", {}) or {}
        brand_affinity = getattr(profile, "brand_affinity", {}) or {}
        score += 0.6 * float(category_affinity.get(entry.category_id, 0.0))
        score += 0.25 * float(brand_affinity.get(entry.brand_id, 0.0))
        price_range = getattr(profile, "preferred_price_range", None)
        if price_range and price_range[0] <= entry.effective_price <= price_range[1]:
            score += 0.15
        return min(score, 1.0)

    @staticmethod
    def _sort(results: list[RankedResult], sort: str, intent: dict[str, Any]) -> list[RankedResult]:
        from app.search.index import get_index_manager  # local import avoids a cycle at module load

        index = get_index_manager().peek()
        entries = index.entries if index else {}

        if sort == "price_asc" or "price_ascending" in intent.get("modifiers", []):
            return sorted(results, key=lambda r: entries[r.product_id].effective_price if r.product_id in entries else 0)
        if sort == "price_desc":
            return sorted(results, key=lambda r: -(entries[r.product_id].effective_price if r.product_id in entries else 0))
        if sort == "rating":
            return sorted(results, key=lambda r: -(entries[r.product_id].rating_avg if r.product_id in entries else 0))
        if sort == "newest":
            return sorted(results, key=lambda r: -r.product_id)
        if sort == "discount":
            return sorted(results, key=lambda r: -(entries[r.product_id].discount_pct if r.product_id in entries else 0))
        return sorted(results, key=lambda r: -r.score)

    def _facets(self, candidate_ids: Sequence[int]) -> dict[str, Any]:
        categories: dict[str, dict[str, Any]] = {}
        brands: dict[str, dict[str, Any]] = {}
        prices: list[float] = []
        ratings = {"4+": 0, "3+": 0, "2+": 0, "any": 0}
        in_stock = 0
        on_sale = 0

        for pid in candidate_ids:
            entry = self.index.entries.get(pid)
            if entry is None:
                continue
            cat = categories.setdefault(entry.category_slug, {"slug": entry.category_slug,
                                                              "name": entry.category_name, "count": 0})
            cat["count"] += 1
            brand = brands.setdefault(str(entry.brand_id), {"id": entry.brand_id, "name": entry.brand_name,
                                                            "count": 0})
            brand["count"] += 1
            prices.append(entry.effective_price)
            ratings["any"] += 1
            if entry.rating_avg >= 4:
                ratings["4+"] += 1
            if entry.rating_avg >= 3:
                ratings["3+"] += 1
            if entry.rating_avg >= 2:
                ratings["2+"] += 1
            in_stock += 1 if entry.inventory > 0 else 0
            on_sale += 1 if entry.discount_pct > 0 else 0

        return {
            "categories": sorted(categories.values(), key=lambda c: -c["count"])[:15],
            "brands": sorted(brands.values(), key=lambda b: -b["count"])[:15],
            "price": {
                "min": round(min(prices), 2) if prices else 0.0,
                "max": round(max(prices), 2) if prices else 0.0,
                "avg": round(float(np.mean(prices)), 2) if prices else 0.0,
            },
            "rating": ratings,
            "availability": {"in_stock": in_stock, "out_of_stock": len(candidate_ids) - in_stock},
            "on_sale": on_sale,
        }

    # ---- suggestions ----------------------------------------------------
    def autocomplete(self, prefix: str, limit: int = 8) -> list[dict[str, Any]]:
        prefix = self.normalise(prefix)
        if len(prefix) < 2:
            return []
        key = self.cache.key("autocomplete", prefix, limit)
        cached = self.cache.get_json(key)
        if cached is not None:
            return cached

        suggestions: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1. product titles that contain the prefix
        for entry in self.index.entries.values():
            if prefix in entry.title.lower() and entry.title.lower() not in seen:
                seen.add(entry.title.lower())
                suggestions.append({"text": entry.title, "type": "product", "product_id": entry.id})
            if len(suggestions) >= limit:
                break

        # 2. historical queries that performed well
        if len(suggestions) < limit:
            rows = self.db.execute(
                select(SearchEvent.normalized_query, func.count(SearchEvent.id).label("n"))
                .where(SearchEvent.normalized_query.like(f"{prefix}%"))
                .group_by(SearchEvent.normalized_query)
                .order_by(func.count(SearchEvent.id).desc())
                .limit(limit)
            ).all()
            for text, count in rows:
                if text and text not in seen:
                    seen.add(text)
                    suggestions.append({"text": text, "type": "query", "popularity": int(count)})

        # 3. vocabulary terms
        if len(suggestions) < limit:
            for term in self.index.suggestions:
                if term.startswith(prefix) and term not in seen:
                    seen.add(term)
                    suggestions.append({"text": term, "type": "term"})
                if len(suggestions) >= limit:
                    break

        result = suggestions[:limit]
        self.cache.set_json(key, result, 300)
        return result

    def popular_queries(self, limit: int = 10, days: int = 30) -> list[dict[str, Any]]:
        from datetime import timedelta

        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.db.execute(
            select(
                SearchEvent.normalized_query,
                func.count(SearchEvent.id).label("searches"),
                func.sum(func.coalesce(SearchEvent.clicked_product_id, 0) > 0).label("clicks"),
            )
            .where(SearchEvent.occurred_at >= since, SearchEvent.normalized_query != "")
            .group_by(SearchEvent.normalized_query)
            .order_by(func.count(SearchEvent.id).desc())
            .limit(limit)
        ).all()
        out = []
        for query, searches, clicks in rows:
            searches = int(searches or 0)
            clicks = int(clicks or 0)
            out.append({
                "query": query,
                "searches": searches,
                "clicks": clicks,
                "ctr": round(clicks / searches, 4) if searches else 0.0,
            })
        return out

    def history(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute(
            select(SearchEvent.query, SearchEvent.occurred_at, SearchEvent.result_count)
            .where(SearchEvent.user_id == user_id)
            .order_by(SearchEvent.occurred_at.desc())
            .limit(limit)
        ).all()
        return [
            {"query": q, "occurred_at": ts.isoformat() if ts else None, "result_count": int(rc or 0)}
            for q, ts, rc in rows
        ]

    def log_search(self, *, query: str, user_id: int | None, session_id: str, result_count: int,
                   latency_ms: float, filters: dict[str, Any] | None = None) -> None:
        try:
            self.db.add(SearchEvent(
                user_id=user_id, session_id=session_id or "anonymous", query=query[:255],
                normalized_query=self.normalise(query)[:255], result_count=result_count,
                latency_ms=latency_ms, filters=filters or {}, occurred_at=datetime.now(timezone.utc),
            ))
            self.db.commit()
        except Exception as exc:  # noqa: BLE001 - telemetry must not break search
            self.db.rollback()
            logger.warning("search_log_failed", error=str(exc))
