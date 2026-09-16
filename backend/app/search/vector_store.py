"""Vector store abstraction.

Two backends implement the same interface:
  * NumpyVectorStore  - in-process exact cosine search (default, zero setup)
  * PgVectorStore     - PostgreSQL + pgvector, used when VECTOR_DB_URL is set
                        and the extension is available

Exact search over a catalogue of this size is fast (a single dense matmul), so
the numpy backend is a legitimate production choice for small/medium catalogues
rather than a stub.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from app.core.logging_config import get_logger

logger = get_logger("vector_store")


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: Sequence[int], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 20,
               allowed_ids: set[int] | None = None) -> list[tuple[int, float]]: ...

    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def backend(self) -> str: ...


class NumpyVectorStore(VectorStore):
    def __init__(self) -> None:
        self._ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._position: dict[int, int] = {}

    @property
    def backend(self) -> str:
        return "numpy"

    def upsert(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must align")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.where(norms == 0, 1.0, norms)
        self._ids = np.asarray(ids, dtype=np.int64)
        self._position = {int(v): i for i, v in enumerate(self._ids)}

    def search(self, query: np.ndarray, k: int = 20,
               allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        if self._vectors.size == 0:
            return []
        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        scores = self._vectors @ (query / norm)

        if allowed_ids is not None:
            mask = np.zeros(len(self._ids), dtype=bool)
            for pid in allowed_ids:
                pos = self._position.get(int(pid))
                if pos is not None:
                    mask[pos] = True
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)

        k = min(k, len(self._ids))
        top = np.argpartition(-scores, kth=k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(int(self._ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def vector_for(self, entity_id: int) -> np.ndarray | None:
        pos = self._position.get(int(entity_id))
        return self._vectors[pos] if pos is not None else None

    def size(self) -> int:
        return int(len(self._ids))


class PgVectorStore(VectorStore):  # pragma: no cover - requires a live PostgreSQL+pgvector
    """PostgreSQL/pgvector backend. Falls back to numpy if the extension is absent."""

    def __init__(self, dsn: str, dim: int, table: str = "product_embeddings"):
        from sqlalchemy import create_engine, text

        self._engine = create_engine(dsn, pool_pre_ping=True)
        self._table = table
        self._dim = dim
        self._ready = False
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"  product_id INTEGER PRIMARY KEY,"
                f"  embedding vector({dim})"
                f")"
            ))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_embedding ON {table} "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            ))
        self._ready = True

    @property
    def backend(self) -> str:
        return "pgvector"

    def upsert(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        from sqlalchemy import text

        vectors = np.asarray(vectors, dtype=np.float32)
        with self._engine.begin() as conn:
            for entity_id, vector in zip(ids, vectors):
                literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
                conn.execute(
                    text(f"INSERT INTO {self._table} (product_id, embedding) VALUES (:pid, :emb) "
                         f"ON CONFLICT (product_id) DO UPDATE SET embedding = EXCLUDED.embedding"),
                    {"pid": int(entity_id), "emb": literal},
                )

    def search(self, query: np.ndarray, k: int = 20,
               allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        from sqlalchemy import text

        literal = "[" + ",".join(f"{v:.6f}" for v in np.asarray(query, dtype=np.float32).ravel()) + "]"
        sql = f"SELECT product_id, 1 - (embedding <=> :q) AS score FROM {self._table}"
        params: dict = {"q": literal, "k": k}
        if allowed_ids:
            sql += " WHERE product_id = ANY(:ids)"
            params["ids"] = list(allowed_ids)
        sql += " ORDER BY embedding <=> :q LIMIT :k"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).all()
        return [(int(pid), float(score)) for pid, score in rows]

    def size(self) -> int:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {self._table}")).scalar_one())


def build_vector_store(dim: int) -> VectorStore:
    from app.core.config import settings

    if settings.VECTOR_DB_URL:
        try:
            store = PgVectorStore(settings.VECTOR_DB_URL, dim)
            logger.info("vector_store_selected", backend="pgvector")
            return store
        except Exception as exc:  # noqa: BLE001 - any pgvector problem -> numpy
            logger.warning("pgvector_unavailable_falling_back", error=str(exc))
    logger.info("vector_store_selected", backend="numpy")
    return NumpyVectorStore()
