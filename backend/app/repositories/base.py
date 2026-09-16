"""Generic repository base - keeps query construction out of the service layer."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.core.db import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, db: Session):
        self.db = db

    def get(self, entity_id: int) -> ModelT | None:
        return self.db.get(self.model, entity_id)

    def get_many(self, ids: Sequence[int]) -> list[ModelT]:
        if not ids:
            return []
        return list(self.db.execute(select(self.model).where(self.model.id.in_(list(ids)))).scalars().all())

    def count(self, stmt: Select | None = None) -> int:
        base = stmt if stmt is not None else select(self.model)
        return int(self.db.execute(select(func.count()).select_from(base.subquery())).scalar_one() or 0)

    def paginate(self, stmt: Select, page: int = 1, page_size: int = 20) -> tuple[list[Any], int]:
        total = self.count(stmt)
        offset = max(page - 1, 0) * page_size
        rows = list(self.db.execute(stmt.offset(offset).limit(page_size)).scalars().all())
        return rows, total

    def add(self, entity: ModelT) -> ModelT:
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity
