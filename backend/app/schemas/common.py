"""Shared response envelopes and pagination."""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PageMeta(BaseModel):
    page: int = Field(..., ge=1, description="Current 1-based page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total matching items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool
    has_previous: bool

    @classmethod
    def build(cls, page: int, page_size: int, total: int) -> PageMeta:
        total_pages = (total + page_size - 1) // page_size if page_size else 0
        return cls(
            page=page, page_size=page_size, total=total, total_pages=total_pages,
            has_next=page < total_pages, has_previous=page > 1,
        )


class Page(BaseModel, Generic[T]):
    items: list[T]
    meta: PageMeta


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["not_found"])
    message: str = Field(..., examples=["The requested resource was not found."])
    request_id: str = Field(..., examples=["a1b2c3d4e5f6a7b8"])
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class MessageResponse(BaseModel):
    message: str
    detail: Any | None = None
