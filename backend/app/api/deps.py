"""Shared FastAPI dependencies."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Query, Request
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import Principal, get_optional_principal, require_admin

DbSession = Annotated[Session, Depends(get_db)]
AdminPrincipal = Annotated[Principal, Depends(require_admin)]
OptionalPrincipal = Annotated[Principal | None, Depends(get_optional_principal)]


class Pagination:
    """Standard page/page_size pair, bounded to protect the database."""

    def __init__(
        self,
        page: int = Query(1, ge=1, le=10_000, description="1-based page number"),
        page_size: int = Query(24, ge=1, le=100, description="Items per page (max 100)"),
    ):
        self.page = page
        self.page_size = page_size

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


PaginationDep = Annotated[Pagination, Depends()]


def get_session_id(
    request: Request,
    x_session_id: str | None = Header(None, alias="X-Session-Id",
                                         description="Client-generated session identifier"),
) -> str:
    """Resolve a session id from the header, falling back to the client address."""
    if x_session_id:
        return x_session_id[:64]
    client = request.client.host if request.client else "anonymous"
    return f"anon-{client}"[:64]


SessionId = Annotated[str, Depends(get_session_id)]


def resolve_user_id(
    user_id: int | None = Query(None, ge=1, description="Acting user id (omitted for anonymous traffic)"),
    principal: Principal | None = Depends(get_optional_principal),
) -> int | None:
    """Prefer the authenticated subject; fall back to an explicit query parameter."""
    if principal is not None and principal.subject.isdigit():
        return int(principal.subject)
    return user_id


ActingUser = Annotated[int | None, Depends(resolve_user_id)]
