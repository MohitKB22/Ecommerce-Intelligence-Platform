"""Application error hierarchy and FastAPI exception handlers.

Stack traces are never exposed to clients; every error becomes a stable,
machine-readable envelope.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging_config import get_logger, request_id_ctx

logger = get_logger("errors")


class AppError(Exception):
    """Base class for all expected application failures."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: Any = None, code: str | None = None):
        self.message = message or self.message
        self.details = details
        if code:
            self.code = code
        super().__init__(self.message)

    def to_payload(self) -> dict:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": request_id_ctx.get(),
            }
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        return payload


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "The requested resource was not found."


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"
    message = "The request payload failed validation."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "The request conflicts with the current state."


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"
    message = "Authentication credentials were missing or invalid."


class AuthorizationError(AppError):
    status_code = 403
    code = "forbidden"
    message = "You do not have permission to perform this action."


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Too many requests. Please slow down."


class DependencyError(AppError):
    """A downstream dependency (DB, cache, vector store) failed."""

    status_code = 503
    code = "dependency_unavailable"
    message = "A required downstream service is unavailable."


class ModelNotAvailableError(AppError):
    """A model artefact is missing; callers should degrade gracefully."""

    status_code = 503
    code = "model_unavailable"
    message = "The requested model is not currently available."


class InferenceError(AppError):
    status_code = 500
    code = "inference_failed"
    message = "Model inference failed."


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_request: Request, exc: AppError) -> JSONResponse:
        log = logger.warning if exc.status_code < 500 else logger.error
        log("app_error", code=exc.code, status=exc.status_code, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e.get("loc", [])[1:]) or "body", "issue": e.get("msg", "invalid")}
            for e in exc.errors()
        ]
        err = ValidationError(details=details)
        logger.warning("request_validation_error", details=details)
        return JSONResponse(status_code=err.status_code, content=err.to_payload())

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        mapping = {
            401: AuthenticationError,
            403: AuthorizationError,
            404: NotFoundError,
            409: ConflictError,
            429: RateLimitError,
        }
        cls = mapping.get(exc.status_code, AppError)
        err = cls(str(exc.detail) if exc.detail else None)
        err.status_code = exc.status_code
        return JSONResponse(status_code=exc.status_code, content=err.to_payload())

    @app.exception_handler(IntegrityError)
    async def _integrity(_request: Request, exc: IntegrityError) -> JSONResponse:
        """A constraint violation is a client conflict, not a server fault."""
        logger.warning("integrity_error", error=str(exc.orig) if exc.orig else str(exc))
        err = ConflictError(
            "That operation conflicts with existing data (a uniqueness or "
            "referential constraint was violated)."
        )
        return JSONResponse(status_code=err.status_code, content=err.to_payload())

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), error_type=type(exc).__name__, exc_info=True)
        return JSONResponse(status_code=500, content=AppError().to_payload())
