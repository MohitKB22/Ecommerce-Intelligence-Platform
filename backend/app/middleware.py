"""Custom middleware: request correlation, access logging, rate limiting, headers."""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.cache import get_cache
from app.core.config import settings
from app.core.errors import RateLimitError
from app.core.logging_config import get_logger, new_request_id, request_id_ctx
from app.core.security import client_identity

logger = get_logger("http")

# Probes and docs must not be rate limited or they will flap under load.
EXEMPT_PATHS = ("/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready", "/docs", "/redoc",
                "/openapi.json", "/metrics")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request and emit one structured access log."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-Id") or new_request_id()
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "request_failed", method=request.method, path=request.url.path,
                duration_ms=round(duration_ms, 2), exc_info=True,
            )
            request_id_ctx.reset(token)
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        if request.url.path not in ("/api/v1/health/live",):
            logger.info(
                "request_completed", method=request.method, path=request.url.path,
                status=response.status_code, duration_ms=round(duration_ms, 2),
                query=str(request.url.query)[:200],
            )
        request_id_ctx.reset(token)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-XSS-Protection", "0")  # modern browsers: CSP instead
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        if settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline'",
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client rate limiting backed by the cache layer."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not settings.RATE_LIMIT_ENABLED or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        identity = client_identity(request)
        allowed, count = get_cache().rate_limit(
            identity, settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS
        )
        if not allowed:
            error = RateLimitError(
                f"Rate limit of {settings.RATE_LIMIT_REQUESTS} requests per "
                f"{settings.RATE_LIMIT_WINDOW_SECONDS}s exceeded."
            )
            logger.warning("rate_limited", identity=identity, count=count, path=request.url.path)
            return JSONResponse(
                status_code=error.status_code, content=error.to_payload(),
                headers={
                    "Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS),
                    "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(max(settings.RATE_LIMIT_REQUESTS - count, 0))
        return response
