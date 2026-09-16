"""Password hashing, JWT issuance/verification and auth dependencies."""
from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings
from app.core.errors import AuthenticationError, AuthorizationError

_PBKDF2_ROUNDS = 260_000
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with a per-password random salt."""
    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt_hex, hash_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def create_access_token(subject: str, *, role: str = "user", expires_minutes: int | None = None,
                        extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload: dict[str, Any] = {"sub": str(subject), "role": role, "iat": now, "exp": expire, "iss": "ecommerce-intelligence"}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM], issuer="ecommerce-intelligence")
    except JWTError as exc:
        raise AuthenticationError("Token is invalid or has expired.") from exc


class Principal:
    __slots__ = ("subject", "role", "claims")

    def __init__(self, subject: str, role: str, claims: dict[str, Any]):
        self.subject = subject
        self.role = role
        self.claims = claims

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Principal(subject={self.subject!r}, role={self.role!r})"


def _principal_from_credentials(creds: HTTPAuthorizationCredentials | None) -> Principal | None:
    if creds is None or not creds.credentials:
        return None
    claims = decode_token(creds.credentials)
    return Principal(str(claims.get("sub", "")), str(claims.get("role", "user")), claims)


def get_current_principal(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
    principal = _principal_from_credentials(creds)
    if principal is None:
        raise AuthenticationError()
    return principal


def get_optional_principal(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal | None:
    try:
        return _principal_from_credentials(creds)
    except AuthenticationError:
        return None


def require_admin(principal: Principal = Depends(get_current_principal)) -> Principal:
    if not principal.is_admin:
        raise AuthorizationError("Administrator privileges are required for this endpoint.")
    return principal


def client_identity(request: Request) -> str:
    """Best-effort client identity for rate limiting."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"
