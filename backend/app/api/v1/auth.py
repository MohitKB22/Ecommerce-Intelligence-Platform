"""Authentication endpoints (JWT)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.api.deps import DbSession
from app.core.config import settings
from app.core.errors import AuthenticationError
from app.core.security import Principal, create_access_token, get_current_principal, verify_password
from app.models import User
from app.schemas.users import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse, summary="Exchange credentials for a JWT",
             responses={401: {"description": "Invalid email or password"}})
def login(payload: LoginRequest, db: DbSession) -> TokenResponse:
    """Issue an access token. Admin endpoints require a token with the `admin` role."""
    user = db.execute(select(User).where(User.email == payload.email)).scalars().first()
    # Constant-ish response regardless of which half failed, to avoid user enumeration.
    if user is None or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise AuthenticationError("Invalid email or password.")
    if not user.is_active:
        raise AuthenticationError("This account is disabled.")

    token = create_access_token(str(user.id), role=user.role, extra={"email": user.email})
    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        role=user.role,
        user_id=user.id,
    )


@router.get("/me", response_model=UserOut, summary="Current authenticated user",
            responses={401: {"description": "Missing or invalid token"}})
def me(db: DbSession, principal: Principal = Depends(get_current_principal)) -> UserOut:
    """Resolve the token subject back to the full user record."""
    if not principal.subject.isdigit():
        raise AuthenticationError("Token subject is not a valid user id.")
    user = db.get(User, int(principal.subject))
    if user is None or not user.is_active:
        raise AuthenticationError("The account for this token no longer exists.")
    return UserOut.model_validate(user)
