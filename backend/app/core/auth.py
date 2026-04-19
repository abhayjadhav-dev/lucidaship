"""
Clerk JWT authentication — verifies tokens issued by Clerk.
No login/register endpoints needed — Clerk owns all user identity.
"""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional

import httpx
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database import get_db
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Bearer scheme for FastAPI dependency injection ───────────────
bearer_scheme = HTTPBearer()

# ── JWKS Cache ───────────────────────────────────────────────────
_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_timestamp: float = 0
JWKS_CACHE_TTL = 3600  # 1 hour in seconds
CLERK_JWKS_URL = "https://api.clerk.com/v1/jwks"


def _normalize_origin(value: str) -> str:
    return value.strip().rstrip("/")


def _allowed_azp_origins() -> List[str]:
    """Resolve allowed token authorized-party origins."""
    explicit = settings.clerk_allowed_azp_origins_list
    if explicit:
        return explicit

    return [
        _normalize_origin(origin)
        for origin in settings.cors_origins_list
        if origin.strip() and origin.strip() != "*"
    ]


def _validate_authorized_party(payload: Dict[str, Any]) -> None:
    """Validate `azp` claim against configured frontend origins when present."""
    azp = payload.get("azp")
    if not azp:
        return

    allowed = set(_allowed_azp_origins())
    if allowed and _normalize_origin(str(azp)) not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token authorized party is not allowed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate_audience(payload: Dict[str, Any]) -> None:
    """Validate optional audience claim against configured accepted audiences."""
    expected = set(settings.clerk_jwt_audience_list)
    if not expected:
        return

    aud = payload.get("aud")
    if isinstance(aud, str):
        actual = {aud}
    elif isinstance(aud, list):
        actual = {str(item) for item in aud}
    else:
        actual = set()

    if not actual or expected.isdisjoint(actual):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token audience is not allowed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _fetch_jwks() -> Dict[str, Any]:
    """
    Fetch Clerk's JWKS (JSON Web Key Set) from their API.
    Caches the result for 1 hour to avoid hitting Clerk on every request.
    """
    global _jwks_cache, _jwks_cache_timestamp

    now = time.time()
    if _jwks_cache and (now - _jwks_cache_timestamp) < JWKS_CACHE_TTL:
        return _jwks_cache

    try:
        headers = {"Authorization": f"Bearer {settings.CLERK_SECRET_KEY}"}
        response = httpx.get(CLERK_JWKS_URL, headers=headers, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_timestamp = now
        logger.info("Clerk JWKS refreshed (%d keys)", len(_jwks_cache.get("keys", [])))
        return _jwks_cache
    except Exception as e:
        logger.error("Failed to fetch Clerk JWKS: %s", e)
        if _jwks_cache:
            logger.warning("Using stale JWKS cache")
            return _jwks_cache
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify authentication (JWKS fetch failed)",
        )


def _get_signing_key(token: str) -> Dict[str, Any]:
    """
    Extract the signing key from JWKS that matches the token's kid header.
    """
    jwks = _fetch_jwks()
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'kid' header",
        )

    for key in jwks.get("keys", []):
        if key["kid"] == kid:
            return key

    # kid not found — force refresh JWKS once
    global _jwks_cache_timestamp
    _jwks_cache_timestamp = 0
    jwks = _fetch_jwks()

    for key in jwks.get("keys", []):
        if key["kid"] == kid:
            return key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No matching key found in Clerk JWKS",
    )


def verify_clerk_token(token: str) -> Dict[str, Any]:
    """
    Verify a Clerk-issued JWT:
    1. Fetch JWKS (cached)
    2. Find matching signing key by kid
    3. Verify signature + exp claim
    4. Return decoded payload

    Raises HTTPException 401 on any failure.
    """
    try:
        signing_key = _get_signing_key(token)

        issuer = settings.clerk_jwt_issuer
        decode_kwargs: Dict[str, Any] = {}
        if issuer:
            decode_kwargs["issuer"] = issuer

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            leeway=max(0, settings.CLERK_JWT_LEEWAY_SECONDS),
            options={
                "verify_exp": True,
                "verify_aud": False,
                "verify_iss": bool(issuer),
            },
            **decode_kwargs,
        )

        _validate_audience(payload)
        _validate_authorized_party(payload)
        return payload

    except JWTError as e:
        logger.warning("JWT verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected auth error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


def _auto_provision_user(clerk_user_id: str, email: str, conn) -> Dict[str, Any]:
    """
    Auto-provision a new tenant + user when a Clerk user hits our API
    for the first time. Returns user dict.
    """
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    # Create tenant (personal org for the user)
    conn.execute(
        "INSERT INTO tenants (id, clerk_org_id, name, plan) VALUES (?, ?, ?, ?)",
        [tenant_id, None, f"{email.split('@')[0]}'s Organization", "free"],
    )

    # Create user
    conn.execute(
        "INSERT INTO users (id, clerk_user_id, tenant_id, email, role) VALUES (?, ?, ?, ?, ?)",
        [user_id, clerk_user_id, tenant_id, email, "admin"],
    )

    logger.info("Auto-provisioned user: clerk_id=%s email=%s tenant=%s", clerk_user_id, email, tenant_id)

    return {
        "id": user_id,
        "clerk_user_id": clerk_user_id,
        "tenant_id": tenant_id,
        "email": email,
        "role": "admin",
    }


def _local_dev_user() -> Dict[str, Any]:
    """
    Return (or create) a local development user for running without Clerk.
    Uses a fixed tenant/user ID so the experience is consistent across restarts.
    """
    LOCAL_TENANT_ID = "local-dev-tenant"
    LOCAL_USER_ID = "local-dev-user"
    LOCAL_EMAIL = "dev@localhost"

    conn = get_db()

    try:
        # Ensure tenant exists
        result = conn.execute("SELECT id FROM tenants WHERE id = ?", [LOCAL_TENANT_ID])
        if not result.rows:
            conn.execute(
                "INSERT INTO tenants (id, clerk_org_id, name, plan) VALUES (?, ?, ?, ?)",
                [LOCAL_TENANT_ID, None, "Local Development", "free"],
            )

        # Ensure user exists
        result = conn.execute("SELECT id FROM users WHERE id = ?", [LOCAL_USER_ID])
        if not result.rows:
            conn.execute(
                "INSERT INTO users (id, clerk_user_id, tenant_id, email, role) VALUES (?, ?, ?, ?, ?)",
                [LOCAL_USER_ID, "local_clerk_id", LOCAL_TENANT_ID, LOCAL_EMAIL, "admin"],
            )
    except Exception as exc:
        logger.warning("Local dev auth bootstrap could not persist user records: %s", exc)

    return {
        "id": LOCAL_USER_ID,
        "clerk_user_id": "local_clerk_id",
        "tenant_id": LOCAL_TENANT_ID,
        "email": LOCAL_EMAIL,
        "role": "admin",
    }


# Use Optional bearer scheme so requests without tokens don't 403 in dev mode
_optional_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
) -> Dict[str, Any]:
    """
    FastAPI dependency — verifies Clerk JWT and returns user dict.

    In LOCAL DEV mode (no CLERK_SECRET_KEY), bypasses auth entirely
    and returns a local dev user so the API is fully usable without Clerk.

    Flow (production):
    1. Extract Bearer token
    2. Verify with Clerk JWKS
    3. Look up user in our DB by clerk_user_id
    4. If not found, auto-provision (create tenant + user)
    5. Return user dict with id, tenant_id, email, role

    Usage:
        @router.get("/protected")
        def my_route(user: dict = Depends(get_current_user)):
            tenant_id = user["tenant_id"]
    """
    # ── LOCAL DEV BYPASS (opt-in only) ───────────────────────
    if not settings.CLERK_SECRET_KEY:
        if settings.is_production or not settings.AUTH_BYPASS_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Authentication is not configured. Set CLERK_SECRET_KEY "
                    "or enable AUTH_BYPASS_ENABLED for local development only."
                ),
            )

        logger.debug("No CLERK_SECRET_KEY — using local dev user (auth bypassed)")
        return _local_dev_user()

    # ── PRODUCTION: Clerk JWT verification ───────────────────
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = verify_clerk_token(token)

    # Extract Clerk user ID from JWT subject (sub claim)
    clerk_user_id = payload.get("sub")
    if not clerk_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'sub' claim",
        )

    # Extract email from Clerk JWT payload
    # Clerk stores email in various places depending on version
    email = (
        payload.get("email")
        or payload.get("email_address")
        or payload.get("primary_email_address")
        or f"{clerk_user_id}@clerk.user"
    )

    # Look up user in our Turso DB
    conn = get_db()
    result = conn.execute(
        "SELECT id, clerk_user_id, tenant_id, email, role FROM users WHERE clerk_user_id = ?",
        [clerk_user_id],
    )

    if result.rows:
        row = result.rows[0]
        user = {
            "id": row[0],
            "clerk_user_id": row[1],
            "tenant_id": row[2],
            "email": row[3],
            "role": row[4],
        }
        logger.debug("Authenticated: user=%s tenant=%s", user["id"][:8], user["tenant_id"][:8])
        return user

    # User not in our DB — auto-provision
    user = _auto_provision_user(clerk_user_id, email, conn)
    return user


def require_role(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.delete("/resource")
        def delete_resource(user: dict = Depends(require_role(["admin"]))):
            ...
    """
    def role_checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {allowed_roles}. You have: {user['role']}",
            )
        return user
    return role_checker
