"""
Auth API routes — Clerk-compatible.
No register/login endpoints — Clerk handles all authentication.
Only provides /auth/me for getting current user info.
"""

import logging
from fastapi import APIRouter, Depends

from app.database import get_db
from app.core.auth import get_current_user
from app.core.responses import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """
    Return the currently authenticated user's profile.
    Clerk handles login/register — this just returns our DB record.
    """
    conn = get_db()

    # Get tenant name
    tenant_name = None
    result = conn.execute(
        "SELECT name, plan FROM tenants WHERE id = ?",
        [user["tenant_id"]],
    )
    if result.rows:
        row = result.rows[0]
        tenant_name = row[0]
        plan = row[1]
    else:
        plan = "free"

    return success_response(
        data={
            "id": user["id"],
            "clerk_user_id": user["clerk_user_id"],
            "email": user["email"],
            "role": user["role"],
            "tenant_id": user["tenant_id"],
            "company_name": tenant_name,
            "plan": plan,
        }
    )
