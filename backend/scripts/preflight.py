"""
Deployment preflight checks for Lucida backend.

This script intentionally avoids importing the ML stack so it can run even on
older local machines that cannot import the pinned NumPy wheels.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings


def main() -> int:
    settings = get_settings()
    errors = []
    warnings = []

    if settings.is_production:
        if settings.AUTH_BYPASS_ENABLED:
            errors.append("AUTH_BYPASS_ENABLED must be false in production.")
        if not settings.CLERK_SECRET_KEY:
            errors.append("CLERK_SECRET_KEY must be set in production.")
        elif settings.CLERK_SECRET_KEY.startswith("sk_test_"):
            warnings.append("CLERK_SECRET_KEY uses a test key in production mode.")
        if not settings.clerk_jwt_issuer:
            errors.append(
                "CLERK_JWT_ISSUER must be set in production, or CLERK_PUBLISHABLE_KEY must be set so issuer can be inferred."
            )
        explicit_origins = [origin.strip() for origin in settings.cors_origins_list if origin.strip() and origin.strip() != "*"]
        if not explicit_origins:
            errors.append("CORS_ORIGINS must include at least one explicit origin in production.")
        explicit_hosts = [host.strip() for host in settings.trusted_hosts_list if host.strip() and host.strip() != "*"]
        if not explicit_hosts:
            errors.append("TRUSTED_HOSTS must include at least one explicit host in production.")
        if settings.REQUIRE_TURSO_IN_PRODUCTION and (not settings.TURSO_DATABASE_URL or not settings.TURSO_AUTH_TOKEN):
            errors.append(
                "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in production when REQUIRE_TURSO_IN_PRODUCTION=true."
            )
        if settings.ENABLE_API_DOCS:
            warnings.append("ENABLE_API_DOCS=true in production mode. Consider disabling API docs.")
    else:
        if not settings.CLERK_SECRET_KEY:
            warnings.append("CLERK_SECRET_KEY is unset; local dev auth bypass will be used.")

    if not settings.MODEL_ARTIFACTS_DIR:
        errors.append("MODEL_ARTIFACTS_DIR must not be empty.")

    if errors:
        print("Lucida preflight: FAILED")
        for item in errors:
            print(f"ERROR: {item}")
        for item in warnings:
            print(f"WARNING: {item}")
        return 1

    print("Lucida preflight: OK")
    for item in warnings:
        print(f"WARNING: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
