"""
Lucida – Universal Adaptive Lead Scoring API
─────────────────────────────────────────────
Production-ready entry point — Clerk auth + Turso database.

Run with:
  uvicorn main:app --reload          (development)
  gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker  (production)
"""

import os
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.database import init_db, check_db_connectivity
from app.services import model_storage
from app.api.scoring import init_models_cache, trained_models

# ── Logging Setup ────────────────────────────────────────────

settings = get_settings()


def validate_runtime_settings():
    """Fail fast on production misconfiguration."""
    if settings.is_production:
        if settings.AUTH_BYPASS_ENABLED:
            raise RuntimeError("AUTH_BYPASS_ENABLED must be false in production.")
        if not settings.CLERK_SECRET_KEY:
            raise RuntimeError("CLERK_SECRET_KEY must be configured in production.")
        if not settings.clerk_jwt_issuer:
            raise RuntimeError(
                "CLERK_JWT_ISSUER must be configured in production, or CLERK_PUBLISHABLE_KEY "
                "must be set to infer issuer automatically."
            )
        if settings.REQUIRE_TURSO_IN_PRODUCTION and (not settings.TURSO_DATABASE_URL or not settings.TURSO_AUTH_TOKEN):
            raise RuntimeError(
                "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be configured in production "
                "when REQUIRE_TURSO_IN_PRODUCTION=true."
            )
        if not any(origin.strip() and origin.strip() != "*" for origin in settings.cors_origins_list):
            raise RuntimeError("CORS_ORIGINS must include at least one explicit origin in production.")
        if not any(host.strip() and host.strip() != "*" for host in settings.trusted_hosts_list):
            raise RuntimeError("TRUSTED_HOSTS must include at least one explicit host in production.")


def setup_logging():
    """Configure structured logging based on environment."""
    log_level = logging.DEBUG if not settings.is_production else logging.INFO
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if not settings.is_production
        else '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    )

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger("lucida")


# ── Rate Limiter ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan (startup/shutdown) ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Runs on startup and shutdown."""
    # ── STARTUP ──
    validate_runtime_settings()
    logger.info("Starting Lucida v%s [%s]", settings.APP_VERSION, settings.ENVIRONMENT)

    # Create database tables
    try:
        init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        logger.warning("Starting in degraded mode — DB unavailable")

    # Reload saved models from disk
    all_models = model_storage.load_all_models()
    init_models_cache(all_models)
    total = sum(len(m) for m in all_models.values())
    logger.info("Loaded %d models from disk", total)

    yield

    # ── SHUTDOWN ──
    from app.database import close_db
    from app.services.job_queue import shutdown_job_queue
    
    shutdown_job_queue()
    close_db()
    logger.info("Lucida shutting down")


# ── App Factory ──────────────────────────────────────────────

app = FastAPI(
    title="Lucida – Universal Adaptive Lead Scorer",
    description=(
        "Zero-configuration lead scoring SaaS. "
        "Upload ANY CSV → Auto-train ML model → Score leads. "
        "Fully authenticated via Clerk, tenant-isolated, Turso-backed."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if (not settings.is_production or settings.ENABLE_API_DOCS) else None,
    redoc_url="/redoc" if (not settings.is_production or settings.ENABLE_API_DOCS) else None,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Response compression
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Block requests with unexpected Host headers
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts_list or ["localhost", "127.0.0.1"],
)

# CORS
cors_origins = settings.cors_origins_list
if settings.is_production:
    # Never allow wildcard in production
    cors_origins = [o for o in cors_origins if o != "*"]
else:
    # In development, also allow wildcard for easy testing
    if "*" not in cors_origins:
        cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Attach security-focused response headers."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if settings.is_production:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
    return response


# ── Request Logging Middleware ───────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status, and duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    # Don't log health checks or static assets
    path = request.url.path
    if path not in ("/health", "/favicon.ico"):
        logger.info(
            "%s %s -> %d (%sms)",
            request.method, path, response.status_code, duration_ms,
        )
    return response


# ── Include Routers ──────────────────────────────────────────

from app.api.auth import router as auth_router
from app.api.scoring import router as scoring_router
from app.api.models_api import router as models_router

app.include_router(auth_router)
app.include_router(scoring_router)
app.include_router(models_router)


# ── Health Check (no auth required) ─────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Health check for Railway / Docker / load balancers."""
    # Check database connectivity
    db_ok = check_db_connectivity()

    total_models = sum(len(m) for m in trained_models.values())

    status_val = "healthy" if db_ok else "degraded"

    return {
        "status": status_val,
        "database": "connected" if db_ok else "disconnected",
        "models_loaded": total_models,
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Web UI (temporary — will be replaced by React frontend) ─

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def web_ui():
    """Serve the existing HTML UI."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Lucida API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


# ── Global Exception Handler ────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return standardised error format."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
            },
        },
    )
