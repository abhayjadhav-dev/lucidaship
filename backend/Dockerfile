# This image is the intended runtime for modern cloud CPUs.
# Older local x86 hosts may fail to import the ML wheel stack.

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
ENV PIP_ONLY_BINARY=:all:
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

# Security: run as non-root user
RUN groupadd -r lucida && useradd -r -g lucida lucida

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create model artifacts directory
RUN mkdir -p /app/model_artifacts && chown -R lucida:lucida /app

USER lucida

# Environment
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start with Gunicorn + Uvicorn workers
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
