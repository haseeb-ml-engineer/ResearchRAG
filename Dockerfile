# ============================================================================
# ResearchRAG — Production Dockerfile
# ============================================================================
# Multi-stage build using Python 3.12 slim.
# Stage 1: Install dependencies into a virtual environment.
# Stage 2: Copy the venv and application code into a minimal runtime image.
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System packages required to compile native extensions (PyTorch, ChromaDB)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment so the runtime stage can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (maximises Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Security: run as a non-root user
RUN groupadd --system appgroup && \
    useradd  --system --gid appgroup --create-home appuser

WORKDIR /app

# Carry over the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Application-level defaults (overridable at runtime via docker run -e)
ENV APP_ENVIRONMENT=production \
    APP_DEBUG=false \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_WORKERS=2

# Copy only application code (dev tooling, notebooks, tests excluded)
COPY src/        ./src/
COPY scripts/    ./scripts/
COPY frontend/   ./frontend/
COPY docs/       ./docs/
COPY requirements.txt pyproject.toml config.yaml ./
COPY .env.example ./.env.example

# Create writable directories for runtime data and logs
RUN mkdir -p /app/data/uploads /app/data/vectorstore /app/logs && \
    chown -R appuser:appgroup /app

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck using the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request, sys; urllib.request.urlopen('http://localhost:8000/health'); sys.exit(0)"]

# Drop privileges
USER appuser

# Launch the production ASGI server
CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]
