# =============================================================================
# Timonel RAG — Multi-stage Dockerfile
#
# Stage 1 (builder): install Python runtime dependencies into a venv.
# Stage 2 (runtime): copy only the venv + source; run as a non-root user.
#
# Build:  docker build -t timonel-rag .
# Run:    docker compose up
# =============================================================================

# ---- Stage 1: dependency installation ----------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Create an isolated virtual environment so the runtime stage only needs to
# copy /venv — nothing else from the builder leaks into the final image.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy requirements first to maximise layer-cache reuse: this layer is only
# rebuilt when requirements.txt changes, not on every source-code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime image --------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user — principle of least privilege.
RUN groupadd --system timonel \
    && useradd --system --gid timonel --no-create-home timonel

WORKDIR /app

# Bring in the pre-built venv from the builder stage (no pip in final image).
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy only runtime artefacts — tests, .git, .env, and data dirs are excluded
# via .dockerignore so they never reach the build context.
COPY src/ ./src/
COPY app.py cli.py pyproject.toml ./

# Create mount-point directories and hand ownership to the runtime user.
# These paths are mounted as Docker volumes in docker-compose.yml so that
# PDFs, ChromaDB, and the SQLite audit log persist across container restarts.
RUN mkdir -p /app/data /app/chroma_db \
    && chown -R timonel:timonel /app

USER timonel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Single uvicorn worker is appropriate for an async RAG API.
# Scale horizontally with docker compose --scale or a load balancer if needed.
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
