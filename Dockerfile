# ==================== Builder Stage ====================
# NOTE: Project requires Python >=3.12 (see pyproject.toml)
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY pyproject.toml ./
COPY src/ ./src/
COPY vendor/qlib/ ./vendor/qlib/

# Install Python dependencies including Qlib extras
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -e ".[qlib]"

# ==================== Production Stage ====================
FROM python:3.12-slim as production

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 iqfmp && \
    useradd --uid 1000 --gid iqfmp --shell /bin/bash --create-home iqfmp

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/src /app/src
COPY --from=builder /app/vendor/qlib /app/vendor/qlib
COPY --from=builder /app/pyproject.toml /app/

# Install application with Qlib dependencies
RUN pip install --no-cache-dir /wheels/*.whl && \
    pip install --no-cache-dir gym cvxpy && \
    rm -rf /wheels

# Copy application code
COPY src/ /app/src/
COPY vendor/qlib/ /app/vendor/qlib/

# Set ownership
RUN chown -R iqfmp:iqfmp /app

# Switch to non-root user
USER iqfmp

# Create Qlib data directory
RUN mkdir -p /home/iqfmp/.qlib/qlib_data

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app/vendor/qlib \
    QLIB_AUTO_INIT=true \
    QLIB_DATA_DIR=/home/iqfmp/.qlib/qlib_data \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Default command
CMD ["python", "-m", "uvicorn", "iqfmp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
