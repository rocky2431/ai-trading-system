# ==================== Production Stage ====================
FROM python:3.12-slim as production

WORKDIR /app

# Install build dependencies (needed for some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 iqfmp && \
    useradd --uid 1000 --gid iqfmp --shell /bin/bash --create-home iqfmp

# Copy source code first
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY vendor/qlib/ ./vendor/qlib/

# Install all dependencies including the package itself
RUN pip install --no-cache-dir -e ".[qlib]" && \
    pip install --no-cache-dir gym cvxpy
COPY start.sh /app/start.sh

# Set ownership and permissions
RUN chown -R iqfmp:iqfmp /app && chmod +x /app/start.sh

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

# Expose port (Railway will override PORT)
EXPOSE 8000

# Default command - uses shell to properly expand PORT
CMD ["/bin/sh", "/app/start.sh"]
