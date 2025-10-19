# ==============================================================================
# Hermes Backend - Unified Dockerfile
# ==============================================================================
# Build for Production: docker build --build-arg INSTALL_ML=false .
# Build for Development: docker build --build-arg INSTALL_ML=true .
# ==============================================================================

# ==============================================================================
# Build Stage - Compile dependencies
# ==============================================================================
FROM python:3.11-slim as builder

# Build arguments
ARG INSTALL_ML=false
ARG DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && if [ "$INSTALL_ML" = "true" ]; then \
        apt-get install -y --no-install-recommends libsndfile1 ffmpeg; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies
# Conditionally filter out ML and dev packages for production
RUN python -m pip install --upgrade pip && \
    pip install --user --no-cache-dir "numpy>=1.24.0,<2.0.0" && \
    if [ "$INSTALL_ML" = "true" ]; then \
        echo "Installing ALL dependencies (including ML/dev)..."; \
        pip install --user --no-cache-dir -r requirements.txt; \
    else \
        echo "Installing production dependencies (excluding ML/dev)..."; \
        grep -v "^torch" requirements.txt | \
        grep -v "^torchaudio" | \
        grep -v "^chatterbox-tts" | \
        grep -v "^transformers" | \
        grep -v "^pytest" | \
        grep -v "^black" | \
        grep -v "^flake8" | \
        grep -v "^mypy" | \
        grep -v "^pylint" | \
        grep -v "^ipython" | \
        grep -v "^ipdb" | \
        grep -v "^flask-testing" | \
        grep -v "^iniconfig" | \
        grep -v "^pluggy" | \
        pip install --user --no-cache-dir -r /dev/stdin; \
    fi

# ==============================================================================
# Runtime Stage - Minimal production image
# ==============================================================================
FROM python:3.11-slim

# Build arguments (need to redeclare in new stage)
ARG INSTALL_ML=false
ARG DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/home/hermes/.local/bin:$PATH

# Install only runtime dependencies
# Note: ffmpeg is always installed (required for Prism audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 hermes && \
    mkdir -p /app/log && \
    chown -R hermes:hermes /app

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder to hermes user's home
COPY --from=builder --chown=hermes:hermes /root/.local /home/hermes/.local

# Copy application code (excluding files in .dockerignore)
COPY --chown=hermes:hermes . .

# Ensure proper permissions
RUN chmod +x /app/*.py 2>/dev/null || true && \
    chmod -R 755 /app/log

# Switch to non-root user
USER hermes

# Expose port (Cloud Run will override this via PORT env var)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Run the application with Gunicorn (using config file for gevent worker)
# Use full path to ensure gunicorn is found
CMD ["/home/hermes/.local/bin/gunicorn", "--config", "gunicorn.conf.py", "run:app"]
