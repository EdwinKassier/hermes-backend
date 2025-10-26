# ==============================================================================
# Hermes Backend - Single Stage Production Dockerfile
# ==============================================================================
FROM python:3.12-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 hermes && \
    mkdir -p /app/log && \
    chown -R hermes:hermes /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=hermes:hermes . .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8080

# Set user
USER hermes

# Use gunicorn with config file and correct entry point
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]