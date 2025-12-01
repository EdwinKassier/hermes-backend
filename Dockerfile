# ==============================================================================
# Hermes Backend - Single Stage Production Dockerfile
# ==============================================================================
FROM python:3.12-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080

# Install system dependencies (optimized for TTS operations)
# Fix GPG signature issues by refreshing package lists
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
        redis-server \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 hermes && \
    mkdir -p /app/log && \
    chown -R hermes:hermes /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=hermes:hermes . .

# Make startup script executable
RUN chmod +x /app/start.sh

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8080

# Set user
USER hermes

# Use startup script to run Redis and Gunicorn
CMD ["/app/start.sh"]
