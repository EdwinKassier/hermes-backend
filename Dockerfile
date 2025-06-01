# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PYTHONOPTIMIZE=2

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
# Install torch and torchaudio with specific versions first
RUN pip install --no-cache-dir --retries 10 --timeout 180 torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
# Install blinker explicitly to satisfy Flask's dependency
RUN pip install --no-cache-dir blinker
# Then install the rest of the requirements
RUN pip install --no-cache-dir --retries 10 --timeout 180 -r requirements.txt

# Runtime stage
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PYTHONOPTIMIZE=2 \
    WEB_CONCURRENCY=4

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . .

# Pre-compile Python files
RUN python -m compileall -q .

# Create app data directory and set permissions
RUN mkdir -p /app/data/tts_cache && \
    chown -R appuser:appuser /app/data && \
    chmod -R 755 /app/data

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variable for TTS cache directory
ENV TTS_CACHE_DIR=/app/data/tts_cache

# Expose the port
EXPOSE 8080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/status || exit 1

# Use gunicorn with config file for production
CMD ["gunicorn", "--config", "gunicorn.conf.py", "run:app"]
