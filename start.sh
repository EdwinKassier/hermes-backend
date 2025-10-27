#!/bin/bash
# Startup script for Cloud Run - starts Redis and Gunicorn

set -e

echo "🚀 Starting Hermes Backend..."

# Start Redis in the background
echo "📦 Starting local Redis server..."
redis-server --daemonize yes \
  --bind 127.0.0.1 \
  --port 6379 \
  --maxmemory 256mb \
  --maxmemory-policy allkeys-lru \
  --save "" \
  --appendonly no \
  --loglevel warning

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
timeout 10 bash -c 'until redis-cli ping 2>/dev/null; do sleep 0.5; done'

if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# Start Gunicorn
echo "🦄 Starting Gunicorn..."
exec gunicorn -c gunicorn.conf.py "app:create_app()"
