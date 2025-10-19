#!/bin/bash
# ==============================================================================
# Docker Build Verification Script
# ==============================================================================
# This script builds the Docker image locally and verifies gunicorn is installed
# and can be executed. Run this before deploying to Cloud Run.
# ==============================================================================

set -e

echo "=================================================="
echo "  Docker Build Verification"
echo "=================================================="
echo ""

# Build the image
echo "🔨 Building Docker image locally..."
docker build \
  --build-arg INSTALL_ML=false \
  -t hermes-backend:verify \
  -f Dockerfile \
  .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker build succeeded"
echo ""

# Verify gunicorn is installed
echo "🔍 Verifying gunicorn installation..."
GUNICORN_PATH=$(docker run --rm hermes-backend:verify which gunicorn 2>/dev/null || echo "")

if [ -z "$GUNICORN_PATH" ]; then
    echo "❌ ERROR: gunicorn not found in image!"
    exit 1
fi

echo "✅ gunicorn found at: $GUNICORN_PATH"
echo ""

# Verify gunicorn version
echo "🔍 Checking gunicorn version..."
GUNICORN_VERSION=$(docker run --rm hermes-backend:verify gunicorn --version 2>/dev/null || echo "")

if [ -z "$GUNICORN_VERSION" ]; then
    echo "❌ ERROR: gunicorn cannot be executed!"
    exit 1
fi

echo "✅ $GUNICORN_VERSION"
echo ""

# Verify gevent is installed
echo "🔍 Verifying gevent installation..."
GEVENT_CHECK=$(docker run --rm hermes-backend:verify python -c "import gevent; print(f'gevent {gevent.__version__}')" 2>/dev/null || echo "")

if [ -z "$GEVENT_CHECK" ]; then
    echo "❌ ERROR: gevent not found!"
    exit 1
fi

echo "✅ $GEVENT_CHECK"
echo ""

# Test that the application can start (quick check)
echo "🔍 Testing application startup (5 second check)..."
CONTAINER_ID=$(docker run -d --rm -e PORT=8080 hermes-backend:verify)

sleep 5

if docker ps | grep -q "$CONTAINER_ID"; then
    echo "✅ Application started successfully"
    docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
else
    echo "❌ ERROR: Application failed to start"
    docker logs "$CONTAINER_ID" 2>&1 | tail -20
    exit 1
fi

echo ""
echo "=================================================="
echo "  ✅ All Verifications Passed!"
echo "=================================================="
echo ""
echo "Your Docker image is ready for deployment."
echo ""
echo "📋 Next steps:"
echo "   1. Deploy to Cloud Run: ./scripts/deployment/deploy-fixed.sh"
echo "   2. Monitor logs: gcloud run logs tail master-hermes-backend --region=us-central1"
echo ""


