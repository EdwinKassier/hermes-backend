#!/bin/bash
set -e  # Exit on any error

# ==============================================================================
# Hermes Backend - Production Deployment Script (Cloud Run)
# ==============================================================================

# Navigate to project root (two levels up from scripts/deployment/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}"

echo "=================================================="
echo "  Hermes Backend Deployment to Google Cloud Run"
echo "=================================================="
echo "📂 Project root: ${PROJECT_ROOT}"
echo ""

# ==============================================================================
# Configuration Variables
# ==============================================================================

# GCP Configuration
PROJECT_ID="${GCP_PROJECT_ID:-edwin-portfolio-358212}"
SERVICE_NAME="${SERVICE_NAME:-master-hermes-backend}"
REGION="${REGION:-us-central1}"
ARTIFACT_REGISTRY_REPO="ashes"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}"

# Resource Configuration
CPU_COUNT="${CPU_COUNT:-2}"
MEMORY_LIMIT="${MEMORY_LIMIT:-4Gi}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
CONCURRENCY="${CONCURRENCY:-50}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300s}"

# Build Configuration
PLATFORM="linux/amd64"
BUILD_TAG="${BUILD_TAG:-$(date +%Y%m%d-%H%M%S)}"

# Deployment Configuration
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"  # Set to false for production
TRAFFIC_PERCENTAGE="${TRAFFIC_PERCENTAGE:-100}"  # Gradual rollout: start with 10, increase to 100

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

echo ""
echo "🔍 Running pre-flight checks..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ ERROR: gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi
echo "✅ gcloud CLI found"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker not found. Please install Docker."
    exit 1
fi
echo "✅ Docker found"

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo "⚠️  Docker buildx not found. Creating builder..."
    docker buildx create --use --name hermes-builder --driver docker-container
fi
echo "✅ Docker buildx ready"

# Verify GCP project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "⚠️  Switching to project: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
fi
echo "✅ GCP Project: $PROJECT_ID"

# Check for required environment variables (for Secret Manager)
if [ -z "$GOOGLE_API_KEY" ] && [ -z "$USE_SECRET_MANAGER" ]; then
    echo "⚠️  WARNING: GOOGLE_API_KEY not set. Ensure it's in Secret Manager."
fi

if [ -z "$API_KEY" ] && [ -z "$USE_SECRET_MANAGER" ]; then
    echo "⚠️  WARNING: API_KEY not set. Ensure it's in Secret Manager."
fi

if [ -z "$ATTENDEE_API_KEY" ] && [ -z "$USE_SECRET_MANAGER" ]; then
    echo "⚠️  WARNING: ATTENDEE_API_KEY not set. Ensure it's in Secret Manager."
fi

if [ -z "$SUPABASE_URL" ] && [ -z "$USE_SECRET_MANAGER" ]; then
    echo "⚠️  WARNING: SUPABASE_URL not set. Ensure it's in Secret Manager."
fi

if [ -z "$SUPABASE_SERVICE_ROLE_KEY" ] && [ -z "$USE_SECRET_MANAGER" ]; then
    echo "⚠️  WARNING: SUPABASE_SERVICE_ROLE_KEY not set. Ensure it's in Secret Manager."
fi

# Verify credentials file is NOT in the directory
if [ -f "credentials.json" ]; then
    echo "⚠️  WARNING: credentials.json found in directory."
    echo "    This should NOT be copied into the Docker image."
    echo "    Ensure it's in .dockerignore"
fi

echo ""
echo "=================================================="
echo "  Build Configuration"
echo "=================================================="
echo "Image: $IMAGE_NAME:$BUILD_TAG"
echo "Platform: $PLATFORM"
echo "Resources: ${CPU_COUNT} CPU, ${MEMORY_LIMIT} RAM"
echo "Instances: ${MIN_INSTANCES}-${MAX_INSTANCES}"
echo "Concurrency: ${CONCURRENCY}"
echo "Timeout: ${REQUEST_TIMEOUT}"
echo "=================================================="
echo ""

read -p "Continue with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# ==============================================================================
# Build Docker Image
# ==============================================================================

echo ""
echo "🔨 Building Docker image..."
echo "   Tag: $BUILD_TAG"
echo "   Platform: $PLATFORM"
echo ""

docker buildx build \
  --platform "${PLATFORM}" \
  -f Dockerfile \
  --build-arg INSTALL_ML=false \
  --no-cache \
  -t "${IMAGE_NAME}:${BUILD_TAG}" \
  -t "${IMAGE_NAME}:latest" \
  --push \
  --progress=plain \
  .

if [ $? -ne 0 ]; then
  echo "❌ Docker build or push failed. Exiting."
  exit 1
fi

echo "✅ Docker image built and pushed successfully"
echo "   ${IMAGE_NAME}:${BUILD_TAG}"

# ==============================================================================
# Deploy to Cloud Run
# ==============================================================================

echo ""
echo "🚀 Deploying to Cloud Run..."
echo ""

# Build the deployment command
DEPLOY_CMD="gcloud run deploy ${SERVICE_NAME}"
DEPLOY_CMD="${DEPLOY_CMD} --image ${IMAGE_NAME}:${BUILD_TAG}"
DEPLOY_CMD="${DEPLOY_CMD} --platform managed"
DEPLOY_CMD="${DEPLOY_CMD} --region ${REGION}"
DEPLOY_CMD="${DEPLOY_CMD} --execution-environment gen2"

# Resource configuration
DEPLOY_CMD="${DEPLOY_CMD} --cpu ${CPU_COUNT}"
DEPLOY_CMD="${DEPLOY_CMD} --memory ${MEMORY_LIMIT}"
DEPLOY_CMD="${DEPLOY_CMD} --no-cpu-throttling"
DEPLOY_CMD="${DEPLOY_CMD} --min-instances ${MIN_INSTANCES}"
DEPLOY_CMD="${DEPLOY_CMD} --max-instances ${MAX_INSTANCES}"
DEPLOY_CMD="${DEPLOY_CMD} --concurrency ${CONCURRENCY}"
DEPLOY_CMD="${DEPLOY_CMD} --timeout ${REQUEST_TIMEOUT}"

# Health checks & performance
DEPLOY_CMD="${DEPLOY_CMD} --cpu-boost"

# Authentication
if [ "$ALLOW_UNAUTH" = "true" ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --allow-unauthenticated"
else
    DEPLOY_CMD="${DEPLOY_CMD} --no-allow-unauthenticated"
fi

# Environment variables (non-sensitive)
# Note: PORT is automatically set by Cloud Run and should NOT be included
DEPLOY_CMD="${DEPLOY_CMD} --set-env-vars=PROJECT_ID=${PROJECT_ID}"
DEPLOY_CMD="${DEPLOY_CMD},SERVICE_NAME=${SERVICE_NAME}"
DEPLOY_CMD="${DEPLOY_CMD},REGION_DEPLOYED=${REGION}"
DEPLOY_CMD="${DEPLOY_CMD},APPLICATION_ENV=production"
DEPLOY_CMD="${DEPLOY_CMD},APP_NAME=hermes-backend"

# Secrets from Secret Manager (RECOMMENDED)
# Uncomment and configure these after setting up secrets in Secret Manager:
# 
# Core API Keys:
# DEPLOY_CMD="${DEPLOY_CMD} --set-secrets=API_KEY=api-key:latest"
# DEPLOY_CMD="${DEPLOY_CMD},GOOGLE_API_KEY=google-api-key:latest"
# DEPLOY_CMD="${DEPLOY_CMD},GOOGLE_PROJECT_ID=google-project-id:latest"
# DEPLOY_CMD="${DEPLOY_CMD},GOOGLE_PROJECT_LOCATION=google-project-location:latest"
#
# Prism Domain (Attendee API Integration):
# DEPLOY_CMD="${DEPLOY_CMD},ATTENDEE_API_KEY=attendee-api-key:latest"
# DEPLOY_CMD="${DEPLOY_CMD},PRISM_BASE_PROMPT=prism-base-prompt:latest"
#
# Supabase (Vector Store for Gemini RAG):
# DEPLOY_CMD="${DEPLOY_CMD},SUPABASE_URL=supabase-url:latest"
# DEPLOY_CMD="${DEPLOY_CMD},SUPABASE_DATABASE_URL=supabase-database-url:latest"
# DEPLOY_CMD="${DEPLOY_CMD},SUPABASE_SERVICE_ROLE_KEY=supabase-service-role-key:latest"
#
# Base Prompts for different personas:
# DEPLOY_CMD="${DEPLOY_CMD},BASE_PROMPT=base-prompt:latest"
# DEPLOY_CMD="${DEPLOY_CMD},PRISMA_BASE_PROMPT=prisma-base-prompt:latest"
#
# Webhook & WebSocket URLs (will be set to Cloud Run service URL after deployment):
# These should be set after initial deployment using:
# gcloud run services update ${SERVICE_NAME} --region=${REGION} \
#   --update-env-vars WEBHOOK_BASE_URL=https://YOUR-SERVICE-URL \
#   --update-env-vars WEBSOCKET_BASE_URL=wss://YOUR-SERVICE-URL

# Traffic management (for gradual rollout)
if [ "$TRAFFIC_PERCENTAGE" -ne 100 ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --no-traffic"
    echo "⚠️  Deploying without traffic (gradual rollout)"
fi

# Execute deployment
eval $DEPLOY_CMD

if [ $? -ne 0 ]; then
  echo "❌ Cloud Run deployment failed. Exiting."
  exit 1
fi

echo "✅ Cloud Run deployment successful"

# ==============================================================================
# Traffic Management
# ==============================================================================

echo ""
echo "🔀 Managing traffic routing..."

if [ "$TRAFFIC_PERCENTAGE" -eq 100 ]; then
    echo "   Routing 100% traffic to new revision"
    
    gcloud run services update-traffic "${SERVICE_NAME}" \
      --to-latest \
      --region="${REGION}"
    
    echo "✅ All traffic routed to new revision"
else
    echo "   Routing ${TRAFFIC_PERCENTAGE}% traffic to new revision (gradual rollout)"
    
    gcloud run services update-traffic "${SERVICE_NAME}" \
      --to-revisions=LATEST=${TRAFFIC_PERCENTAGE} \
      --region="${REGION}"
    
    echo "✅ Traffic split configured"
    echo "   To route 100% traffic: gcloud run services update-traffic ${SERVICE_NAME} --to-latest --region=${REGION}"
fi

# ==============================================================================
# Post-Deployment Verification
# ==============================================================================

echo ""
echo "🔍 Verifying deployment..."

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --format 'value(status.url)' 2>/dev/null)

# Get active revision and traffic percentage
ACTIVE_REVISION=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --format 'value(status.traffic[0].revisionName)' 2>/dev/null)

TRAFFIC_PERCENT=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --format 'value(status.traffic[0].percent)' 2>/dev/null)

if [ -z "$SERVICE_URL" ]; then
    echo "⚠️  Could not retrieve service URL"
else
    echo "✅ Service URL: $SERVICE_URL"
    
    if [ -n "$ACTIVE_REVISION" ]; then
        echo "✅ Active Revision: $ACTIVE_REVISION (${TRAFFIC_PERCENT}% traffic)"
    fi
    
    # Test health endpoint
    echo ""
    echo "Testing /health endpoint..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" || echo "000")
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Health check passed (HTTP $HTTP_CODE)"
    else
        echo "⚠️  Health check returned HTTP $HTTP_CODE"
        echo "   This may be normal if the app is still starting up."
    fi
    
    # Configure Prism webhook and WebSocket URLs
    echo ""
    echo "🔧 Configuring Prism webhook and WebSocket URLs..."
    WEBHOOK_URL="${SERVICE_URL}"
    WEBSOCKET_URL=$(echo "$SERVICE_URL" | sed 's/https:/wss:/')
    
    gcloud run services update "${SERVICE_NAME}" \
      --region="${REGION}" \
      --update-env-vars="WEBHOOK_BASE_URL=${WEBHOOK_URL},WEBSOCKET_BASE_URL=${WEBSOCKET_URL}" \
      --quiet
    
    if [ $? -eq 0 ]; then
        echo "✅ Prism URLs configured:"
        echo "   WEBHOOK_BASE_URL: ${WEBHOOK_URL}"
        echo "   WEBSOCKET_BASE_URL: ${WEBSOCKET_URL}"
    else
        echo "⚠️  Failed to configure Prism URLs. You can set them manually with:"
        echo "   gcloud run services update ${SERVICE_NAME} --region=${REGION} \\"
        echo "     --update-env-vars WEBHOOK_BASE_URL=${WEBHOOK_URL},WEBSOCKET_BASE_URL=${WEBSOCKET_URL}"
    fi
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "=================================================="
echo "  Deployment Summary"
echo "=================================================="
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}:${BUILD_TAG}"
echo "URL: ${SERVICE_URL}"
if [ -n "$ACTIVE_REVISION" ]; then
    echo "Active Revision: ${ACTIVE_REVISION} (${TRAFFIC_PERCENT}% traffic)"
fi
echo "Resources: ${CPU_COUNT} CPU, ${MEMORY_LIMIT} RAM"
echo "Scaling: ${MIN_INSTANCES}-${MAX_INSTANCES} instances"
echo "=================================================="
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Test the API: curl ${SERVICE_URL}/health"
echo "   2. Test Hermes: curl -X POST ${SERVICE_URL}/api/v1/hermes/chat \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"message\": \"Hello\", \"user_id\": \"test\"}'"
echo "   3. Test Prism: Use WebSocket client to connect to ${SERVICE_URL}/api/v1/prism/start-session"
echo "   4. Monitor logs: gcloud run logs tail ${SERVICE_NAME} --region=${REGION}"
echo "   5. View metrics: gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
if [ "$TRAFFIC_PERCENTAGE" -ne 100 ]; then
    echo "   6. Route 100% traffic: gcloud run services update-traffic ${SERVICE_NAME} --to-latest --region=${REGION}"
fi
echo ""
echo "⚠️  IMPORTANT: Configure Secret Manager secrets before production use:"
echo "   See commented section in this script for all required secrets"
echo ""

