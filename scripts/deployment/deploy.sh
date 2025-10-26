#!/bin/bash
set -e

# ==============================================================================
# Hermes Backend - Unified Deployment Script
# ==============================================================================
# Deploys Hermes backend to Google Cloud Run with smart defaults and options
#
# Usage:
#   ./scripts/deployment/deploy.sh [OPTIONS]
#
# Options:
#   --quick          Fast deployment with cache (default)
#   --clean          Clean build without cache
#   --check          Run diagnostics before deploying
#   --verify         Build and verify locally before deploying
#   --no-deploy      Build and push only, skip Cloud Run deployment
#   --help           Show this help message
#
# Environment Variables:
#   GCP_PROJECT_ID        GCP project (default: edwin-portfolio-358212)
#   SERVICE_NAME          Cloud Run service (default: master-hermes-backend)
#   REGION                GCP region (default: us-central1)
#   CPU_COUNT             Number of CPUs (default: 2)
#   MEMORY_LIMIT          Memory limit (default: 4Gi)
#   MAX_INSTANCES         Max instances (default: 10)
#   TRAFFIC_PERCENTAGE    Traffic to new revision (default: 100)
#
# Examples:
#   ./scripts/deployment/deploy.sh                    # Quick deploy
#   ./scripts/deployment/deploy.sh --clean            # Clean build
#   ./scripts/deployment/deploy.sh --check --verify   # Full checks
#   CPU_COUNT=4 MEMORY_LIMIT=8Gi ./scripts/deployment/deploy.sh
# ==============================================================================

# ==============================================================================
# Configuration
# ==============================================================================

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}"

# Parse command line arguments
USE_CACHE=true
RUN_DIAGNOSTICS=false
RUN_VERIFY=false
SKIP_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            USE_CACHE=true
            shift
            ;;
        --clean)
            USE_CACHE=false
            shift
            ;;
        --check)
            RUN_DIAGNOSTICS=true
            shift
            ;;
        --verify)
            RUN_VERIFY=true
            shift
            ;;
        --no-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --help|-h)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# GCP Configuration
PROJECT_ID="${GCP_PROJECT_ID:-edwin-portfolio-358212}"
SERVICE_NAME="${SERVICE_NAME:-master-hermes-backend}"
REGION="${REGION:-us-central1}"
ARTIFACT_REGISTRY_REPO="ashes"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}"

# Resource Configuration
# Increased memory to account for local Redis + 8 Gunicorn workers
# Memory breakdown: Redis (256MB) + Workers (2.5GB) + Buffer (1.5GB) = ~4.25GB minimum
# Allocated 6Gi for safety margin with AI operations, WebSockets, and audio processing
CPU_COUNT="${CPU_COUNT:-2}"
MEMORY_LIMIT="${MEMORY_LIMIT:-6Gi}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"  # Keep 1 instance warm to avoid Redis startup latency
MAX_INSTANCES="${MAX_INSTANCES:-10}"
CONCURRENCY="${CONCURRENCY:-50}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300s}"

# Build Configuration
PLATFORM="linux/amd64"
BUILD_TAG="${BUILD_TAG:-$(date +%Y%m%d-%H%M%S)}"

# Deployment Configuration
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"
TRAFFIC_PERCENTAGE="${TRAFFIC_PERCENTAGE:-100}"

# ==============================================================================
# Helper Functions
# ==============================================================================

check_requirements() {
    echo "🔍 Checking requirements..."
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        echo "❌ ERROR: gcloud CLI not found. Install Google Cloud SDK."
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        echo "❌ ERROR: Docker not found. Install Docker."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info > /dev/null 2>&1; then
        echo "❌ ERROR: Docker daemon not running. Start Docker."
        exit 1
    fi
    
    # Check buildx
    if ! docker buildx version &> /dev/null; then
        echo "⚠️  Creating Docker buildx builder..."
        docker buildx create --use --name hermes-builder --driver docker-container
    fi
    
    echo "✅ All requirements met"
}

run_diagnostics() {
    echo ""
    echo "=================================================="
    echo "  Running Diagnostics"
    echo "=================================================="
    
    # Check .dockerignore
    if [ -f ".dockerignore" ]; then
        echo "✅ .dockerignore exists ($(wc -l < .dockerignore) lines)"
    else
        echo "⚠️  WARNING: .dockerignore missing - large build context!"
    fi
    
    # Check Docker disk usage
    echo ""
    echo "📊 Docker disk usage:"
    docker system df
    
    # Warn if lots of reclaimable space
    RECLAIMABLE=$(docker system df | grep "Images" | awk '{print $4}' | tr -d '()%')
    if [ -n "$RECLAIMABLE" ] && [ "$RECLAIMABLE" -gt 50 ]; then
        echo ""
        echo "⚠️  WARNING: ${RECLAIMABLE}% of Docker images are unused"
        echo "   Consider running: docker system prune -a"
    fi
    
    # Check authentication
    echo ""
    echo "🔐 Authentication:"
    CURRENT_ACCOUNT=$(gcloud config get-value account 2>/dev/null)
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    
    if [ -n "$CURRENT_ACCOUNT" ]; then
        echo "✅ Logged in as: $CURRENT_ACCOUNT"
        echo "✅ Project: $CURRENT_PROJECT"
        
        if grep -q "${REGION}-docker.pkg.dev" ~/.docker/config.json 2>/dev/null; then
            echo "✅ Docker authenticated with Artifact Registry"
        else
            echo "⚠️  Authenticating Docker with Artifact Registry..."
            gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
        fi
    else
        echo "❌ ERROR: Not logged in to gcloud"
        exit 1
    fi
    
    echo ""
}

run_verification() {
    echo ""
    echo "=================================================="
    echo "  Verifying Local Build"
    echo "=================================================="
    
    echo "🔨 Building test image..."
    docker build \
        -t hermes-backend:verify \
        -f Dockerfile \
        . > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Test build failed"
        exit 1
    fi
    echo "✅ Test build succeeded"
    
    echo "🔍 Verifying gunicorn..."
    if docker run --rm hermes-backend:verify which gunicorn > /dev/null 2>&1; then
        echo "✅ gunicorn installed"
    else
        echo "❌ ERROR: gunicorn not found"
        exit 1
    fi
    
    echo "🔍 Verifying application startup..."
    CONTAINER_ID=$(docker run -d --rm -e PORT=8080 hermes-backend:verify)
    sleep 3
    
    if docker ps | grep -q "$CONTAINER_ID"; then
        echo "✅ Application starts successfully"
        docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
    else
        echo "❌ ERROR: Application failed to start"
        docker logs "$CONTAINER_ID" 2>&1 | tail -10
        exit 1
    fi
    
    echo "🔍 Verifying agent prompts..."
    if docker run --rm hermes-backend:verify ls /app/docs/AgentPrompts/HERMES.md > /dev/null 2>&1; then
        echo "✅ HERMES agent prompt included"
    else
        echo "❌ ERROR: HERMES agent prompt missing"
        echo "   Check that docs/AgentPrompts/ is not excluded in .dockerignore"
        exit 1
    fi
    
    if docker run --rm hermes-backend:verify ls /app/docs/AgentPrompts/PRISM.md > /dev/null 2>&1; then
        echo "✅ PRISM agent prompt included"
    else
        echo "⚠️  WARNING: PRISM agent prompt missing"
    fi
    
    echo "🔍 Verifying tools load..."
    TOOL_OUTPUT=$(docker run --rm \
        -e GOOGLE_API_KEY=test \
        -e GOOGLE_PROJECT_ID=test \
        -e GOOGLE_PROJECT_LOCATION=us-east1 \
        hermes-backend:verify \
        python3 -c "from app.shared.utils.toolhub import get_all_tools; print(len(get_all_tools()))" 2>&1 | tail -1)
    
    if echo "$TOOL_OUTPUT" | grep -qE "^[1-9][0-9]*$"; then
        echo "✅ $TOOL_OUTPUT tool(s) loaded"
    else
        echo "❌ ERROR: Tools failed to load"
        echo "$TOOL_OUTPUT"
        exit 1
    fi
    
    echo ""
}

# ==============================================================================
# Main Script
# ==============================================================================

echo "=================================================="
echo "  Hermes Backend Deployment"
echo "=================================================="
echo "📂 Project: ${PROJECT_ROOT}"
echo "🎯 Mode: $([ "$USE_CACHE" = "true" ] && echo "Quick (with cache)" || echo "Clean (no cache)")"
echo ""

# Run optional checks
check_requirements

if [ "$RUN_DIAGNOSTICS" = "true" ]; then
    run_diagnostics
fi

if [ "$RUN_VERIFY" = "true" ]; then
    run_verification
fi

# Authenticate with Artifact Registry
if [ "$RUN_DIAGNOSTICS" = "false" ]; then
    echo "🔐 Authenticating with Artifact Registry..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null
fi

# Verify/switch GCP project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "⚠️  Switching to project: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
fi

echo ""
echo "=================================================="
echo "  Build Configuration"
echo "=================================================="
echo "Image:      $IMAGE_NAME:$BUILD_TAG"
echo "Platform:   $PLATFORM"
echo "Cache:      $([ "$USE_CACHE" = "true" ] && echo "Enabled" || echo "Disabled")"
echo "Resources:  ${CPU_COUNT} CPU, ${MEMORY_LIMIT} RAM"
echo "Scaling:    ${MIN_INSTANCES}-${MAX_INSTANCES} instances"
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
echo ""

# Build command
BUILD_CMD="docker buildx build"
BUILD_CMD="${BUILD_CMD} --platform ${PLATFORM}"
BUILD_CMD="${BUILD_CMD} -f Dockerfile"
BUILD_CMD="${BUILD_CMD} --progress=plain"

if [ "$USE_CACHE" = "false" ]; then
    BUILD_CMD="${BUILD_CMD} --no-cache"
fi

BUILD_CMD="${BUILD_CMD} -t ${IMAGE_NAME}:${BUILD_TAG}"
BUILD_CMD="${BUILD_CMD} -t ${IMAGE_NAME}:latest"
BUILD_CMD="${BUILD_CMD} --load"
BUILD_CMD="${BUILD_CMD} ."

eval $BUILD_CMD

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# ==============================================================================
# Push to Artifact Registry
# ==============================================================================

echo ""
echo "📤 Pushing to Artifact Registry..."
echo "   (This may take a few minutes...)"
echo ""

docker push "${IMAGE_NAME}:${BUILD_TAG}"
if [ $? -ne 0 ]; then
    echo "❌ Failed to push tagged image"
    exit 1
fi

docker push "${IMAGE_NAME}:latest"
if [ $? -ne 0 ]; then
    echo "❌ Failed to push latest image"
    exit 1
fi

echo "✅ Images pushed successfully"

# Skip deployment if requested
if [ "$SKIP_DEPLOY" = "true" ]; then
    echo ""
    echo "✅ Build and push complete (deployment skipped)"
    echo "   Image: ${IMAGE_NAME}:${BUILD_TAG}"
    exit 0
fi

# ==============================================================================
# Deploy to Cloud Run
# ==============================================================================

echo ""
echo "🚀 Deploying to Cloud Run..."
echo ""

DEPLOY_CMD="gcloud run deploy ${SERVICE_NAME}"
DEPLOY_CMD="${DEPLOY_CMD} --image ${IMAGE_NAME}:${BUILD_TAG}"
DEPLOY_CMD="${DEPLOY_CMD} --platform managed"
DEPLOY_CMD="${DEPLOY_CMD} --region ${REGION}"
DEPLOY_CMD="${DEPLOY_CMD} --execution-environment gen2"
DEPLOY_CMD="${DEPLOY_CMD} --cpu ${CPU_COUNT}"
DEPLOY_CMD="${DEPLOY_CMD} --memory ${MEMORY_LIMIT}"
DEPLOY_CMD="${DEPLOY_CMD} --no-cpu-throttling"
DEPLOY_CMD="${DEPLOY_CMD} --min-instances ${MIN_INSTANCES}"
DEPLOY_CMD="${DEPLOY_CMD} --max-instances ${MAX_INSTANCES}"
DEPLOY_CMD="${DEPLOY_CMD} --concurrency ${CONCURRENCY}"
DEPLOY_CMD="${DEPLOY_CMD} --timeout ${REQUEST_TIMEOUT}"
DEPLOY_CMD="${DEPLOY_CMD} --cpu-boost"

if [ "$ALLOW_UNAUTH" = "true" ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --allow-unauthenticated"
else
    DEPLOY_CMD="${DEPLOY_CMD} --no-allow-unauthenticated"
fi

# Create temporary YAML file for environment variables
# This approach handles complex values (spaces, special chars) much better than --set-env-vars
ENV_FILE=$(mktemp).yaml

# Start with base environment variables
cat > "$ENV_FILE" << EOF
PROJECT_ID: "${PROJECT_ID}"
SERVICE_NAME: "${SERVICE_NAME}"
REGION_DEPLOYED: "${REGION}"
APPLICATION_ENV: "production"
APP_NAME: "hermes-backend"
EOF

# Load additional environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env file..."
    
    # Read .env and add vars to YAML file
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Skip keys that shouldn't be in Cloud Run
        case "$key" in
            GOOGLE_APPLICATION_CREDENTIALS|TTS_DEVICE|REDIS_URL|PORT)
                # Skip local-only or Cloud Run-managed variables
                continue
                ;;
        esac
        
        # Remove quotes from value
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        
        # Add to environment variables YAML file
        # Use literal block style for prompts (multi-line text), quoted strings for everything else
        if [[ "$key" =~ _PROMPT$ ]] || [[ "$key" == "BASE_PROMPT" ]]; then
            # Use literal block style for long prompts
            echo "${key}: |-" >> "$ENV_FILE"
            echo "  ${value}" >> "$ENV_FILE"
        else
            # Use quoted string for simple values
            escaped_value=$(printf '%s' "$value" | sed 's/"/\\"/g')
            echo "${key}: \"${escaped_value}\"" >> "$ENV_FILE"
        fi
        
        echo "   ✓ ${key}"
    done < ".env"
    
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found - using default environment variables only"
fi

echo ""
echo "📄 Environment variables YAML file created"

# Show YAML file for debugging if SHOW_ENV_FILE is set
if [ "${SHOW_ENV_FILE}" = "true" ]; then
    echo "📋 Environment variables file contents:"
    cat "$ENV_FILE"
    echo ""
fi

DEPLOY_CMD="${DEPLOY_CMD} --env-vars-file=${ENV_FILE}"

if [ "$TRAFFIC_PERCENTAGE" -ne 100 ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --no-traffic"
fi

# Execute deployment
eval $DEPLOY_CMD
DEPLOY_EXIT_CODE=$?

# Clean up temp file
rm -f "$ENV_FILE"

if [ $DEPLOY_EXIT_CODE -ne 0 ]; then
    echo "❌ Cloud Run deployment failed"
    echo "💡 Tip: Run with SHOW_ENV_FILE=true to debug environment variables"
    exit 1
fi

echo "✅ Cloud Run deployment successful"

# ==============================================================================
# Configure Traffic
# ==============================================================================

echo ""
echo "🔀 Managing traffic..."

if [ "$TRAFFIC_PERCENTAGE" -eq 100 ]; then
    gcloud run services update-traffic "${SERVICE_NAME}" \
        --to-latest \
        --region="${REGION}" \
        --quiet
    echo "✅ Routing 100% traffic to new revision"
else
    gcloud run services update-traffic "${SERVICE_NAME}" \
        --to-revisions=LATEST=${TRAFFIC_PERCENTAGE} \
        --region="${REGION}" \
        --quiet
    echo "✅ Routing ${TRAFFIC_PERCENTAGE}% traffic to new revision"
fi

# ==============================================================================
# Post-Deployment
# ==============================================================================

echo ""
echo "🔍 Verifying deployment..."

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --format 'value(status.url)' 2>/dev/null)

if [ -z "$SERVICE_URL" ]; then
    echo "⚠️  Could not retrieve service URL"
else
    echo "✅ Service URL: $SERVICE_URL"
    
    # Update webhook/websocket URLs ONLY if not already set in .env
    WEBHOOK_URL="${SERVICE_URL}"
    WEBSOCKET_URL=$(echo "$SERVICE_URL" | sed 's/https:/wss:/')
    
    # Check if these were already set via .env
    WEBHOOK_SET=$(grep -E "^WEBHOOK_BASE_URL=" .env 2>/dev/null || echo "")
    WEBSOCKET_SET=$(grep -E "^WEBSOCKET_BASE_URL=" .env 2>/dev/null || echo "")
    
    if [ -z "$WEBHOOK_SET" ] || [ -z "$WEBSOCKET_SET" ]; then
        echo "🔧 Setting webhook/websocket URLs..."
        gcloud run services update "${SERVICE_NAME}" \
            --region="${REGION}" \
            --update-env-vars="WEBHOOK_BASE_URL=${WEBHOOK_URL},WEBSOCKET_BASE_URL=${WEBSOCKET_URL}" \
            --quiet 2>/dev/null
        echo "✅ URLs configured"
    else
        echo "✅ Webhook URLs already set via .env file"
    fi
    
    # Test health endpoint
    echo ""
    echo "Testing /health endpoint..."
    sleep 2
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Health check passed (HTTP $HTTP_CODE)"
    else
        echo "⚠️  Health check returned HTTP $HTTP_CODE"
        echo "   Service may still be starting up"
    fi
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "=================================================="
echo "  Deployment Complete! ✅"
echo "=================================================="
echo "Service:    ${SERVICE_NAME}"
echo "Region:     ${REGION}"
echo "Image:      ${IMAGE_NAME}:${BUILD_TAG}"
echo "URL:        ${SERVICE_URL}"
echo "Resources:  ${CPU_COUNT} CPU, ${MEMORY_LIMIT} RAM"
echo "Scaling:    ${MIN_INSTANCES}-${MAX_INSTANCES} instances"
echo "=================================================="
echo ""
echo "📋 Quick Commands:"
echo "   Test:    curl ${SERVICE_URL}/health"
echo "   Logs:    gcloud run logs tail ${SERVICE_NAME} --region=${REGION}"
echo "   Metrics: gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo ""

if [ "$TRAFFIC_PERCENTAGE" -ne 100 ]; then
    echo "⚠️  Only ${TRAFFIC_PERCENTAGE}% traffic is routed to new revision"
    echo "   Route 100%: gcloud run services update-traffic ${SERVICE_NAME} --to-latest --region=${REGION}"
    echo ""
fi

