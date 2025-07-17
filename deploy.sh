#!/bin/bash
# Set variables
PROJECT_ID="edwin-portfolio-358212"
SERVICE_NAME="master-hermes-backend"
REGION="us-central1"
ARTIFACT_REGISTRY_REPO="ashes"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}"

# Instance Configuration
CPU_COUNT="2"         # Adjust based on your needs
MEMORY_LIMIT="4Gi"    # Adjust based on your needs
MAX_INSTANCES="1"     # Adjust based on your needs
CONCURRENCY="80"      # Default concurrency for Cloud Run
REQUEST_TIMEOUT="300s" # 5 minutes default timeout

# Define target platform
PLATFORM="linux/amd64"

# Step 1: Build the multi-platform Docker image and push to Artifact Registry
echo "Building and pushing the Docker image for ${PLATFORM}..."
docker buildx build \
  --platform "${PLATFORM}" \
  -t "${IMAGE_NAME}:latest" \
  --push \
  .

# Check if build was successful
if [ $? -ne 0 ]; then
  echo "Docker build or push failed. Exiting."
  exit 1
fi

# Step 2: Deploy to Cloud Run
echo "Deploying to Cloud Run (2nd Gen) with specified resources..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --platform managed \
    --region "${REGION}" \
    --execution-environment gen2 \
    --cpu "${CPU_COUNT}" \
    --memory "${MEMORY_LIMIT}" \
    --no-cpu-throttling \
    --concurrency "${CONCURRENCY}" \
    --timeout "${REQUEST_TIMEOUT}" \
    --max-instances "${MAX_INSTANCES}" \
    --allow-unauthenticated \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},SERVICE_NAME=${SERVICE_NAME},REGION_DEPLOYED=${REGION}"

# Check if deployment was successful
if [ $? -ne 0 ]; then
  echo "Cloud Run deployment failed. Exiting."
  exit 1
fi

echo "Deployment complete!"
echo "Service URL: $(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)')"