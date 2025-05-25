#!/bin/bash
# Set variables
PROJECT_ID="edwin-portfolio-358212"
SERVICE_NAME="master-hermes-backend"
REGION="us-central1"
IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/ashes/$SERVICE_NAME"
MEMORY_LIMIT="4096Mi"  # Specify the memory limit
# Define target platforms
PLATFORMS="linux/x86_64"
# Step 1: Build the multi-platform Docker image
echo "Building the multi-platform Docker image..."
docker build \
  --platform "$PLATFORMS" \
  -t "$IMAGE_NAME" \
  --push \
  .
# Step 2: Deploy to Cloud Run with increased memory and second generation
echo "Deploying to Cloud Run (2nd Gen) with increased resources..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --memory "$MEMORY_LIMIT" \
    --cpu "4" \
    --allow-unauthenticated \
    --set-cloudsql-instances="" \
    --set-env-vars=PROJECT_ID="$PROJECT_ID" \
    --set-env-vars=SERVICE_NAME="$SERVICE_NAME" \
    --set-env-vars=REGION="$REGION" \
    --set-env-vars=MEMORY_LIMIT="$MEMORY_LIMIT" \
    --set-env-vars=CPU_LIMIT="1"
echo "Deployment complete!"