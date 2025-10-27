#!/bin/bash
set -e  # Exit on any error

# ==============================================================================
# Secret Manager Setup Script
# ==============================================================================
# This script helps you create secrets in Google Cloud Secret Manager
# for use with Cloud Run deployment.
#
# Usage:
#   1. Set your GCP project: export GCP_PROJECT_ID="your-project-id"
#   2. Run this script: ./scripts/setup-secrets.sh
#   3. The script will prompt you for each secret value
#
# Note: Secrets are stored securely in Secret Manager, not in your code or .env files
# ==============================================================================

echo "=================================================="
echo "  Google Cloud Secret Manager Setup"
echo "=================================================="
echo ""

# Get project ID
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ ERROR: GCP_PROJECT_ID not set and no default project found."
    echo "   Set it with: export GCP_PROJECT_ID='your-project-id'"
    exit 1
fi

echo "📦 Project: $PROJECT_ID"
echo ""

# Function to create or update a secret
create_or_update_secret() {
    local SECRET_NAME=$1
    local SECRET_DESCRIPTION=$2
    local SECRET_VALUE=$3

    echo "🔑 Processing: $SECRET_NAME"

    # Check if secret already exists
    if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" &>/dev/null; then
        echo "   Secret exists. Updating to new version..."
        echo -n "$SECRET_VALUE" | gcloud secrets versions add "$SECRET_NAME" \
            --project="$PROJECT_ID" \
            --data-file=- &>/dev/null
        echo "   ✅ Updated"
    else
        echo "   Creating new secret..."
        echo -n "$SECRET_VALUE" | gcloud secrets create "$SECRET_NAME" \
            --project="$PROJECT_ID" \
            --replication-policy="automatic" \
            --data-file=- &>/dev/null
        echo "   ✅ Created"
    fi
    echo ""
}

# Function to prompt for secret value
prompt_for_secret() {
    local SECRET_NAME=$1
    local SECRET_DESCRIPTION=$2
    local ENV_VAR_NAME=$3

    # Check if already set in environment
    if [ -n "${!ENV_VAR_NAME}" ]; then
        echo "🔍 Found $ENV_VAR_NAME in environment"
        read -p "Use this value? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            create_or_update_secret "$SECRET_NAME" "$SECRET_DESCRIPTION" "${!ENV_VAR_NAME}"
            return
        fi
    fi

    echo "📝 Enter value for: $SECRET_DESCRIPTION"
    read -s SECRET_VALUE
    echo ""

    if [ -z "$SECRET_VALUE" ]; then
        echo "⏭️  Skipping (empty value)"
        echo ""
        return
    fi

    create_or_update_secret "$SECRET_NAME" "$SECRET_DESCRIPTION" "$SECRET_VALUE"
}

echo "=================================================="
echo "  Core API Keys"
echo "=================================================="
echo ""

prompt_for_secret "api-key" "API Key for backend authentication" "API_KEY"
prompt_for_secret "google-api-key" "Google Cloud API Key" "GOOGLE_API_KEY"
prompt_for_secret "google-project-id" "Google Cloud Project ID" "GOOGLE_PROJECT_ID"
prompt_for_secret "google-project-location" "Google Cloud Project Location (e.g., us-central1)" "GOOGLE_PROJECT_LOCATION"

echo "=================================================="
echo "  Prism Domain (Attendee API)"
echo "=================================================="
echo ""

prompt_for_secret "attendee-api-key" "Attendee API Key for voice bot" "ATTENDEE_API_KEY"
prompt_for_secret "prism-base-prompt" "Base prompt for Prism AI persona" "PRISM_BASE_PROMPT"

echo "=================================================="
echo "  Supabase (Vector Store)"
echo "=================================================="
echo ""

prompt_for_secret "supabase-url" "Supabase Project URL" "SUPABASE_URL"
prompt_for_secret "supabase-database-url" "Supabase Database URL (usually same as project URL)" "SUPABASE_DATABASE_URL"
prompt_for_secret "supabase-service-role-key" "Supabase Service Role Key" "SUPABASE_SERVICE_ROLE_KEY"

echo "=================================================="
echo "  Base Prompts (Optional)"
echo "=================================================="
echo ""

prompt_for_secret "base-prompt" "Base prompt for Hermes persona" "BASE_PROMPT"
prompt_for_secret "prisma-base-prompt" "Base prompt for Prisma persona (if used)" "PRISMA_BASE_PROMPT"

echo ""
echo "=================================================="
echo "  ✅ Secret Manager Setup Complete!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo "   1. Grant Cloud Run access to secrets:"
echo "      gcloud projects add-iam-policy-binding $PROJECT_ID \\"
echo "        --member=serviceAccount:$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com \\"
echo "        --role=roles/secretmanager.secretAccessor"
echo ""
echo "   2. Deploy your application:"
echo "      ./scripts/deployment/deploy.sh"
echo ""
echo "📖 View secrets:"
echo "   gcloud secrets list --project=$PROJECT_ID"
echo ""
echo "🔐 Security reminder:"
echo "   - Secrets are encrypted at rest in Secret Manager"
echo "   - Never commit secrets to git"
echo "   - Rotate secrets regularly"
echo ""
