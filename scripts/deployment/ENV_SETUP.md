# Environment Variables Setup

## Overview

The deploy script automatically reads environment variables from your `.env` file and passes them to Cloud Run during deployment.

## Required Environment Variables

Your `.env` file should contain the following variables:

### Core API Keys
- `API_KEY` - Backend authentication key (auto-generated if missing)
- `GOOGLE_API_KEY` - Google Cloud API key for Gemini AI
- `GOOGLE_PROJECT_ID` - GCP project ID
- `GOOGLE_PROJECT_LOCATION` - GCP location (e.g., us-east1)

### AI Prompts
- `HERMES_BASE_PROMPT` - Base prompt for Hermes persona (automatically mapped to BASE_PROMPT)
- `PRISM_BASE_PROMPT` - Base prompt for Prism voice assistant
- `PRISMA_BASE_PROMPT` - Base prompt for Prisma persona (optional)

### Supabase (Vector Store)
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `SUPABASE_PROJECT_URL` - Supabase project URL
- `SUPABASE_DATABASE_URL` - Supabase PostgreSQL connection string

### MCP Server (Database Query Tool)
- `SUPABASE_MCP_SERVER_URL` - MCP server URL (e.g., http://localhost:3001)
- `SUPABASE_MCP_API_KEY` - MCP server API key for authentication
- `MCP_SERVER_ENABLED` - Enable MCP server integration (default: false)
- `MCP_SERVER_PORT` - MCP server port for Docker Compose (default: 3001)

### Prism (Voice Bot)
- `ATTENDEE_API_KEY` - Attendee API key for voice bot integration

### Storage
- `GCS_BUCKET_NAME` - Google Cloud Storage bucket name
- `TTS_DEVICE` - TTS device (cpu or gpu)

## How It Works

1. **During Deployment**: The `deploy.sh` script reads your `.env` file
2. **Parsing**: Extracts the required environment variables
3. **Cloud Run**: Passes them to Cloud Run using `--set-env-vars`

## Example .env File

```bash
# Core API Keys
API_KEY=your-api-key-here
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_PROJECT_LOCATION=us-east1

# AI Prompts
HERMES_BASE_PROMPT="You are Hermes, an AI assistant..."
PRISM_BASE_PROMPT="You are Prism, a voice assistant..."

# Supabase
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_PROJECT_URL=https://xxxxx.supabase.co
SUPABASE_DATABASE_URL=postgresql://postgres.xxxxx...

# MCP Server (Optional - for database query tool)
MCP_SERVER_ENABLED=true
SUPABASE_MCP_SERVER_URL=http://localhost:3001
SUPABASE_MCP_API_KEY=your-mcp-api-key
MCP_SERVER_PORT=3001

# Prism Voice Bot
ATTENDEE_API_KEY=your-attendee-api-key

# Storage
GCS_BUCKET_NAME=your-bucket-name
TTS_DEVICE=cpu
```

## Security Notes

⚠️ **IMPORTANT**:

1. **Never commit `.env` to git** - It contains sensitive API keys
2. **`.dockerignore` excludes `.env`** - Secrets are NOT baked into the image
3. **Environment variables are passed at runtime** - Cloud Run receives them during deployment
4. **For production**, migrate to Secret Manager for better security:
   ```bash
   ./scripts/deployment/setup-secrets.sh
   ```

## Deployment

When you run the deploy script, it will:

```bash
./scripts/deployment/deploy.sh
```

Output:
```
📋 Loading environment variables from .env file...
   ✓ API_KEY
   ✓ GOOGLE_API_KEY
   ✓ GOOGLE_PROJECT_ID
   ✓ GOOGLE_PROJECT_LOCATION
   ✓ BASE_PROMPT (from HERMES_BASE_PROMPT)
   ✓ PRISM_BASE_PROMPT
   ✓ SUPABASE_SERVICE_ROLE_KEY
   ✓ SUPABASE_PROJECT_URL
   ✓ SUPABASE_DATABASE_URL
   ✓ ATTENDEE_API_KEY
   ✓ GCS_BUCKET_NAME
   ✓ TTS_DEVICE
✅ Environment variables loaded
```

## Verification

After deployment, verify environment variables are set:

```bash
gcloud run services describe master-hermes-backend \
  --region=us-central1 \
  --format="value(spec.template.spec.containers[0].env)"
```

## Troubleshooting

### Missing Environment Variable Error

If you see errors like `ATTENDEE_API_KEY not found in environment`:

1. Check your `.env` file exists
2. Verify the variable is spelled correctly
3. Ensure the value is not empty
4. Re-deploy: `./scripts/deployment/deploy.sh`

### Variable Not Updated

If a variable isn't updating after deployment:

```bash
# Force update specific variables
gcloud run services update master-hermes-backend \
  --region=us-central1 \
  --update-env-vars=VARIABLE_NAME=new_value
```

### Migrating to Secret Manager (Recommended for Production)

1. Run the setup script:
   ```bash
   ./scripts/deployment/setup-secrets.sh
   ```

2. Update `deploy.sh` to use `--set-secrets` instead of `--set-env-vars`

3. Remove sensitive values from `.env` after migration
