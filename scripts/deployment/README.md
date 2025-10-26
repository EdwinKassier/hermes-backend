# Deployment Scripts

## Quick Start

Deploy to Google Cloud Run:

```bash
./scripts/deployment/deploy.sh
```

## Scripts

### `deploy.sh` - Main Deployment Script

Unified deployment script with smart defaults and options.

**Quick Deploy (default):**
```bash
./scripts/deployment/deploy.sh
```

**Clean Build:**
```bash
./scripts/deployment/deploy.sh --clean
```

**With Diagnostics:**
```bash
./scripts/deployment/deploy.sh --check
```

**With Verification:**
```bash
./scripts/deployment/deploy.sh --verify
```

**Build Only (no deploy):**
```bash
./scripts/deployment/deploy.sh --no-deploy
```

**Custom Resources:**
```bash
CPU_COUNT=4 MEMORY_LIMIT=8Gi ./scripts/deployment/deploy.sh
```

**Debug Environment Variables:**
```bash
SHOW_ENV_FILE=true ./scripts/deployment/deploy.sh
```

**Gradual Rollout:**
```bash
TRAFFIC_PERCENTAGE=10 ./scripts/deployment/deploy.sh
```

**Help:**
```bash
./scripts/deployment/deploy.sh --help
```

### `setup-secrets.sh` - Configure Secret Manager

Set up secrets in Google Cloud Secret Manager for production deployment.

```bash
./scripts/deployment/setup-secrets.sh
```

This script will prompt you for:
- API keys (Google, Attendee, etc.)
- Database credentials (Supabase)
- Base prompts for AI personas

### `verify-build.sh` - Test Docker Build Locally

Verify Docker build works before deploying.

```bash
./scripts/deployment/verify-build.sh
```

Tests:
- Docker build succeeds
- Gunicorn is installed
- Application starts correctly

## Environment Variables

### GCP Configuration
- `GCP_PROJECT_ID` - GCP project (default: edwin-portfolio-358212)
- `SERVICE_NAME` - Cloud Run service (default: master-hermes-backend)
- `REGION` - GCP region (default: us-central1)

### Resource Configuration
- `CPU_COUNT` - Number of CPUs (default: 2)
- `MEMORY_LIMIT` - Memory limit (default: 4Gi)
- `MIN_INSTANCES` - Min instances (default: 0)
- `MAX_INSTANCES` - Max instances (default: 10)
- `CONCURRENCY` - Requests per instance (default: 50)
- `REQUEST_TIMEOUT` - Request timeout (default: 300s)

### Deployment Configuration
- `ALLOW_UNAUTH` - Allow unauthenticated (default: true)
- `TRAFFIC_PERCENTAGE` - Traffic to new revision (default: 100)
- `BUILD_TAG` - Custom image tag (default: timestamp)

## Common Workflows

### First Time Setup

1. **Ensure .env file has all required variables:**
   ```bash
   # Check your .env file has:
   # - API_KEY
   # - GOOGLE_API_KEY
   # - GOOGLE_PROJECT_ID
   # - ATTENDEE_API_KEY
   # - SUPABASE credentials
   # See ENV_SETUP.md for complete list
   ```

2. **Verify build locally (optional):**
   ```bash
   ./scripts/deployment/verify-build.sh
   ```

3. **Deploy with checks:**
   ```bash
   ./scripts/deployment/deploy.sh --check --verify
   ```

**Note**: The deploy script automatically reads from your `.env` file and passes all variables to Cloud Run.

### Regular Deployment

```bash
./scripts/deployment/deploy.sh
```

### Production Deployment

```bash
# Clean build with verification
./scripts/deployment/deploy.sh --clean --verify

# Or gradual rollout
TRAFFIC_PERCENTAGE=10 ./scripts/deployment/deploy.sh
```

### Development Deployment

```bash
# Fast deployment with cache
./scripts/deployment/deploy.sh --quick
```

## Troubleshooting

### Build is slow or hangs

**Run diagnostics:**
```bash
./scripts/deployment/deploy.sh --check
```

**Clean up Docker:**
```bash
docker system prune -a
```

**Use cache:**
```bash
./scripts/deployment/deploy.sh --quick
```

### Deployment fails

**Check logs:**
```bash
gcloud run logs tail master-hermes-backend --region=us-central1
```

**Verify build locally first:**
```bash
./scripts/deployment/verify-build.sh
```

**Build and push only:**
```bash
./scripts/deployment/deploy.sh --no-deploy
```

### Authentication issues

**Re-authenticate:**
```bash
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Tips

1. **Use cache by default** - Much faster for iterative development
2. **Run --check periodically** - Identifies issues early
3. **Use --verify before production** - Catches build problems locally
4. **Clean Docker regularly** - Prevents disk space issues
5. **Start with gradual rollout** - Safer for production changes

## Quick Reference

```bash
# Standard deploy
./scripts/deployment/deploy.sh

# Clean deploy
./scripts/deployment/deploy.sh --clean

# Full checks
./scripts/deployment/deploy.sh --check --verify

# High resources
CPU_COUNT=4 MEMORY_LIMIT=8Gi ./scripts/deployment/deploy.sh

# Gradual rollout
TRAFFIC_PERCENTAGE=10 ./scripts/deployment/deploy.sh

# Service status
gcloud run services describe master-hermes-backend --region=us-central1

# View logs
gcloud run logs tail master-hermes-backend --region=us-central1

# Rollback
gcloud run services update-traffic master-hermes-backend \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=us-central1
```

