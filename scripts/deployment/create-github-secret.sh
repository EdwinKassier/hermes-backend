#!/bin/bash
set -e

# ==============================================================================
# Create GitHub ENV_FILE Secret
# ==============================================================================
# This script helps you create a properly encoded ENV_FILE secret for GitHub
#
# Usage:
#   ./scripts/deployment/create-github-secret.sh
# ==============================================================================

echo "=================================================="
echo "  GitHub ENV_FILE Secret Creator"
echo "=================================================="
echo ""

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found"
    echo ""
    echo "Please create a .env file in the project root first."
    exit 1
fi

# Check if .env has content
if [ ! -s .env ]; then
    echo "❌ ERROR: .env file is empty"
    exit 1
fi

echo "✅ Found .env file"
echo "📊 File size: $(wc -c < .env) bytes"
echo "📊 Number of lines: $(wc -l < .env)"
echo ""

# Show environment variables (keys only, not values)
echo "Environment variables in .env:"
grep -o '^[^=]*' .env | grep -v '^#' | grep -v '^$' | head -20
echo ""

# Encode to base64 without line wraps
echo "🔐 Encoding .env to base64..."
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - base64 doesn't have -w flag
    BASE64_OUTPUT=$(cat .env | base64)
    echo "$BASE64_OUTPUT" | pbcopy
    echo "✅ Base64-encoded .env copied to clipboard!"
    echo ""
    echo "Next steps:"
    echo "1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions"
    echo "2. Click 'New repository secret'"
    echo "3. Name: ENV_FILE"
    echo "4. Value: Paste from clipboard (Cmd+V)"
    echo "5. Click 'Add secret'"
else
    # Linux - use -w 0 to prevent line wrapping
    BASE64_OUTPUT=$(cat .env | base64 -w 0)
    
    # Try to copy to clipboard if xclip is available
    if command -v xclip &> /dev/null; then
        echo "$BASE64_OUTPUT" | xclip -selection clipboard
        echo "✅ Base64-encoded .env copied to clipboard!"
    else
        echo "✅ Base64-encoded .env:"
        echo ""
        echo "$BASE64_OUTPUT"
        echo ""
        echo "(Install xclip to auto-copy: sudo apt-get install xclip)"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions"
    echo "2. Click 'New repository secret'"
    echo "3. Name: ENV_FILE"
    echo "4. Value: Paste the base64 string above"
    echo "5. Click 'Add secret'"
fi

echo ""
echo "=================================================="
echo ""

# Verify the encoding by decoding it
echo "🔍 Verifying encoding (decoding to temp file)..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "$BASE64_OUTPUT" | base64 --decode > /tmp/test.env 2>/dev/null
else
    echo "$BASE64_OUTPUT" | base64 --decode > /tmp/test.env 2>/dev/null
fi

if [ $? -eq 0 ]; then
    if diff -q .env /tmp/test.env > /dev/null 2>&1; then
        echo "✅ Encoding verified - decode produces identical file"
        rm /tmp/test.env
    else
        echo "⚠️  WARNING: Decoded file differs from original"
        echo "This shouldn't happen - please check your base64 command"
    fi
else
    echo "❌ ERROR: Failed to decode - encoding might be invalid"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ ENV_FILE secret is ready to use in GitHub!"
echo "=================================================="

