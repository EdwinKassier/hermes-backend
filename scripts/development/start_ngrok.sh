#!/bin/bash
# Start ngrok tunnel for local development
# This exposes your local server to the internet for Attendee API webhooks

set -e

PORT=${1:-8080}
NGROK_CONFIG_FILE=".ngrok.yml"

echo "🚀 Starting ngrok tunnel for port $PORT..."
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Error: ngrok is not installed"
    echo ""
    echo "Install ngrok:"
    echo "  macOS:  brew install ngrok/ngrok/ngrok"
    echo "  Or download from: https://ngrok.com/download"
    echo ""
    exit 1
fi

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo "⚠️  Warning: ngrok may not be authenticated"
    echo "Run: ngrok config add-authtoken <your-token>"
    echo "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo ""
fi

# Start ngrok
echo "📡 Creating tunnel..."
echo ""
ngrok http $PORT --log=stdout

# Note: ngrok will run in foreground
# Press Ctrl+C to stop
