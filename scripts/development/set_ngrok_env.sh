#!/bin/bash
# Helper script to get ngrok URL and set environment variables
# Can be sourced in your shell: source scripts/set_ngrok_env.sh

NGROK_API="http://localhost:4040/api/tunnels"

# Check if ngrok is running
if ! curl -s $NGROK_API > /dev/null 2>&1; then
    echo "❌ Error: ngrok is not running"
    echo "Start ngrok first: make ngrok"
    return 1 2>/dev/null || exit 1
fi

# Get ngrok URL
NGROK_URL=$(curl -s $NGROK_API | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)

if [ -z "$NGROK_URL" ]; then
    echo "❌ Error: Could not get ngrok URL"
    return 1 2>/dev/null || exit 1
fi

# Convert to WebSocket URL
NGROK_WSS_URL=$(echo $NGROK_URL | sed 's/https/wss/')

# Export variables
export WEBHOOK_BASE_URL=$NGROK_URL
export WEBSOCKET_BASE_URL=$NGROK_WSS_URL

echo "✅ Environment variables set:"
echo "   WEBHOOK_BASE_URL=$WEBHOOK_BASE_URL"
echo "   WEBSOCKET_BASE_URL=$WEBSOCKET_BASE_URL"
echo ""
echo "You can now start your server:"
echo "   python run.py"

