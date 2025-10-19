#!/usr/bin/env python3
"""
Simple WebSocket test without compression for ngrok compatibility
"""
import asyncio
import websockets
import json
import sys

async def test_session(ws_url, meeting_url):
    print(f"🔗 Connecting to: {ws_url}")
    print(f"📺 Meeting URL: {meeting_url}")
    print()
    
    try:
        # Disable compression for ngrok compatibility
        async with websockets.connect(
            ws_url,
            compression=None,  # Disable compression
            open_timeout=10
        ) as ws:
            # Send the request
            request = {
                "action": "start",
                "meeting_url": meeting_url
            }
            
            await ws.send(json.dumps(request))
            print(f"✅ Sent request")
            print()
            print("📡 Waiting for response...")
            print("=" * 60)
            
            # Listen for messages with timeout
            try:
                async for message in ws:
                    data = json.loads(message)
                    
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'error':
                        print(f"\n❌ ERROR: {data.get('error', data.get('message', 'Unknown error'))}")
                        break
                        
                    elif msg_type == 'status':
                        status = data.get('status', '')
                        message_text = data.get('message', '')
                        print(f"\n📍 Status: {status}")
                        print(f"   Message: {message_text}")
                        
                        if 'session_id' in data:
                            print(f"   Session ID: {data['session_id']}")
                        if 'bot_id' in data:
                            print(f"   🤖 Bot ID: {data['bot_id']}")
                            print(f"\n   ✅ SUCCESS! Bot should be joining your meeting now!")
                            print(f"   Check your Google Meet at: https://meet.google.com/gxd-yfqq-jki")
                        
                    else:
                        print(f"\n📨 Received:")
                        print(json.dumps(data, indent=2))
                        
            except asyncio.TimeoutError:
                print("\n⏱️  Timeout waiting for response")
                    
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./venv/bin/python test_prism_simple.py <base_url> [meeting_url]")
        sys.exit(1)
    
    base_url = sys.argv[1]
    meeting_url = sys.argv[2] if len(sys.argv) > 2 else "https://meet.google.com/wqb-debs-oxs"
    
    # Convert HTTP/HTTPS to WebSocket URL
    if base_url.startswith("https://"):
        ws_url = base_url.replace("https://", "wss://") + "/api/v1/prism/start-session"
    elif base_url.startswith("http://"):
        ws_url = base_url.replace("http://", "ws://") + "/api/v1/prism/start-session"
    else:
        ws_url = base_url
    
    print("=" * 60)
    print("🚀 Prism Session Test (No Compression)")
    print("=" * 60)
    
    asyncio.run(test_session(ws_url, meeting_url))

