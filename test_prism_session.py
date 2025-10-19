#!/usr/bin/env python3
"""
Test script for Prism start-session WebSocket endpoint.
Usage: ./venv/bin/python test_prism_session.py <websocket_url>
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
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            # Send the request with action and meeting_url
            request = {
                "action": "start",
                "meeting_url": meeting_url
            }
            
            await ws.send(json.dumps(request))
            print(f"✅ Sent: {json.dumps(request, indent=2)}")
            print()
            print("📡 Listening for updates...")
            print("=" * 60)
            
            # Listen for messages
            message_count = 0
            async for message in ws:
                message_count += 1
                data = json.loads(message)
                
                msg_type = data.get('type', 'unknown')
                
                if msg_type == 'error':
                    print(f"\n❌ ERROR: {data.get('error', data.get('message', 'Unknown error'))}")
                    print(f"Full response: {json.dumps(data, indent=2)}")
                    break
                    
                elif msg_type == 'status':
                    status = data.get('status', '')
                    message = data.get('message', '')
                    print(f"\n📍 Status: {status}")
                    print(f"   Message: {message}")
                    
                    if 'session_id' in data:
                        print(f"   Session ID: {data['session_id']}")
                    if 'bot_id' in data:
                        print(f"   🤖 Bot ID: {data['bot_id']}")
                        print(f"   ✅ BOT CREATED! It should be joining your meeting now...")
                    
                elif msg_type == 'transcript':
                    speaker = data.get('speaker', 'Unknown')
                    text = data.get('text', '')
                    print(f"\n🗣️  {speaker}: {text}")
                    
                elif msg_type == 'bot_speaking':
                    text = data.get('text', '')
                    print(f"\n🤖 Bot speaking: {text}")
                    
                else:
                    print(f"\n📨 Message #{message_count}:")
                    print(json.dumps(data, indent=2))
                
                # Show we're still listening
                if message_count % 5 == 0:
                    print(f"\n💡 Still listening... ({message_count} messages received)")
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"\n❌ Connection failed with status {e.status_code}")
        print(f"   Make sure the server is running at: {ws_url}")
        
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        
    except ConnectionRefusedError:
        print(f"\n❌ Connection refused to: {ws_url}")
        print("   Is the server running?")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./venv/bin/python test_prism_session.py <websocket_url>")
        print("\nExample:")
        print("  ./venv/bin/python test_prism_session.py ws://localhost:8080/api/v1/prism/start-session")
        print("  ./venv/bin/python test_prism_session.py wss://your-ngrok-url.ngrok-free.dev/api/v1/prism/start-session")
        sys.exit(1)
    
    ws_url = sys.argv[1]
    meeting_url = "https://meet.google.com/gxd-yfqq-jki"
    
    print("=" * 60)
    print("🚀 Prism Session Test")
    print("=" * 60)
    
    asyncio.run(test_session(ws_url, meeting_url))
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)

