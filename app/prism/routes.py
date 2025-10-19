"""
Prism Routes - WebSocket and HTTP endpoints

Endpoints:
1. WebSocket: /api/v1/prism/start-session (User-facing)
2. WebSocket: /api/v1/prism/bot-audio (Attendee bot connection)
3. HTTP POST: /api/v1/prism/webhook (Attendee callbacks)

Following patterns from hermes/routes.py for consistency.
"""
import json
import logging
import os
from datetime import datetime
from flask import Blueprint, request, jsonify
from simple_websocket import Server, ConnectionClosed

from .services import get_prism_service
from .schemas import (
    StartSessionRequest,
    SessionStatusResponse,
    AttendeeWebhookPayload,
    BotStateChangeData,
    TranscriptUpdateData,
    ErrorResponse
)
from .constants import SessionStatus, WebhookTrigger
from .exceptions import PrismException, SessionNotFoundError
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Create blueprint
prism_bp = Blueprint('prism', __name__, url_prefix='/api/v1/prism')

# Get service
prism_service = get_prism_service()


# ==============================================================================
# WEBSOCKET: USER SESSION ENDPOINT
# ==============================================================================

@prism_bp.route('/start-session', websocket=True)
def start_session():
    """
    WebSocket endpoint for users to start a Prism session.
    
    Flow:
    1. Client connects with meeting_url
    2. Server creates session and bot
    3. Server sends status updates as bot joins meeting
    4. Server sends transcripts as they arrive
    5. Client can disconnect to end session
    
    Message Format (Client -> Server):
    {
        "action": "start",
        "meeting_url": "https://meet.google.com/xxx-xxxx-xxx"
    }
    
    Message Format (Server -> Client):
    {
        "type": "status",
        "data": {...}
    }
    or
    {
        "type": "transcript",
        "data": {...}
    }
    """
    ws = Server.accept(request.environ)
    session_id = None
    
    try:
        logger.info("User WebSocket connected")
        
        # Wait for start message
        data = ws.receive()
        if not data:
            _ws_send_error(ws, "No data received")
            return ""
        
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            _ws_send_error(ws, "Invalid JSON")
            return ""
        
        # Validate request
        if message.get("action") != "start":
            _ws_send_error(ws, "Expected 'start' action")
            return ""
        
        meeting_url = message.get("meeting_url")
        if not meeting_url:
            _ws_send_error(ws, "meeting_url is required")
            return ""
        
        # Validate meeting URL
        try:
            StartSessionRequest(meeting_url=meeting_url)
        except ValidationError as e:
            _ws_send_error(ws, f"Invalid meeting URL: {str(e)}")
            return ""
        
        # Create session
        session = prism_service.create_session(meeting_url, request=request)
        session_id = session.session_id
        prism_service.mark_user_connected(session_id)
        
        # Send session created status
        _ws_send_status(ws, session, "Session created")
        
        # Get base URLs from environment (set by ngrok script or manually)
        webhook_base_url = os.getenv("WEBHOOK_BASE_URL")
        if not webhook_base_url:
            # Fallback: try to detect ngrok URL from request headers
            # ngrok adds X-Forwarded-* headers
            forwarded_proto = request.headers.get('X-Forwarded-Proto', 'http')
            forwarded_host = request.headers.get('X-Forwarded-Host')
            
            if forwarded_host:
                webhook_base_url = f"{forwarded_proto}://{forwarded_host}"
                logger.info(
                    "Using ngrok URL from headers: %s", webhook_base_url
                )
            else:
                logger.warning(
                    "WEBHOOK_BASE_URL not set. Using request.host_url. "
                    "Attendee webhooks may fail if not publicly accessible."
                )
                webhook_base_url = request.host_url.rstrip('/')
        
        websocket_base_url = os.getenv("WEBSOCKET_BASE_URL")
        if not websocket_base_url:
            # Convert HTTP(S) to WS(S)
            websocket_base_url = webhook_base_url.replace(
                "https://", "wss://"
            ).replace("http://", "ws://")
        
        # Create bot
        try:
            bot_id = prism_service.create_bot(
                session_id=session_id,
                webhook_base_url=webhook_base_url,
                websocket_base_url=websocket_base_url
            )
            _ws_send_status(ws, session, f"Bot created: {bot_id}")
        except Exception as e:
            logger.error(f"Bot creation failed: {str(e)}")
            _ws_send_error(ws, f"Failed to create bot: {str(e)}")
            return ""
        
        # Keep connection alive and send updates
        while True:
            try:
                # Check for incoming messages (ping/pong or close)
                data = ws.receive(timeout=5)
                
                if data:
                    try:
                        message = json.loads(data)
                        
                        # Handle ping
                        if message.get("type") == "ping":
                            ws.send(json.dumps({"type": "pong"}))
                        
                        # Handle close request
                        elif message.get("action") == "close":
                            logger.info("Client requested close")
                            break
                    except json.JSONDecodeError:
                        pass
                
                # Send periodic status updates
                session = prism_service.get_session(session_id)
                
                # Send any new transcripts (check if there are new ones)
                # Note: In a real implementation, we'd track which transcripts were sent
                
            except TimeoutError:
                # Normal timeout, continue loop
                continue
            except ConnectionClosed:
                logger.info("WebSocket connection closed by client")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {str(e)}")
                break
        
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    
    finally:
        # Cleanup
        if session_id:
            try:
                prism_service.mark_user_disconnected(session_id)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
    
    return ""


# ==============================================================================
# WEBSOCKET: BOT AUDIO ENDPOINT
# ==============================================================================

@prism_bp.route('/bot-audio', websocket=True)
def bot_audio():
    """
    WebSocket endpoint for Attendee bot audio streaming.
    
    This endpoint handles bidirectional audio with the Attendee bot:
    - Receives audio from bot (meeting audio) [currently not processed]
    - Receives transcripts from bot (via Google Meet closed captions)
    - Sends audio to bot (AI responses to play in meeting)
    
    Query params:
    - session_id: Session ID to connect to
    
    Message Format (Bot -> Server):
    {
        "type": "transcript",
        "data": {
            "speaker": "John Doe",
            "text": "Hello everyone",
            "is_final": true
        }
    }
    
    Message Format (Server -> Bot):
    {
        "type": "audio_chunk",
        "data": {
            "audio": "base64_encoded_pcm_data",
            "sequence": 0
        }
    }
    """
    ws = Server.accept(request.environ)
    session_id = request.args.get('session_id')
    
    if not session_id:
        ws.send(json.dumps({"type": "error", "message": "session_id required"}))
        ws.close()
        return ""
    
    try:
        # Validate session exists
        prism_service.get_session(session_id)
        prism_service.mark_bot_connected(session_id)

        logger.info(f"Bot WebSocket connected for session {session_id}")
        
        # Send connection confirmation
        ws.send(json.dumps({
            "type": "status",
            "message": "Bot connected"
        }))
        
        # Handle bidirectional communication
        while True:
            try:
                # Check for incoming messages from bot
                data = ws.receive(timeout=1)
                
                if data:
                    # Check if data is binary (audio) or text (JSON)
                    if isinstance(data, bytes):
                        # Binary data = incoming audio from meeting
                        # We don't currently process incoming audio, just log receipt
                        logger.debug(f"📥 Received binary audio data from bot: {len(data)} bytes")
                    else:
                        # Text data = JSON messages (transcripts, etc.)
                        try:
                            message = json.loads(data)
                            _handle_bot_message(session_id, message)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON from bot: {str(e)}")
                
                # Check for pending audio to send to bot
                # Note: Attendee API expects RAW binary PCM data, not JSON
                pending_audio = prism_service.get_pending_audio(session_id)
                for audio_chunk in pending_audio:
                    try:
                        # Send raw PCM bytes directly (no JSON wrapping)
                        # Attendee expects 16-bit PCM audio at the specified sample rate
                        ws.send(audio_chunk.data)
                        logger.info(f"📤 Sent raw PCM audio chunk {audio_chunk.sequence} ({len(audio_chunk.data)} bytes)")
                        
                        # CRITICAL: Pace audio chunks to match playback duration
                        # 16kHz, 16-bit mono = 32000 bytes/sec
                        # 4096 bytes = 128ms of audio
                        # For the first 10 chunks, send IMMEDIATELY with no delay to establish stream
                        # For chunks 10-20, send at 10% speed
                        # Then pace normally to match playback duration
                        import time
                        chunk_duration = len(audio_chunk.data) / 32000.0  # seconds
                        
                        if audio_chunk.sequence < 10:
                            # Send first 10 chunks IMMEDIATELY with no delay
                            # This establishes the audio stream as fast as possible
                            pass  # No sleep, send immediately
                        elif audio_chunk.sequence < 20:
                            # Send next 10 chunks very fast to keep stream active
                            time.sleep(chunk_duration * 0.1)  # 10% speed
                        else:
                            # Normal pacing for actual speech
                            time.sleep(chunk_duration)
                    except Exception as e:
                        logger.error(f"Failed to send audio chunk: {str(e)}")
                
            except TimeoutError:
                # Normal timeout, continue loop
                continue
            except ConnectionClosed:
                logger.info("Bot WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in bot WebSocket loop: {str(e)}")
                break
    
    except SessionNotFoundError:
        ws.send(json.dumps({"type": "error", "message": "Session not found"}))
        ws.close()
        return ""
    except Exception as e:
        logger.error(f"Bot WebSocket error: {str(e)}")
    
    finally:
        # Cleanup
        try:
            prism_service.mark_bot_disconnected(session_id)
        except Exception as e:
            logger.error(f"Bot cleanup error: {str(e)}")
    
    return ""


def _handle_bot_message(session_id: str, message: dict):
    """
    Handle incoming message from Attendee bot.
    
    Args:
        session_id: Session ID
        message: Parsed JSON message
    """
    msg_type = message.get("type")
    
    if msg_type == "transcript":
        # Handle transcript update
        data = message.get("data", {})
        speaker = data.get("speaker", "Unknown")
        text = data.get("text", "")
        is_final = data.get("is_final", True)
        
        if text:
            prism_service.handle_transcript(
                session_id=session_id,
                speaker=speaker,
                text=text,
                is_final=is_final
            )
    
    elif msg_type == "audio_chunk":
        # Handle incoming audio (not currently used, but ready for future)
        logger.debug("Received audio chunk from bot (not processed)")
    
    elif msg_type == "status":
        # Handle status message
        logger.info(f"Bot status: {message.get('message')}")
    
    elif msg_type == "ping":
        # Ping handled at WebSocket level
        pass
    
    else:
        # Log unknown message types with full message for debugging
        logger.debug(f"Unhandled message from bot: type={msg_type}, message={message}")


# ==============================================================================
# HTTP: WEBHOOK ENDPOINT
# ==============================================================================

@prism_bp.route('/webhook', methods=['POST'])
def webhook():
    """
    Webhook endpoint for Attendee callbacks.
    
    Handles:
    - bot.state_change: Bot joined/left meeting
    - transcript.update: New transcript from closed captions
    - bot.error: Bot encountered an error
    
    Payload format:
    {
        "trigger": "bot.state_change" | "transcript.update" | "bot.error",
        "bot_id": "bot_xxx",
        "data": {...}
    }
    """
    try:
        payload = request.get_json()
        
        if not payload:
            return jsonify({"error": "No payload"}), 400
        
        # Log raw payload for debugging
        logger.info(f"=== WEBHOOK RECEIVED === Payload: {payload}")
        
        # Validate payload
        try:
            webhook_data = AttendeeWebhookPayload(**payload)
            logger.info(f"Webhook validated: trigger={webhook_data.trigger}, bot_id={webhook_data.bot_id}, data={webhook_data.data}")
        except ValidationError as e:
            logger.error(f"Invalid webhook payload: {str(e)}")
            return jsonify({"error": str(e)}), 400
        
        # Get session by bot ID
        session = prism_service.get_session_by_bot_id(webhook_data.bot_id)
        if not session:
            logger.warning(f"Received webhook for unknown bot: {webhook_data.bot_id}")
            return jsonify({"error": "Bot not found"}), 404
        
        # Handle different triggers
        if webhook_data.trigger == WebhookTrigger.BOT_STATE_CHANGE:
            _handle_bot_state_change(session.session_id, webhook_data)
        
        elif webhook_data.trigger == WebhookTrigger.TRANSCRIPT_UPDATE:
            _handle_transcript_update(session.session_id, webhook_data)
        
        elif webhook_data.trigger == WebhookTrigger.BOT_ERROR:
            _handle_bot_error(session.session_id, webhook_data)
        
        else:
            logger.warning(f"Unknown webhook trigger: {webhook_data.trigger}")
        
        return jsonify({"status": "ok"}), 200
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def _handle_bot_state_change(session_id: str, webhook_data: AttendeeWebhookPayload):
    """Handle bot state change webhook."""
    try:
        logger.info(f"=== HANDLING STATE CHANGE === Session: {session_id}, Bot: {webhook_data.bot_id}")
        logger.info(f"State change data: {webhook_data.data}")
        state_data = BotStateChangeData(**webhook_data.data)
        logger.info(f"Calling handle_bot_state_change with state: {state_data.new_state}")
        prism_service.handle_bot_state_change(
            bot_id=webhook_data.bot_id,
            state=state_data.new_state,
            error=state_data.error_message
        )
        logger.info(f"✅ Bot state changed successfully: {state_data.new_state}")
    except ValidationError as e:
        logger.error(f"❌ Invalid state change data: {str(e)}")


def _handle_transcript_update(session_id: str, webhook_data: AttendeeWebhookPayload):
    """Handle transcript update webhook."""
    try:
        transcript_data = TranscriptUpdateData(**webhook_data.data)
        prism_service.handle_transcript(
            session_id=session_id,
            speaker=transcript_data.speaker,
            text=transcript_data.text,
            is_final=transcript_data.is_final,
            idempotency_key=webhook_data.idempotency_key
        )
        logger.info(f"Transcript: {transcript_data.speaker}: {transcript_data.text}")
    except ValidationError as e:
        logger.error(f"Invalid transcript data: {str(e)}")


def _handle_bot_error(session_id: str, webhook_data: AttendeeWebhookPayload):
    """Handle bot error webhook."""
    error_message = webhook_data.data.get("message", "Unknown error")
    logger.error(f"Bot error for session {session_id}: {error_message}")
    
    # Update session status
    session = prism_service.get_session(session_id)
    session.update_status(SessionStatus.ERROR, error=error_message)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _ws_send_error(ws: Server, message: str):
    """Send error message via WebSocket."""
    try:
        ws.send(json.dumps({
            "type": "error",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }))
    except Exception as e:
        logger.error(f"Failed to send error via WebSocket: {str(e)}")


def _ws_send_status(ws: Server, session, message: str):
    """Send status update via WebSocket."""
    try:
        response = SessionStatusResponse(
            session_id=session.session_id,
            status=session.status.value,
            bot_state=session.bot_state.value if session.bot_state else None,
            bot_id=session.bot_id,
            message=message
        )
        # Use model_dump_json() to properly serialize datetime fields
        ws.send(response.model_dump_json())
    except Exception as e:
        logger.error(f"Failed to send status via WebSocket: {str(e)}")


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@prism_bp.errorhandler(PrismException)
def handle_prism_exception(error: PrismException):
    """Handle Prism domain exceptions."""
    response = ErrorResponse(
        error=error.__class__.__name__,
        message=error.message,
        status_code=error.status_code
    )
    return jsonify(response.dict()), error.status_code


@prism_bp.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors."""
    response = ErrorResponse(
        error="NotFound",
        message="Resource not found",
        status_code=404
    )
    return jsonify(response.dict()), 404


@prism_bp.errorhandler(500)
def handle_internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    response = ErrorResponse(
        error="InternalServerError",
        message="An internal error occurred",
        status_code=500
    )
    return jsonify(response.dict()), 500

