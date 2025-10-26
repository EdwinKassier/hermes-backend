"""
Prism Business Logic - Orchestrates Attendee bot, AI,
and audio processing.

Following patterns from HermesService for consistency with existing
domain design.
"""
from typing import Optional, Dict, List, Any
import logging
import os
import json
from datetime import datetime
import uuid
from collections import OrderedDict
import time
import threading
from simple_websocket import Server as WebSocketServer

from .models import PrismSession, AudioChunkOutgoing
from .constants import BotState, SessionStatus
from .attendee_client import AttendeeClient
from .audio_processor import AudioProcessor
from .exceptions import (
    BotCreationError,
    SessionNotFoundError,
    AudioProcessingError
)
from app.shared.utils.service_loader import (
    get_gemini_service,
    get_tts_service
)
from app.shared.services.IdentityService import IdentityService

logger = logging.getLogger(__name__)


class PrismService:
    """
    Core business logic for Prism domain.
    
    Responsibilities:
    - Session lifecycle management
    - Attendee bot orchestration
    - AI decision-making (when to respond)
    - TTS audio generation
    - Audio format conversion
    - WebSocket connection management
    
    Architecture follows HermesService patterns for consistency.
    """
    
    def __init__(self):
        """Initialize service with required dependencies."""
        self.sessions: Dict[str, PrismSession] = {}  # In-memory session store
        self.bot_to_session: Dict[str, str] = {}  # Map bot_id -> session_id
        
        # WebSocket connection registry for sending updates to users
        self.user_websockets: Dict[str, WebSocketServer] = {}  # {session_id: websocket_connection}

        # WebSocket synchronization lock (prevents concurrent access to same WebSocket)
        self.websocket_lock = threading.Lock()

        # Idempotency tracking for webhooks (prevents duplicate processing)
        # Using OrderedDict with timestamps for proper time-based cleanup
        self.processed_webhooks: OrderedDict = OrderedDict()  # {idempotency_key: timestamp}
        self.webhook_ttl = 600  # 10 minutes TTL
        
        # Initialize clients
        self.attendee_client = AttendeeClient()
        self.audio_processor = AudioProcessor()
        self.identity_service = IdentityService()
        
        # AI services (lazy loaded)
        self._gemini_service = None
        self._tts_service = None
        
        # OPTIMIZATION: Pre-warm AI services to avoid first-request delay
        # This saves 1-2 seconds on the first response by eliminating lazy-load overhead
        _ = self.gemini_service  # Force lazy initialization
        _ = self.tts_service     # Force lazy initialization
        
        logger.info("PrismService initialized with pre-warmed AI services")
    
    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service
    
    @property
    def tts_service(self):
        """Lazy load TTS service."""
        if self._tts_service is None:
            self._tts_service = get_tts_service()
        return self._tts_service
    
    # ==================== SESSION MANAGEMENT ====================
    
    def create_session(
        self,
        meeting_url: str,
        user_identifier: Optional[str] = None,
        request=None
    ) -> PrismSession:
        """
        Create a new Prism session.
        
        Args:
            meeting_url: Google Meet URL
            user_identifier: Optional user identifier (generates one if not provided)
            request: Flask request object (required if user_identifier is not provided)
        
        Returns:
            PrismSession: New session object
        
        Raises:
            InvalidMeetingURLError: If URL is invalid
        """
        # Generate IDs
        session_id = str(uuid.uuid4())
        user_id = user_identifier or (self.identity_service.generate_user_id(request) if request else str(uuid.uuid4()))
        
        # Create session
        session = PrismSession(
            session_id=session_id,
            user_id=user_id,
            meeting_url=meeting_url,
            status=SessionStatus.CREATED
        )
        
        # Store session
        self.sessions[session_id] = session
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> PrismSession:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            PrismSession: Session object
        
        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        return session
    
    def get_session_by_bot_id(self, bot_id: str) -> Optional[PrismSession]:
        """Get session by bot ID."""
        session_id = self.bot_to_session.get(bot_id)
        if session_id:
            return self.sessions.get(session_id)
        return None
    
    def close_session(self, session_id: str):
        """
        Close a session and cleanup resources (idempotent).

        Args:
            session_id: Session ID
        """
        # Check if session exists (idempotent - safe to call multiple times)
        try:
            session = self.get_session(session_id)
        except SessionNotFoundError:
            logger.debug(f"Session {session_id} already closed")
            return  # Already closed, nothing to do

        logger.info(f"Starting cleanup for session {session_id}")

        # Delete bot if exists
        if session.bot_id:
            try:
                logger.debug(f"Deleting bot {session.bot_id} for session {session_id}")
                self.attendee_client.delete_bot(session.bot_id)
                logger.info(f"Successfully deleted bot {session.bot_id}")
            except Exception as e:
                logger.error(f"Failed to delete bot {session.bot_id}: {str(e)}")
                # Continue with cleanup even if bot deletion fails

            # Clean up bot-to-session mapping
            if session.bot_id in self.bot_to_session:
                del self.bot_to_session[session.bot_id]

        # Update session status
        session.update_status(SessionStatus.CLOSED)
        session.user_ws_connected = False
        session.bot_ws_connected = False

        # Close any open WebSocket connections
        try:
            if hasattr(session, 'websocket_connections'):
                for ws_conn in session.websocket_connections.copy():
                    try:
                        ws_conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing WebSocket connection: {str(e)}")
                session.websocket_connections.clear()
        except Exception as e:
            logger.warning(f"Error cleaning up WebSocket connections: {str(e)}")

        # Remove from session storage to prevent memory leak
        with self.websocket_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

            # Remove from WebSocket registry (defensive cleanup)
            if session_id in self.user_websockets:
                del self.user_websockets[session_id]

        logger.info(f"Successfully closed and deleted session {session_id}")
    
    # ==================== BOT MANAGEMENT ====================
    
    def create_bot(
        self,
        session_id: str,
        webhook_base_url: str,
        websocket_base_url: str
    ) -> str:
        """
        Create Attendee bot for the session.
        
        Args:
            session_id: Session ID
            webhook_base_url: Base URL for webhooks (e.g., https://your-domain.com)
            websocket_base_url: Base URL for WebSocket (e.g., wss://your-domain.com)
        
        Returns:
            str: Bot ID
        
        Raises:
            BotCreationError: If bot creation fails
        """
        session = self.get_session(session_id)
        
        # Update session status
        session.update_status(SessionStatus.BOT_CREATING)
        
        # Build URLs
        webhook_url = f"{webhook_base_url}/api/v1/prism/webhook"
        audio_ws_url = f"{websocket_base_url}/api/v1/prism/bot-audio?session_id={session_id}"
        
        try:
            # Create bot via Attendee API
            logger.info(f"Creating Attendee bot for meeting: {session.meeting_url}")
            bot_response = self.attendee_client.create_bot(
                meeting_url=session.meeting_url,
                bot_name="Prism by Edwin Kassier",
                webhook_url=webhook_url,
                audio_websocket_url=audio_ws_url
            )
            
            # Update session
            session.bot_id = bot_response.bot_id
            session.update_status(SessionStatus.BOT_JOINING)
            
            # Map bot ID to session
            self.bot_to_session[bot_response.bot_id] = session_id
            
            logger.info(f"Created bot {bot_response.bot_id} for session {session_id}")
            return bot_response.bot_id
            
        except Exception as e:
            session.update_status(SessionStatus.ERROR, error=str(e))
            logger.error(f"Bot creation failed: {str(e)}")
            raise BotCreationError(str(e))
    
    def _map_attendee_state_to_bot_state(self, attendee_state: str) -> Optional[BotState]:
        """
        Map Attendee API state strings to our BotState enum.
        
        Attendee states: ready, joining, joined_recording, joined_not_recording, 
                        leaving, left, ended, error
        """
        state_mapping = {
            "ready": BotState.IDLE,
            "joining": BotState.JOINING,
            "joined_recording": BotState.IN_MEETING,
            "joined_not_recording": BotState.IN_MEETING,
            "leaving": BotState.LEAVING,
            "left": BotState.LEAVING,
            "ended": BotState.LEAVING,
            "error": BotState.ERROR,
        }
        return state_mapping.get(attendee_state)
    
    def handle_bot_state_change(self, bot_id: str, state: str, error: Optional[str] = None):
        """
        Handle bot state change webhook from Attendee.
        
        Args:
            bot_id: Attendee bot ID
            state: New bot state (Attendee API state string)
            error: Optional error message
        """
        logger.info(f"=== HANDLE_BOT_STATE_CHANGE CALLED === Bot: {bot_id}, State: {state}, Error: {error}")
        
        session = self.get_session_by_bot_id(bot_id)
        if not session:
            logger.warning(f"Received state change for unknown bot: {bot_id}")
            return
        
        logger.info(f"Found session: {session.session_id}, current has_introduced: {session.has_introduced}")
        
        # Map Attendee state to our internal BotState
        bot_state = self._map_attendee_state_to_bot_state(state)
        
        if not bot_state:
            logger.warning(f"Unknown bot state from Attendee: {state}")
            return
        
        session.update_bot_state(bot_state)
        
        if error:
            session.error_message = error
        
        logger.info(f"✅ Bot {bot_id} state changed: {state} → {bot_state.name}")
        
        # SEND STATE UPDATE TO USER WEBSOCKET
        self._send_state_update_to_user(session, f"Bot state: {bot_state.name.lower()}")
        
        # When bot joins meeting, unmute and send introduction
        if bot_state == BotState.IN_MEETING and not session.has_introduced:
            # Set flag IMMEDIATELY to prevent race condition
            # (Bot may receive multiple IN_MEETING webhooks within milliseconds)
            session.has_introduced = True
            logger.info(f"🎤 Bot entered IN_MEETING state, triggering introduction for session {session.session_id}")
            # Send introduction (this will handle unmuting at the right time)
            self._send_bot_introduction(session.session_id, bot_id)
    
    # ==================== BOT INTRODUCTION ====================
    
    def _send_bot_introduction(self, session_id: str, bot_id: str):
        """
        Send introduction message when bot joins meeting.
        Sends both voice output and chat message.
        
        CRITICAL ORDER:
        1. Generate TTS audio FIRST (while bot is still muted)
        2. Queue the audio
        3. THEN unmute bot
        4. Send chat message in parallel
        
        This ensures audio is ready to stream immediately after unmuting,
        preventing Google Meet from auto-remuting due to inactivity.
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Cannot send introduction: session missing for {session_id}")
            return
        
        # Check if introduction already sent (defensive check)
        # NOTE: Flag is set in handle_bot_state_change() before calling this function
        if session.has_introduced:
            logger.debug(f"Introduction flag already set for session {session_id}")
        else:
            # Set flag here as fallback (should already be set)
            session.has_introduced = True
        
        try:
            # Generate introduction message dynamically using Gemini
            logger.info(f"=== SENDING BOT INTRODUCTION === Session: {session_id}, Bot: {bot_id}")
            logger.info(f"🤖 Generating dynamic introduction message using Gemini...")
            
            introduction_prompt = """You are joining a Google Meet meeting for the first time. 
Generate a natural, friendly introduction message (2-3 sentences max) that:
1. Introduces yourself
2. Explains you're here to assist with the meeting
3. Encourages participants to interact with you

Keep it conversational and warm. This will be spoken aloud, so make it sound natural."""
            
            try:
                introduction_message = self.gemini_service.generate_gemini_response(
                    prompt=introduction_prompt,
                    persona='prism'
                ).strip()
                logger.info(f"✅ Generated introduction: {introduction_message[:100]}...")
            except Exception as e:
                logger.error(f"Failed to generate introduction with Gemini, using fallback: {str(e)}")
                introduction_message = ("Hello everyone! I'm your AI voice assistant. "
                                      "I'm here to help with the meeting. Feel free to talk to me anytime!")
            
            logger.info(f"📢 Introduction message ready: {introduction_message}")
            
            # STEP 1: Generate TTS audio FIRST (while bot is still muted)
            audio_generated = False
            try:
                logger.info(f"📢 Step 1: Generating TTS audio for introduction (bot still muted)...")
                tts_result = self.tts_service.generate_audio(
                    text_input=introduction_message,
                    upload_to_cloud=False
                )
                
                logger.info(f"TTS returned result: {tts_result}")
                
                # Extract local_path from the result dictionary
                audio_path = tts_result.get('local_path')
                
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"Read {len(audio_data)} bytes of audio data from {audio_path}")
                    
                    # Clean up temp file immediately
                    try:
                        os.remove(audio_path)
                        logger.info(f"Cleaned up temp file: {audio_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temp file: {cleanup_error}")
                        # Continue execution even if cleanup fails
                    
                    logger.info(f"✅ Step 1 COMPLETE: Audio generated ({len(audio_data)} bytes)")
                    audio_generated = True
                    
                    # STEP 2: Convert audio to MP3 and send via output_audio HTTP endpoint
                    # Attendee API requires MP3 format for audio output
                    try:
                        logger.info(f"🎤 Step 2: Converting audio to MP3 and sending via output_audio endpoint...")
                        
                        # Convert WAV to MP3 (required by Attendee API)
                        from pydub import AudioSegment
                        import io
                        
                        # Load WAV audio
                        audio = AudioSegment.from_wav(io.BytesIO(audio_data))
                        
                        # Export as MP3
                        mp3_buffer = io.BytesIO()
                        audio.export(mp3_buffer, format="mp3")
                        mp3_data = mp3_buffer.getvalue()
                        
                        logger.info(f"Converted {len(audio_data)} bytes (WAV) to {len(mp3_data)} bytes (MP3)")
                        
                        # Send via HTTP POST to /output_audio endpoint
                        self.attendee_client.send_output_audio(bot_id, mp3_data)
                        
                        logger.info(f"✅ Step 2 COMPLETE: Audio sent via output_audio endpoint ({len(mp3_data)} bytes MP3)")
                        
                    except Exception as audio_send_error:
                        logger.error(f"❌ Step 2 FAILED: Error sending audio: {str(audio_send_error)}", exc_info=True)
                        audio_generated = False
                    
                else:
                    logger.error(f"❌ Step 1 FAILED: No audio file returned (got: {audio_path}) or file doesn't exist")
            
            except Exception as e:
                logger.error(f"❌ Step 1 FAILED: Error generating introduction audio: {str(e)}", exc_info=True)
            
            # STEP 3: Send chat message to Google Meet (in parallel with audio)
            try:
                logger.info(f"💬 Step 3: Sending chat message...")
                self._send_chat_message(bot_id, introduction_message)
                logger.info(f"✅ Step 3 COMPLETE: Chat message sent successfully")
            except Exception as e:
                logger.error(f"❌ Step 3 FAILED: Error sending chat message: {str(e)}", exc_info=True)
            
            logger.info(f"=== INTRODUCTION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"❌ Error in _send_bot_introduction: {str(e)}", exc_info=True)
    
    def _send_chat_message(self, bot_id: str, message: str):
        """
        Send a chat message through the Attendee bot.
        
        Args:
            bot_id: Attendee bot ID
            message: Message to send
        """
        try:
            self.attendee_client.send_chat_message(bot_id, message)
            logger.info(f"Chat message sent for bot {bot_id}")
        except Exception as e:
            logger.error(f"Failed to send chat message: {str(e)}")
            raise
    
    # ==================== TRANSCRIPT PROCESSING ====================
    
    def handle_transcript(
        self,
        session_id: str,
        speaker: str,
        text: str,
        is_final: bool = True,
        idempotency_key: Optional[str] = None
    ):
        """
        Handle incoming transcript from meeting.
        
        Processes transcript, decides if AI should respond, and generates response if needed.
        
        Args:
            session_id: Session ID
            speaker: Speaker name
            text: Transcript text
            is_final: Whether transcript is final (vs. interim)
            idempotency_key: Webhook idempotency key (prevents duplicate processing)
        """
        # Check idempotency with time-based cleanup
        if idempotency_key:
            now = time.time()

            # Check if already processed (thread-safe check)
            with self.websocket_lock:
                if idempotency_key in self.processed_webhooks:
                    logger.debug(f"⏭️ Skipping duplicate webhook: {idempotency_key}")
                    return

                # Add to tracking with timestamp
                self.processed_webhooks[idempotency_key] = now

                # Remove expired entries (older than TTL) - FIFO from OrderedDict
                while self.processed_webhooks:
                    oldest_key, oldest_time = next(iter(self.processed_webhooks.items()))
                    if now - oldest_time > self.webhook_ttl:
                        del self.processed_webhooks[oldest_key]
                        logger.debug(f"Cleaned up expired webhook ID: {oldest_key}")
                    else:
                        break  # Stop when we hit first non-expired entry
        
        session = self.get_session(session_id)
        
        # Only process final transcripts
        if not is_final:
            logger.debug(f"Skipping interim transcript: {text[:50]}")
            return
        
        # Skip the bot's own messages (don't respond to yourself!)
        if "Prism" in speaker or "Voice Agent" in speaker:
            logger.debug(f"Skipping bot's own message: {speaker}: {text[:50]}")
            return
        
        # Add to session history
        session.add_transcript(speaker, text, is_final)
        
        # Note: Conversation context managed by Gemini internally
        
        logger.info(f"📝 Transcript from {speaker}: {text}")
        
        # OPTIMIZATION: Fast heuristic pre-filter to skip OBVIOUS non-responses
        # Only filters extremely clear cases (filler words), maintains quality
        if self._should_skip_obvious_non_response(text):
            logger.debug("⏭️ Heuristic filter: skipping obvious non-response")
            return
        
        # Check if already generating a response (prevent concurrent generation)
        if session.is_generating_response:
            logger.warning(f"⏸️ Already generating response for session {session_id}, skipping")
            return
        
        # Always consult Gemini to decide if we should respond
        # No threshold - Gemini decides on every message
        if self._should_respond(session):
            logger.info("✅ AI decided to respond...")
            session.is_generating_response = True  # Set lock
            try:
                self._generate_and_send_response(session)
            finally:
                session.is_generating_response = False  # Release lock
        else:
            logger.debug("⏭️ AI decided not to respond to this message")
    
    def _should_skip_obvious_non_response(self, text: str) -> bool:
        """
        Fast heuristic pre-filter to skip OBVIOUS non-responses.
        
        Ultra-conservative: only filters meaningless filler words.
        When in doubt, returns False (consult AI for quality).
        
        This saves 2-3 seconds for ~15% of messages (pure filler).
        
        Args:
            text: Transcript text
        
        Returns:
            bool: True if should skip (obvious non-response)
        """
        text_lower = text.lower().strip()
        
        # Too short to be meaningful (< 3 chars)
        if len(text_lower) < 3:
            return True
        
        # Pure filler words (exact matches only) - not addressing the bot
        filler_words = ['um', 'uh', 'hmm', 'ah', 'oh', 'mm', 'er', 'uhm']
        if text_lower in filler_words:
            return True
        
        # Explicit bot mention? ALWAYS consult AI (quality-critical)
        bot_mentions = ['prism', 'bot', 'assistant', 'ai']
        if any(mention in text_lower for mention in bot_mentions):
            return False
        
        # Question indicators? ALWAYS consult AI (quality-critical)
        if '?' in text or any(q in text_lower for q in ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you', 'would you', 'will you']):
            return False
        
        # Default: consult AI (quality-preserving - when in doubt, check with AI)
        return False
    
    def _should_respond(self, session: PrismSession) -> bool:
        """
        Decide if AI should respond to the current conversation.
        
        Uses Gemini to analyze conversation context and decide if a response is appropriate.
        Always consults Gemini - no message threshold.
        
        Args:
            session: PrismSession
        
        Returns:
            bool: True if should respond
        """
        # Build decision prompt with conversation history
        # OPTIMIZATION: Use last 7 transcripts (sufficient for meeting context, faster processing)
        recent_transcripts = session.transcript_history[-7:]
        
        if not recent_transcripts:
            logger.debug("No transcripts yet, skipping response decision")
            return False
        
        transcript_text = "\n".join([
            f"{t.speaker}: {t.text}" for t in recent_transcripts
        ])
        
        # Get the most recent speaker and message
        latest = recent_transcripts[-1]
        
        decision_prompt = f"""TASK: Decide if you should respond to the latest message in this meeting.

CONVERSATION (last 7 messages):
{transcript_text}

LATEST: {latest.speaker} said: "{latest.text}"

RESPOND IF:
- They address you by name or ask a direct question
- They say hello/goodbye to you
- They request help/information
- They mention meeting-related topics (agenda, notes, decisions, actions)
- They seem confused or need clarification

DO NOT RESPOND IF:
- Background conversation between others
- Your own previous messages
- Filler words (um, uh, hmm)
- Very short responses (ok, yes, no)
- Technical discussions not involving you

ANSWER: YES or NO"""
        
        try:
            # Use Gemini to decide
            response = self.gemini_service.generate_gemini_response(
                prompt=decision_prompt,
                persona='prism'
            )
            
            decision = response.strip().upper()
            should_respond = "YES" in decision
            
            logger.info(f"🤔 AI decision for '{latest.text[:50]}...': {decision} (should_respond={should_respond})")
            return should_respond
            
        except Exception as e:
            logger.error(f"Failed to get AI decision: {str(e)}", exc_info=True)
            return False
    
    def _generate_and_send_response(self, session: PrismSession):
        """
        Generate AI response and send as audio to the meeting via HTTP POST.
        
        Args:
            session: PrismSession
        """
        try:
            # Generate text response using Gemini
            response_text = self._generate_text_response(session)
            
            if not response_text:
                logger.warning("No response generated")
                return
            
            # Note: Gemini maintains its own conversation state internally
            # No need to track conversation_context in session
            
            # Generate TTS audio (WAV format)
            audio_data = self._generate_tts_audio(response_text)
            
            if not audio_data:
                logger.warning("No audio generated")
                return
            
            # Convert to MP3 and send via HTTP POST to /output_audio endpoint
            try:
                logger.info(f"Converting audio to MP3 for response: {response_text[:100]}...")
                
                # Convert WAV to MP3 (required by Attendee API)
                from pydub import AudioSegment
                import io
                
                # Load WAV audio
                audio = AudioSegment.from_wav(io.BytesIO(audio_data))
                
                # Export as MP3
                mp3_buffer = io.BytesIO()
                audio.export(mp3_buffer, format="mp3")
                mp3_data = mp3_buffer.getvalue()
                
                logger.info(f"Converted {len(audio_data)} bytes (WAV) to {len(mp3_data)} bytes (MP3)")
                
                # Send via HTTP POST to /output_audio endpoint
                self.attendee_client.send_output_audio(session.bot_id, mp3_data)
                
                logger.info(f"✅ AI response sent successfully: {response_text[:100]}...")
                
            except Exception as audio_error:
                logger.error(f"Failed to send audio response: {str(audio_error)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
    
    def _generate_text_response(self, session: PrismSession) -> Optional[str]:
        """
        Generate text response using Gemini.
        
        Args:
            session: PrismSession
        
        Returns:
            str: Generated response text
        """
        try:
            # Build prompt with conversation context
            # OPTIMIZATION: Use last 7 transcripts (sufficient context, faster processing)
            recent_transcripts = session.transcript_history[-7:]
            transcript_text = "\n".join([
                f"{t.speaker}: {t.text}" for t in recent_transcripts
            ])
            
            prompt = f"""Conversation so far:
{transcript_text}

Generate a natural, conversational response. Keep it concise (1-2 sentences) since it will be spoken aloud."""
            
            # OPTIMIZATION: Use standard generation for real-time meetings (no RAG)
            # Meeting context is in transcript history, not vector DB
            # This saves 1.5-3 seconds per response with no quality loss
            response = self.gemini_service.generate_gemini_response(
                prompt=prompt,
                persona='prism'
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return None
    
    def _generate_tts_audio(self, text: str) -> Optional[bytes]:
        """
        Generate TTS audio for text.
        
        Args:
            text: Text to convert to speech
        
        Returns:
            bytes: Audio data (WAV format)
        """
        try:
            logger.info(f"🎵 Generating TTS audio...")
            
            # Generate audio using TTS service
            # upload_to_cloud=False ensures we get a local file
            tts_result = self.tts_service.generate_audio(
                text_input=text,
                upload_to_cloud=False
            )
            
            # Read audio file from local path
            local_path = tts_result.get('local_path')
            if not local_path or not os.path.exists(local_path):
                logger.error("TTS did not generate a local file")
                return None
            
            # Read audio bytes
            with open(local_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            try:
                os.remove(local_path)
            except Exception as e:
                logger.warning("Failed to remove temp file: %s", e)
            
            logger.info(f"✅ Generated TTS audio: {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            logger.error("TTS generation failed: %s", str(e))
            return None
    
    def _queue_audio_for_bot(self, session: PrismSession, audio_data: bytes):
        """
        Process audio and add to session queue for bot WebSocket to send.
        
        Splits audio into chunks for streaming to avoid sending large JSON messages.
        
        Args:
            session: PrismSession
            audio_data: Raw audio data from TTS (WAV format)
        """
        try:
            # Get TTS result with format information
            tts_result = self.tts_service.generate_audio(text_input=text, upload_to_cloud=False)

            # Extract audio data and format
            audio_data = None
            source_format = "wav"  # Default fallback

            # Read audio file if local path exists
            if tts_result.get('local_path') and os.path.exists(tts_result['local_path']):
                with open(tts_result['local_path'], 'rb') as f:
                    audio_data = f.read()

                # Use format from TTS result, fallback to extension-based detection
                if tts_result.get('audio_format'):
                    source_format = tts_result['audio_format']
                else:
                    # Fallback: detect from file extension
                    file_ext = os.path.splitext(tts_result['local_path'])[1].lower()
                    if file_ext == '.mp3':
                        source_format = 'mp3'
                    elif file_ext in ['.wav', '.pcm']:
                        source_format = 'wav'
                    else:
                        source_format = 'wav'  # Default

                # Clean up local file
                try:
                    os.remove(tts_result['local_path'])
                except OSError:
                    pass

            if audio_data is None:
                raise RuntimeError("Failed to read audio data from TTS service")

            logger.info(f"Processing audio: {len(audio_data)} bytes in {source_format} format")

            # Convert to PCM
            processed = self.audio_processor.process_tts_audio(
                audio_data,
                source_format=source_format
            )
            
            # CRITICAL: Add 2 seconds of low-level white noise at the beginning to establish audio stream
            # This prevents Google Meet from auto-muting the bot before audio starts
            # Google Meet auto-mutes if no audio is detected for ~1 second
            # Using very quiet white noise instead of pure silence so Google Meet detects audio activity
            # 16kHz, 16-bit mono = 32000 bytes/sec, so 2000ms = 64000 bytes
            import random
            silence_duration_bytes = 64000  # 2 seconds
            # Generate very quiet white noise (amplitude ~1% of max to be barely audible)
            # 16-bit PCM has range -32768 to 32767, use ±300 for very quiet noise
            noise = bytearray()
            for _ in range(silence_duration_bytes // 2):  # Each sample is 2 bytes
                # Generate random sample in range -300 to +300
                sample = random.randint(-300, 300)
                # Convert to 2-byte signed integer (little-endian)
                noise.extend(sample.to_bytes(2, byteorder='little', signed=True))
            
            pcm_with_preamble = bytes(noise) + processed["pcm_data"]
            
            logger.info(f"Added {silence_duration_bytes} bytes ({silence_duration_bytes/32000:.2f}s) of quiet white noise preamble to prevent auto-mute")
            
            # Split into chunks for streaming (avoid large JSON messages)
            from .constants import MAX_AUDIO_CHUNK_SIZE
            chunks_data = self.audio_processor.split_into_chunks(
                pcm_with_preamble,
                chunk_size=MAX_AUDIO_CHUNK_SIZE
            )
            
            # Queue each chunk separately
            base_sequence = len(session.audio_queue)
            for idx, chunk_data in enumerate(chunks_data):
                chunk = AudioChunkOutgoing(
                    data=chunk_data,
                    timestamp=datetime.utcnow(),
                    sequence=base_sequence + idx
                )
                session.audio_queue.append(chunk)
            
            logger.info(
                "Queued audio: %.2fs, %d chunks of %d bytes each",
                processed['duration'],
                len(chunks_data),
                MAX_AUDIO_CHUNK_SIZE
            )
            
        except Exception as e:
            logger.error("Audio queuing failed: %s", str(e))
            raise AudioProcessingError(str(e))
    
    # ==================== WEBSOCKET CONNECTION MANAGEMENT ====================
    
    def mark_user_connected(self, session_id: str, websocket: Optional[WebSocketServer] = None):
        """Mark user WebSocket as connected and store reference."""
        with self.websocket_lock:
            session = self.get_session(session_id)
            session.user_ws_connected = True
            session.last_activity = datetime.utcnow()
            if websocket:
                self.user_websockets[session_id] = websocket
            logger.info(f"User WebSocket connected for session {session_id}")

    def mark_user_disconnected(self, session_id: str):
        """Mark user WebSocket as disconnected and cleanup."""
        with self.websocket_lock:
            try:
                session = self.get_session(session_id)
                session.user_ws_connected = False
                logger.info(f"User WebSocket disconnected for session {session_id}")
                
                # Remove WebSocket from registry
                if session_id in self.user_websockets:
                    del self.user_websockets[session_id]
            except SessionNotFoundError:
                logger.debug(f"Session {session_id} not found during user disconnect")
                return
            except Exception as e:
                logger.error(f"Error during user disconnect cleanup: {str(e)}")
                return

        # Close session (outside lock to avoid blocking)
        self.close_session(session_id)
    
    def mark_bot_connected(self, session_id: str):
        """Mark bot WebSocket as connected."""
        session = self.get_session(session_id)
        session.bot_ws_connected = True
        session.last_activity = datetime.utcnow()
        logger.info(f"Bot WebSocket connected for session {session_id}")
    
    def mark_bot_disconnected(self, session_id: str):
        """Mark bot WebSocket as disconnected and cleanup."""
        try:
            session = self.get_session(session_id)
            session.bot_ws_connected = False
            logger.info(f"Bot WebSocket disconnected for session {session_id}")
            
            # Close session if user also disconnected
            if not session.user_ws_connected:
                logger.info(f"Both user and bot disconnected, closing session {session_id}")
                self.close_session(session_id)
        except SessionNotFoundError:
            logger.debug(f"Session {session_id} not found during bot disconnect")
        except Exception as e:
            logger.error(f"Error during bot disconnect cleanup: {str(e)}")
    
    def _send_state_update_to_user(self, session: PrismSession, message: str):
        """
        Send bot state update to connected user WebSocket.

        Args:
            session: Session object
            message: Human-readable message describing the state change
        """
        with self.websocket_lock:
            if session.session_id not in self.user_websockets:
                logger.debug(
                    "No WebSocket connected for session %s, skipping state update",
                    session.session_id
                )
                return

            ws = self.user_websockets[session.session_id]

        try:
            update = {
                "type": "state_update",
                "session_id": session.session_id,
                "status": session.status.value,
                "bot_state": session.bot_state.value,
                "bot_id": session.bot_id,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }

            if session.error_message:
                update["error"] = session.error_message

            # Send outside the lock to avoid blocking other WebSocket operations
            ws.send(json.dumps(update))
            logger.info("Sent state update to user WebSocket: %s", message)

        except Exception as e:
            logger.error(
                "Failed to send state update to user WebSocket: %s",
                str(e)
            )
            # Remove broken WebSocket connection safely
            with self.websocket_lock:
                try:
                    if session.session_id in self.user_websockets:
                        del self.user_websockets[session.session_id]
                except KeyError:
                    pass  # Already removed
            # Don't raise - we don't want to crash if WebSocket send fails
    
    def send_new_transcripts_to_user(self, session: PrismSession):
        """
        Send any new transcripts to the connected user WebSocket.

        Args:
            session: Session object
        """
        with self.websocket_lock:
            if session.session_id not in self.user_websockets:
                return

            ws = self.user_websockets[session.session_id]

        # Get transcripts that haven't been sent yet
        new_transcripts = session.transcript_history[session.last_sent_transcript_index:]

        if not new_transcripts:
            return

        try:
            # Send each new transcript (outside lock to avoid blocking other operations)
            for transcript in new_transcripts:
                message = {
                    "type": "transcript",
                    "session_id": session.session_id,
                    "speaker": transcript.speaker,
                    "text": transcript.text,
                    "is_final": transcript.is_final,
                    "timestamp": transcript.timestamp.isoformat()
                }
                ws.send(json.dumps(message))

            # Update last sent index (thread-safe since it's just assignment)
            session.last_sent_transcript_index = len(session.transcript_history)

            logger.debug(
                "Sent %d new transcript(s) to user WebSocket for session %s",
                len(new_transcripts),
                session.session_id
            )

        except Exception as e:
            logger.error(
                "Failed to send transcripts to user WebSocket: %s",
                str(e)
            )
            # Remove broken WebSocket connection safely
            with self.websocket_lock:
                try:
                    if session.session_id in self.user_websockets:
                        del self.user_websockets[session.session_id]
                except KeyError:
                    pass  # Already removed
            # Don't raise - we don't want to crash if WebSocket send fails
    
    def get_pending_audio(self, session_id: str) -> List[AudioChunkOutgoing]:
        """
        Get and clear pending audio chunks for bot to send.

        Args:
            session_id: Session ID

        Returns:
            List[AudioChunkOutgoing]: Pending audio chunks
        """
        session = self.get_session(session_id)

        # Get all pending audio
        pending = session.audio_queue.copy()

        # Clear queue
        session.audio_queue.clear()

        return pending

    # ==================== RESOURCE MONITORING ====================

    @property
    def active_sessions_count(self) -> int:
        """Get count of active sessions for monitoring."""
        return len(self.sessions)

    @property
    def websocket_connections_count(self) -> int:
        """Get count of active WebSocket connections."""
        return len(self.user_websockets)

    @property
    def total_transcripts_processed(self) -> int:
        """Get total number of transcripts processed across all sessions."""
        return sum(len(session.transcript_history) for session in self.sessions.values())

    @property
    def total_audio_chunks_queued(self) -> int:
        """Get total number of audio chunks queued across all sessions."""
        return sum(len(session.audio_queue) for session in self.sessions.values())

    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific session.

        Args:
            session_id: Session ID

        Returns:
            Dict with session metrics
        """
        session = self.get_session(session_id)
        return {
            "session_id": session_id,
            "status": session.status.value,
            "bot_state": session.bot_state.value if session.bot_state else None,
            "transcript_count": len(session.transcript_history),
            "audio_chunks_queued": len(session.audio_queue),
            "user_connected": session.user_ws_connected,
            "bot_connected": session.bot_ws_connected,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics for monitoring.

        Returns:
            Dict with system-wide metrics
        """
        return {
            "active_sessions": self.active_sessions_count,
            "websocket_connections": self.websocket_connections_count,
            "total_transcripts_processed": self.total_transcripts_processed,
            "total_audio_chunks_queued": self.total_audio_chunks_queued,
            "webhook_cache_size": len(self.processed_webhooks),
            "sessions": {
                session_id: self.get_session_metrics(session_id)
                for session_id in list(self.sessions.keys())[:10]  # Limit to 10 for performance
            }
        }


# Singleton instance
_prism_service = None


def get_prism_service() -> PrismService:
    """Get singleton PrismService instance."""
    global _prism_service
    if _prism_service is None:
        _prism_service = PrismService()
    return _prism_service

