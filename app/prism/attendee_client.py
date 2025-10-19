"""
Attendee API Client - Handles all interactions with Attendee API

Based on official Attendee documentation:
- Base URL: https://api.attendee.dev
- Authentication: Bearer token via ATTENDEE_API_KEY
- Main endpoint: POST /bots (create bot with meeting URL)
"""
import os
import logging
import requests
from typing import Optional, Dict, Any

from .constants import (
    ATTENDEE_API_BASE_URL,
    ATTENDEE_API_VERSION,
    AUDIO_SAMPLE_RATE,
    AUDIO_ENCODING
)
from .exceptions import AttendeeAPIError, BotCreationError
from .schemas import CreateBotResponse

logger = logging.getLogger(__name__)


class AttendeeClient:
    """
    Client for Attendee API interactions.
    
    Handles:
    - Bot creation with WebSocket configuration
    - Bot status retrieval
    - Bot deletion
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Attendee client.
        
        Args:
            api_key: Attendee API key (defaults to ATTENDEE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ATTENDEE_API_KEY")
        if not self.api_key:
            raise ValueError("ATTENDEE_API_KEY not found in environment")
        
        self.base_url = f"{ATTENDEE_API_BASE_URL}/{ATTENDEE_API_VERSION}"
        
        # Use requests.Session() for HTTP connection pooling
        # This reuses TCP connections, saving 50-150ms per request
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        })
        
        self.headers = self.session.headers  # For backward compatibility
    
    def create_bot(
        self,
        meeting_url: str,
        bot_name: str,
        webhook_url: str,
        audio_websocket_url: str
    ) -> CreateBotResponse:
        """
        Create an Attendee bot for a Google Meet session.
        
        Per Attendee documentation, creates bot with:
        - Closed caption transcription (Google Meet native captions)
        - WebSocket audio output (for voice responses)
        - Webhook for state changes and transcripts
        
        Args:
            meeting_url: Google Meet URL
            bot_name: Display name for bot
            webhook_url: Your webhook URL for callbacks
            audio_websocket_url: Your WebSocket URL for audio streaming
        
        Returns:
            CreateBotResponse: Bot ID and WebSocket URL
        
        Raises:
            BotCreationError: If bot creation fails
        """
        endpoint = f"{self.base_url}/bots"
        
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "webhooks": [
                {
                    "url": webhook_url,
                    "triggers": ["bot.state_change", "transcript.update"]
                }
            ],
            "transcription_settings": {
                "meeting_closed_captions": {}
            },
            "websocket_settings": {
                "audio": {
                    "url": audio_websocket_url,
                    "sample_rate": AUDIO_SAMPLE_RATE
                }
            }
        }
        
        try:
            logger.info(f"Creating Attendee bot for meeting: {meeting_url}")
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code not in (200, 201):
                error_msg = self._parse_error(response)
                logger.error(f"Bot creation failed (HTTP {response.status_code}): {error_msg}")
                logger.error(f"Response body: {response.text[:500]}")  # Log first 500 chars
                raise BotCreationError(error_msg)
            
            data = response.json()
            logger.info(f"Bot creation response: {data}")
            
            # Attendee API may return id instead of bot_id
            bot_id = data.get("id") or data.get("bot_id")
            
            if not bot_id:
                logger.error(f"Bot API response missing ID. Full response: {data}")
                raise BotCreationError(f"Invalid API response: missing bot ID. Response keys: {list(data.keys())}")
            
            logger.info(f"Bot created successfully with ID: {bot_id}")
            
            return CreateBotResponse(
                bot_id=bot_id,
                status=data.get("status", "created"),
                meeting_url=meeting_url,
                websocket_url=data.get("websocket_url")
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Attendee API failed: {str(e)}")
            raise BotCreationError(f"Network error: {str(e)}")
        except KeyError as e:
            logger.error(f"Invalid response format: {str(e)}")
            raise BotCreationError(f"Invalid API response: missing {str(e)}")
    
    def get_bot_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Get current status of a bot.
        
        Args:
            bot_id: Attendee bot ID
        
        Returns:
            dict: Bot status information
        
        Raises:
            AttendeeAPIError: If request fails
        """
        endpoint = f"{self.base_url}/bots/{bot_id}"
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                error_msg = self._parse_error(response)
                raise AttendeeAPIError(error_msg, response.status_code)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get bot status: {str(e)}")
            raise AttendeeAPIError(f"Network error: {str(e)}")
    
    def unmute_bot(self, bot_id: str) -> bool:
        """
        Unmute the bot's microphone in the meeting.
        
        Args:
            bot_id: Attendee bot ID
            
        Returns:
            bool: True if successful
            
        Raises:
            AttendeeAPIError: If unmute fails
        """
        endpoint = f"{self.base_url}/bots/{bot_id}/unmute"
        
        try:
            logger.info(f"Unmuting bot {bot_id}")
            response = requests.post(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code not in (200, 201):
                error_msg = self._parse_error(response)
                logger.error(f"Bot unmute failed (HTTP {response.status_code}): {error_msg}")
                raise AttendeeAPIError(error_msg)
            
            logger.info(f"Bot {bot_id} unmuted successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Attendee API failed: {str(e)}")
            raise AttendeeAPIError(f"Network error: {str(e)}")
    
    def send_chat_message(self, bot_id: str, message: str) -> bool:
        """
        Send a chat message through the bot to the meeting.
        
        Args:
            bot_id: Attendee bot ID
            message: Message to send
            
        Returns:
            bool: True if successful
            
        Raises:
            AttendeeAPIError: If message sending fails
        """
        endpoint = f"{self.base_url}/bots/{bot_id}/send_chat_message"
        
        payload = {
            "message": message
        }
        
        try:
            logger.info(f"Sending chat message for bot {bot_id}")
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=10
            )
            
            if response.status_code not in (200, 201):
                error_msg = self._parse_error(response)
                logger.error(f"Chat message failed (HTTP {response.status_code}): {error_msg}")
                raise AttendeeAPIError(error_msg)
            
            logger.info(f"Chat message sent successfully for bot {bot_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Attendee API failed: {str(e)}")
            raise AttendeeAPIError(f"Network error: {str(e)}")
    
    def send_output_audio(self, bot_id: str, audio_data: bytes, content_type: str = "audio/wav") -> bool:
        """
        Send audio output to the bot to be played in the meeting.
        
        Uses the /output_audio endpoint as per Attendee API best practices.
        This is the CORRECT way to send audio to Google Meet via Attendee.
        
        Args:
            bot_id: Attendee bot ID
            audio_data: Audio data (WAV or PCM format)
            content_type: MIME type for audio (default: "audio/wav")
        
        Returns:
            bool: True if successful
        
        Raises:
            AttendeeAPIError: If audio sending fails
        """
        endpoint = f"{self.base_url}/bots/{bot_id}/output_audio"
        
        try:
            logger.info(f"Sending {len(audio_data)} bytes of audio to bot {bot_id} via output_audio endpoint")
            
            # Attendee API expects JSON with base64-encoded MP3 audio
            # Payload format: {"data": "base64_encoded_mp3", "type": "audio/mp3"}
            import base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            payload = {
                "data": audio_base64,
                "type": "audio/mp3"
            }
            
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=30  # Longer timeout for audio upload
            )
            
            if response.status_code not in (200, 201, 202):
                error_msg = self._parse_error(response)
                logger.error(f"Audio output failed (HTTP {response.status_code}): {error_msg}")
                raise AttendeeAPIError(error_msg)
            
            logger.info(f"✅ Audio sent successfully to bot {bot_id} ({len(audio_data)} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to output_audio endpoint failed: {str(e)}")
            raise AttendeeAPIError(f"Network error: {str(e)}")
    
    def delete_bot(self, bot_id: str) -> bool:
        """
        Delete/stop a bot (removes from meeting).
        
        Args:
            bot_id: Attendee bot ID
        
        Returns:
            bool: True if successful
        
        Raises:
            AttendeeAPIError: If request fails
        """
        endpoint = f"{self.base_url}/bots/{bot_id}"
        
        try:
            logger.info(f"Deleting bot: {bot_id}")
            response = self.session.delete(
                endpoint,
                timeout=10
            )
            
            if response.status_code not in [200, 204]:
                error_msg = self._parse_error(response)
                logger.warning(f"Bot deletion failed: {error_msg}")
                return False
            
            logger.info(f"Bot deleted successfully: {bot_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete bot: {str(e)}")
            return False
    
    def _parse_error(self, response: requests.Response) -> str:
        """Parse error message from API response."""
        try:
            error_data = response.json()
            return error_data.get(
                "message",
                error_data.get("error", str(error_data))
            )
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"

