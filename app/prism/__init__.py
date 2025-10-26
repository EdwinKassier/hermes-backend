"""
Prism Domain - Attendee Voice Agent Integration

This domain handles real-time voice interactions in Google Meet using the Attendee API.
Users provide a meeting URL, and Prism creates an AI-powered voice bot that can:
- Listen to meeting transcripts (via Google Meet closed captions)
- Decide when to respond using Gemini AI
- Generate speech responses using TTS
- Play audio in the meeting via WebSocket audio streaming

Architecture:
- User WebSocket: /api/v1/prism/start-session (session management)
- Bot WebSocket: /api/v1/prism/bot-audio (bidirectional audio streaming)
- Webhook: /api/v1/prism/webhook (Attendee callbacks)
"""

from .routes import prism_bp

__all__ = ['prism_bp']

