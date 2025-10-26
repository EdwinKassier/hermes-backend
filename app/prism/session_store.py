"""Redis-backed session store for Prism to enable multi-worker support."""
import json
import logging
import os
from typing import Optional, Dict
from datetime import datetime
from dataclasses import asdict
import redis

from .models import PrismSession, BotState, SessionStatus, TranscriptEntry, AudioChunkOutgoing

logger = logging.getLogger(__name__)


class RedisSessionStore:
    """Redis-backed session storage for multi-worker support."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis session store.
        
        Args:
            redis_url: Redis connection URL. If None, uses REDIS_URL env var.
        """
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _session_key(self, session_id: str) -> str:
        """Get Redis key for session data."""
        return f"prism:session:{session_id}"
    
    def _bot_mapping_key(self, bot_id: str) -> str:
        """Get Redis key for bot_id -> session_id mapping."""
        return f"prism:bot_mapping:{bot_id}"
    
    def save_session(self, session: PrismSession) -> None:
        """
        Save session to Redis.
        
        Args:
            session: PrismSession object to save
        """
        try:
            # Convert session to dict
            session_dict = asdict(session)
            
            # Convert datetime objects to ISO format strings
            session_dict['created_at'] = session.created_at.isoformat()
            session_dict['updated_at'] = session.updated_at.isoformat()
            session_dict['last_activity'] = session.last_activity.isoformat()
            
            # Convert enums to values
            session_dict['bot_state'] = session.bot_state.value
            session_dict['status'] = session.status.value
            
            # Serialize to JSON
            session_json = json.dumps(session_dict)
            
            # Save to Redis with 24h TTL (sessions shouldn't last longer than this)
            key = self._session_key(session.session_id)
            self.redis_client.setex(key, 86400, session_json)
            
            # If bot_id is set, create bot_id -> session_id mapping
            if session.bot_id:
                mapping_key = self._bot_mapping_key(session.bot_id)
                self.redis_client.setex(mapping_key, 86400, session.session_id)
                
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[PrismSession]:
        """
        Get session from Redis.
        
        Args:
            session_id: Session ID
            
        Returns:
            PrismSession object or None if not found
        """
        try:
            key = self._session_key(session_id)
            session_json = self.redis_client.get(key)
            
            if not session_json:
                return None
            
            # Deserialize from JSON
            session_dict = json.loads(session_json)
            
            # Convert ISO format strings back to datetime objects
            session_dict['created_at'] = datetime.fromisoformat(session_dict['created_at'])
            session_dict['updated_at'] = datetime.fromisoformat(session_dict['updated_at'])
            session_dict['last_activity'] = datetime.fromisoformat(session_dict['last_activity'])
            
            # Convert enum values back to enums
            session_dict['bot_state'] = BotState(session_dict['bot_state'])
            session_dict['status'] = SessionStatus(session_dict['status'])
            
            # Reconstruct session object
            return PrismSession(**session_dict)
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def get_session_by_bot_id(self, bot_id: str) -> Optional[PrismSession]:
        """
        Get session by bot ID.
        
        Args:
            bot_id: Bot ID
            
        Returns:
            PrismSession object or None if not found
        """
        try:
            # Get session_id from bot mapping
            mapping_key = self._bot_mapping_key(bot_id)
            session_id = self.redis_client.get(mapping_key)
            
            if not session_id:
                return None
            
            # Get session by ID
            return self.get_session(session_id)
            
        except Exception as e:
            logger.error(f"Failed to get session by bot_id {bot_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete session from Redis.
        
        Args:
            session_id: Session ID
        """
        try:
            # Get session first to find bot_id
            session = self.get_session(session_id)
            
            # Delete session
            key = self._session_key(session_id)
            self.redis_client.delete(key)
            
            # Delete bot mapping if exists
            if session and session.bot_id:
                mapping_key = self._bot_mapping_key(session.bot_id)
                self.redis_client.delete(mapping_key)
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
    
    def list_all_session_ids(self) -> list[str]:
        """
        List all active session IDs.
        
        Returns:
            List of session IDs
        """
        try:
            # Get all session keys
            pattern = "prism:session:*"
            keys = self.redis_client.keys(pattern)
            
            # Extract session IDs from keys
            session_ids = [key.replace("prism:session:", "") for key in keys]
            return session_ids
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

