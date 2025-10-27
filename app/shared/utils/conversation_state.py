"""Generic State Management Utility.

This module provides a service for managing application states in a SQLite
database. It handles serialization/deserialization of states and provides
methods for state persistence and retrieval.
"""

import json
import logging
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar, Iterator
from dataclasses import asdict, is_dataclass

# Import LangChain message types
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    # Fallback for when LangChain is not available
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
    
    class AIMessage:
        def __init__(self, content: str):
            self.content = content
    
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for state type
T = TypeVar('T')


class State(Generic[T]):
    """A generic state container that can hold any JSON-serializable data.

    Attributes:
        data: The actual state data
        metadata: Additional metadata about the state
    """

    def __init__(
        self,
        data: Optional[T] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a new State instance.

        Args:
            data: The initial state data
            metadata: Optional metadata dictionary
        """
        self.data = data or {}
        self.metadata = metadata or {
            'version': '1.0',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }

    def update(self, **updates: Any) -> None:
        """Update the state with new values.

        Args:
            **updates: Key-value pairs to update in the state
        """
        self.data.update(updates)
        self.metadata['updated_at'] = datetime.utcnow().isoformat()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state with an optional default.

        Args:
            key: The key to get
            default: Default value if key doesn't exist

        Returns:
            The value or default if key doesn't exist
        """
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary.

        Returns:
            Dictionary representation of the state
        """
        return {
            'data': self.data,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create a State instance from a dictionary.

        Args:
            data: Dictionary containing 'data' and 'metadata' keys

        Returns:
            A new State instance
        """
        state = cls()
        state.data = data.get('data', {})
        state.metadata = data.get('metadata', {})
        return state

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Recursively serialize an object to be JSON-serializable.

        Args:
            obj: The object to serialize

        Returns:
            A JSON-serializable representation of the object
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: State._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [State._serialize(item) for item in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if is_dataclass(obj):
            return asdict(obj)
        return str(obj)

    @staticmethod
    def _deserialize(obj: Any) -> Any:
        """Recursively deserialize an object from JSON-serialized form.

        Args:
            obj: The object to deserialize

        Returns:
            The deserialized object
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: State._deserialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [State._deserialize(item) for item in obj]
        return obj


class ConversationState:
    """A service for managing conversation states with SQLite persistence.

    This class provides methods to save, retrieve, and manage conversation
    states with proper serialization and validation.
    """

    def __init__(self, db_path: str = "conversations.db") -> None:
        """Initialize the ConversationState service.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._create_table()
        self._cleanup_old_sessions()

    @contextmanager
    def _get_db_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections.

        Yields:
            A database connection

        Raises:
            sqlite3.Error: If there's an error connecting to the database
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            error_msg = "Database connection error: %s"
            logger.error(error_msg, str(e))
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _create_table(self) -> None:
        """Create the conversation_states table if it doesn't exist."""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS conversation_states (
                user_id TEXT PRIMARY KEY,
                state TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)

            # Add last_updated column if it doesn't exist
            try:
                add_column_sql = """
                    ALTER TABLE conversation_states 
                    ADD COLUMN last_updated TIMESTAMP 
                    DEFAULT CURRENT_TIMESTAMP
                """
                conn.execute(add_column_sql)
            except sqlite3.OperationalError:
                # Column already exists
                pass
    
    @staticmethod
    def validate_state(state: State) -> bool:
        """Validate the structure and content of a conversation state.
        
        Args:
            state: The state object to validate
            
        Returns:
            bool: True if the state is valid, False otherwise
        """
        if not isinstance(state, State):
            logger.error("Invalid state: not a State instance")
            return False
            
        if not isinstance(state.data, dict):
            logger.error("State data must be a dictionary")
            return False
            
        return True
    
    def get_state(self, user_id: str) -> State:
        """Retrieve the conversation state for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            State: The user's conversation state
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state FROM conversation_states WHERE user_id=?",
                    (user_id,)
                )
                result = cursor.fetchone()

                if not result:
                    return State()
                    
                state_data = json.loads(result['state'])
                state = State.from_dict(state_data)
                
                if not self.validate_state(state):
                    logger.warning(
                        "Invalid state detected for user %s, "
                        "resetting to default",
                        user_id
                    )
                    return State()
                
                return state
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(
                "Error retrieving state for user %s: %s",
                user_id,
                str(e)
            )
            return State()
    
    def save_state(self, user_id: str, state: State) -> None:
        """Save the conversation state for a user.
        
        Args:
            user_id: The ID of the user
            state: The state object to save
            
        Raises:
            ValueError: If the state is invalid
            sqlite3.Error: If there's an error saving to the database
        """
        if not self.validate_state(state):
            raise ValueError("Invalid state detected before save")

        try:
            state_dict = state.to_dict()
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO conversation_states 
                    (user_id, state, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (user_id, json.dumps(state_dict)))
                
        except (sqlite3.Error, TypeError) as e:
            logger.error(
                "Error saving conversation state for user %s: %s",
                user_id,
                str(e)
            )
            raise
    
    def clear_conversation(self, user_id: str) -> None:
        """Clear the conversation state for a user.
        
        Args:
            user_id: The ID of the user
            
        Raises:
            sqlite3.Error: If there's an error clearing the state
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM conversation_states WHERE user_id=?",
                    (user_id,)
                )
            self.cleanup_resources(user_id)
            logger.info("Conversation state cleared for user: %s", user_id)
            
        except sqlite3.Error as e:
            logger.error(
                "Error clearing conversation state for user %s: %s",
                user_id,
                str(e)
            )
            raise
    
    def cleanup_resources(self, user_id: str) -> None:
        """Clean up temporary files and resources for a user.
        
        Args:
            user_id: The ID of the user
            
        Note:
            This method logs errors but doesn't raise exceptions to prevent
            cleanup failures from affecting the main application flow.
        """
        # Clean up temporary files
        temp_dir = f"temp/{user_id}"
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                
            # Reset user state
            self.save_state(user_id, State())
                
        except (OSError, sqlite3.Error) as e:
            logger.error(
                "Error cleaning up resources for user %s: %s",
                user_id,
                str(e)
            )
    
    def _cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up old conversation sessions.
        
        Args:
            max_age_hours: Maximum age in hours before a session is considered old
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM conversation_states
                    WHERE datetime(last_updated) < datetime('now', ?)
                """, (f'-{max_age_hours} hours',))
                
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(
                        "Cleaned up %d old sessions (older than %d hours)",
                        deleted_count,
                        max_age_hours
                    )
                    
        except sqlite3.Error as e:
            logger.error(
                "Error cleaning up old sessions: %s",
                str(e)
            )
    
    @staticmethod
    def _serialize_message(message) -> dict:
        """
        Serialize a message object to a dictionary.
        
        Args:
            message: The message to serialize (HumanMessage, AIMessage, or SystemMessage)
            
        Returns:
            dict: The serialized message
        """
        if isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"type": "system", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"type": "ai", "content": message.content}
        return message
    
    @staticmethod
    def _deserialize_message(message_data: dict):
        """
        Deserialize a message dictionary back to a message object.
        
        Args:
            message_data: The serialized message data
            
        Returns:
            The deserialized message object
        """
        if isinstance(message_data, dict) and "type" in message_data and "content" in message_data:
            if message_data["type"] == "human":
                return HumanMessage(content=message_data["content"])
            elif message_data["type"] == "system":
                return SystemMessage(content=message_data["content"])
            elif message_data["type"] == "ai":
                return AIMessage(content=message_data["content"])
        return message_data
