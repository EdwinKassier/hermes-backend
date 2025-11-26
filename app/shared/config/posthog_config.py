import logging
import os
from typing import Any, Dict, Optional

try:
    from posthog import Posthog
    from posthog.ai.langchain import CallbackHandler

    POSTHOG_AVAILABLE = True
except ImportError:
    Posthog = None
    CallbackHandler = None
    POSTHOG_AVAILABLE = False


class PostHogConfig:
    """
    Configuration and factory for PostHog analytics integration.
    Singleton pattern to maintain a single client instance.
    """

    _instance = None
    _client = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostHogConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize_client()
            self._initialized = True

    def _initialize_client(self):
        """Initialize the PostHog client if credentials are available."""
        if not POSTHOG_AVAILABLE:
            logging.warning("PostHog SDK not installed. Analytics disabled.")
            return

        api_key = os.environ.get("POSTHOG_API_KEY")
        host = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com")

        if api_key:
            try:
                self._client = Posthog(api_key, host=host)
                logging.info("PostHog analytics initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize PostHog client: {e}")
                self._client = None
        else:
            logging.info("POSTHOG_API_KEY not set. Analytics disabled.")

    @property
    def is_enabled(self) -> bool:
        """Check if PostHog is successfully initialized."""
        return self._client is not None and POSTHOG_AVAILABLE

    def get_callback_handler(
        self,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Create a LangChain CallbackHandler for tracking LLM events.

        Args:
            user_id: The distinct ID of the user (optional)
            trace_id: Unique ID for the conversation/trace (optional)
            properties: Additional metadata properties (optional)

        Returns:
            CallbackHandler instance or None if PostHog is disabled
        """
        if not self.is_enabled:
            return None

        try:
            return CallbackHandler(
                client=self._client,
                distinct_id=user_id,
                trace_id=trace_id,
                properties=properties or {},
            )
        except Exception as e:
            logging.error(f"Failed to create PostHog callback handler: {e}")
            return None


# Global instance
posthog_config = PostHogConfig()
