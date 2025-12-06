import logging
import os
from typing import Any, Optional

try:
    from langfuse import Langfuse, get_client
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    get_client = None
    CallbackHandler = None
    LANGFUSE_AVAILABLE = False


class LangfuseConfig:
    """
    Configuration and factory for Langfuse observability integration.
    Singleton pattern to maintain a single client instance.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LangfuseConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize_client()
            self._initialized = True

    def _initialize_client(self):
        """Initialize the Langfuse client if credentials are available."""
        if not LANGFUSE_AVAILABLE:
            logging.info("Langfuse SDK not installed. Observability disabled.")
            return

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        # Support both LANGFUSE_BASE_URL and LANGFUSE_HOST for compatibility
        base_url = os.environ.get(
            "LANGFUSE_BASE_URL",
            os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        if public_key and secret_key:
            try:
                # Langfuse v3 uses singleton pattern - initialize once globally
                # This sets up the singleton client that get_client() will return
                Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=base_url,
                )
                logging.info("Langfuse observability initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Langfuse client: {e}")
        else:
            logging.info(
                "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set. "
                "Langfuse observability disabled."
            )

    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse is successfully initialized."""
        if not LANGFUSE_AVAILABLE:
            return False
        try:
            client = get_client()
            return client is not None
        except Exception:
            return False

    def get_callback_handler(self) -> Optional[Any]:
        """
        Create a LangChain CallbackHandler for tracking LLM events.

        Note: Langfuse v3 CallbackHandler doesn't accept constructor parameters.
        Trace attributes (user_id, trace_id) are set via metadata in the config dict
        when invoking the model.

        Returns:
            CallbackHandler instance or None if Langfuse is disabled
        """
        if not self.is_enabled:
            return None

        try:
            # Verify client is available
            client = get_client()
            if client is None:
                return None

            # Create handler - Langfuse v3 CallbackHandler doesn't take constructor args
            # Trace attributes are set via metadata in config dict
            return CallbackHandler()
        except Exception as e:
            logging.error(f"Failed to create Langfuse callback handler: {e}")
            return None

    def flush(self) -> None:
        """
        Flush pending Langfuse events to ensure they are sent.

        Useful for serverless environments or short-lived scripts where events
        might not be sent before the process exits. This method is safe to call
        multiple times and will gracefully handle cases where Langfuse is disabled.

        Note: In long-running applications, Langfuse automatically flushes events
        periodically, so explicit flushing is typically not necessary.
        """
        if not self.is_enabled:
            return

        try:
            client = get_client()
            if client and hasattr(client, "flush"):
                client.flush()
                logging.debug("Langfuse events flushed successfully")
        except Exception as e:
            logging.warning(f"Failed to flush Langfuse events: {e}")

    def shutdown(self) -> None:
        """
        Shutdown the Langfuse client gracefully.

        Flushes any pending events and cleans up resources. This method is safe
        to call multiple times and will gracefully handle cases where Langfuse
        is disabled.

        Note: Typically only needed during application shutdown or in test cleanup.
        """
        if not self.is_enabled:
            return

        try:
            client = get_client()
            if client and hasattr(client, "shutdown"):
                client.shutdown()
                logging.debug("Langfuse client shut down successfully")
        except Exception as e:
            logging.warning(f"Failed to shutdown Langfuse client: {e}")


# Global instance
langfuse_config = LangfuseConfig()
