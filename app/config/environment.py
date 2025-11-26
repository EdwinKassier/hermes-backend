from functools import lru_cache
from os import environ

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_environment():
    """Load and cache environment variables"""
    # Load .env file only once
    load_dotenv()

    return {
        "APP_NAME": environ.get("APP_NAME") or "flask-boilerplate",
        "APPLICATION_ENV": environ.get("APPLICATION_ENV") or "development",
        "API_KEY": environ.get("API_KEY"),
        "GOOGLE_API_KEY": environ.get("GOOGLE_API_KEY"),
        "GOOGLE_PROJECT_ID": environ.get("GOOGLE_PROJECT_ID"),
        "GOOGLE_PROJECT_LOCATION": environ.get("GOOGLE_PROJECT_LOCATION"),
        "EL_API_KEY": environ.get("EL_API_KEY"),  # ElevenLabs API key
        "PORT": int(environ.get("PORT", 8080)),
        # Supabase Configuration
        "SUPABASE_DATABASE_URL": environ.get("SUPABASE_DATABASE_URL"),
        "SUPABASE_PROJECT_URL": environ.get("SUPABASE_PROJECT_URL"),
        "SUPABASE_SERVICE_ROLE_KEY": environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        # Supabase MCP Server Configuration (legacy)
        "SUPABASE_MCP_SERVER_URL": environ.get("SUPABASE_MCP_SERVER_URL"),
        "SUPABASE_MCP_API_KEY": environ.get("SUPABASE_MCP_API_KEY"),
        # Base Prompts for different personas
        "BASE_PROMPT": environ.get(
            "BASE_PROMPT", ""
        ),  # Hermes persona (default/backward compatible)
        "PRISMA_BASE_PROMPT": environ.get("PRISMA_BASE_PROMPT", ""),
        "PRISM_BASE_PROMPT": environ.get(
            "PRISM_BASE_PROMPT", ""
        ),  # Prism voice agent persona
        # Prism Domain Configuration (Attendee Voice Agent Integration)
        "ATTENDEE_API_KEY": environ.get("ATTENDEE_API_KEY"),
        "WEBHOOK_BASE_URL": environ.get(
            "WEBHOOK_BASE_URL"
        ),  # Base URL for webhooks (e.g., https://your-domain.com)
        "WEBSOCKET_BASE_URL": environ.get(
            "WEBSOCKET_BASE_URL"
        ),  # Base URL for WebSocket (e.g., wss://your-domain.com)
    }


def get_env(key: str, default: str = "") -> str:
    """Get environment variable by key"""
    value = load_environment().get(key)
    return value if value is not None else default


def validate_mcp_config() -> None:
    """
    Validate Supabase configuration.

    Raises:
        ValueError: If required Supabase configuration is missing
    """
    import os

    supabase_url = os.getenv("SUPABASE_PROJECT_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase configuration incomplete. Required: "
            "SUPABASE_PROJECT_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
