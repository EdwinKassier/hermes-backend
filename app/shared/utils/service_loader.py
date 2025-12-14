import os
from functools import lru_cache
from typing import Optional

from app.shared.services.AsyncLLMService import AsyncLLMService
from app.shared.services.CloudStorageService import CloudStorageService
from app.shared.services.GeminiService import GeminiService
from app.shared.services.TTSService import TTSService

# Set credentials path only if file exists (for local development)
# In Cloud Run, credentials come from the service account automatically
LOCAL_CREDENTIALS_PATH = "./credentials.json"
if os.path.exists(LOCAL_CREDENTIALS_PATH) and not os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS"
):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = LOCAL_CREDENTIALS_PATH

# Environment variables
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ashes_project_website_artifacts")
GCS_CREDENTIALS_PATH = (
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    and os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
    else None
)
# TTS provider: 'elevenlabs' or 'google'
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")
EL_API_KEY = os.getenv("EL_API_KEY")  # ElevenLabs API key


@lru_cache(maxsize=1)
def get_gemini_service():
    """Lazy load and cache the GeminiService instance."""
    service = GeminiService()

    # Register Legion personas with the service
    try:
        from app.hermes.legion.utils.persona_generator import LegionPersonaProvider

        legion_personas = LegionPersonaProvider.get_legion_personas()
        service.add_personas(legion_personas)
        import logging

        logging.info(
            f"Successfully registered {len(legion_personas)} Legion personas with GeminiService"
        )
    except Exception as e:
        # Log error but don't fail if Legion personas can't be loaded
        import logging

        logging.error(f"Failed to load Legion personas: {e}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")

    return service


@lru_cache(maxsize=1)
def get_async_llm_service() -> AsyncLLMService:
    """
    Lazy load and cache the AsyncLLMService instance.

    This wraps the synchronous GeminiService in an async interface
    to prevent blocking the event loop in async contexts.

    Returns:
        AsyncLLMService instance wrapping the cached GeminiService
    """
    return AsyncLLMService(gemini_service=get_gemini_service())


@lru_cache(maxsize=1)
def get_cloud_storage_config():
    """Cache the cloud storage configuration."""
    return {"bucket_name": GCS_BUCKET_NAME, "credentials_path": GCS_CREDENTIALS_PATH}


@lru_cache(maxsize=1)
def get_tts_service() -> TTSService:
    """Lazy load and cache the TTSService instance.

    Returns:
        TTSService instance configured based on TTS_PROVIDER env var
    """
    cloud_storage_config = get_cloud_storage_config()

    return TTSService(
        tts_provider=TTS_PROVIDER,
        cloud_storage_config=cloud_storage_config,
        elevenlabs_api_key=EL_API_KEY,
    )


@lru_cache(maxsize=1)
def get_cloud_storage_service() -> CloudStorageService:
    """Lazy load and cache the CloudStorageService instance."""
    return CloudStorageService(
        bucket_name=GCS_BUCKET_NAME, credentials_path=GCS_CREDENTIALS_PATH
    )


@lru_cache(maxsize=1)
def get_supabase_database_service():
    """Lazy load and cache the Supabase Database Service instance."""
    from app.shared.services.SupabaseDatabaseService import (
        get_supabase_database_service,
    )

    return get_supabase_database_service()
