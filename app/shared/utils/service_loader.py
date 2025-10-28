import os
from functools import lru_cache
from typing import Optional

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
    return GeminiService()


@lru_cache(maxsize=1)
def get_cloud_storage_config():
    """Cache the cloud storage configuration."""
    return {"bucket_name": GCS_BUCKET_NAME, "credentials_path": GCS_CREDENTIALS_PATH}


@lru_cache(maxsize=1)
def get_tts_service(device: Optional[str] = None) -> TTSService:
    """Lazy load and cache the TTSService instance.

    Args:
        device: The device to load the model on (e.g., 'cuda', 'cpu',
                'mps'). If None, automatically detect the best device.
                Currently not used by any providers.

    Returns:
        TTSService instance configured based on TTS_PROVIDER env var
    """
    cloud_storage_config = get_cloud_storage_config()

    return TTSService(
        tts_provider=TTS_PROVIDER,
        device=device,
        cloud_storage_config=cloud_storage_config,
        elevenlabs_api_key=EL_API_KEY,
    )


@lru_cache(maxsize=1)
def get_cloud_storage_service() -> CloudStorageService:
    """Lazy load and cache the CloudStorageService instance."""
    return CloudStorageService(
        bucket_name=GCS_BUCKET_NAME, credentials_path=GCS_CREDENTIALS_PATH
    )
