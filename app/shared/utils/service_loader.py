from functools import lru_cache
import os
from typing import Optional
from app.shared.services.GeminiService import GeminiService
from app.shared.services.TTSService import TTSService
from app.shared.services.CloudStorageService import CloudStorageService

# Set credentials path (centralized location)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

# Environment variables
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'default-bucket-name')
GCS_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'google')  # 'google' or 'chatterbox'


@lru_cache(maxsize=1)
def get_gemini_service():
    """Lazy load and cache the GeminiService instance."""
    return GeminiService()


@lru_cache(maxsize=1)
def get_cloud_storage_config():
    """Cache the cloud storage configuration."""
    return {
        'bucket_name': GCS_BUCKET_NAME,
        'credentials_path': GCS_CREDENTIALS_PATH
    }


@lru_cache(maxsize=1)
def get_tts_service(device: Optional[str] = None) -> TTSService:
    """Lazy load and cache the TTSService instance.
    
    Args:
        device: The device to load the model on (e.g., 'cuda', 'cpu', 'mps').
                If None, it will automatically detect the best available device.
                Only used for Chatterbox provider.
    
    Returns:
        TTSService instance configured based on TTS_PROVIDER env var
    """
    cloud_storage_config = get_cloud_storage_config()
    
    return TTSService(
        tts_provider=TTS_PROVIDER,
        device=device,
        cloud_storage_config=cloud_storage_config
    )


@lru_cache(maxsize=1)
def get_cloud_storage_service() -> CloudStorageService:
    """Lazy load and cache the CloudStorageService instance."""
    return CloudStorageService(
        bucket_name=GCS_BUCKET_NAME,
        credentials_path=GCS_CREDENTIALS_PATH
    )
