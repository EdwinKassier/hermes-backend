from functools import lru_cache
import os
from typing import Optional
from app.services.GeminiService import GeminiService
from app.services.TTSService import TTSService
from app.services.CloudStorageService import CloudStorageService
from app.services.EmbeddingCacheService import EmbeddingCacheService

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
# Environment variable for GCS bucket name
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'default-bucket-name')
# Environment variable for GCS credentials path (optional)
GCS_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

@lru_cache(maxsize=1)
def get_gemini_service():
    """
    Lazy load and cache the GeminiService instance
    """
    return GeminiService()

@lru_cache(maxsize=1)
def get_cloud_storage_config():
    """
    Cache the cloud storage configuration to avoid repeated environment variable lookups
    """
    return {
        'bucket_name': GCS_BUCKET_NAME,
        'credentials_path': GCS_CREDENTIALS_PATH
    }

@lru_cache(maxsize=1)
def get_tts_service(device: Optional[str] = None) -> TTSService:
    """
    Lazy load and cache the TTSService instance with CloudStorageService
    
    Args:
        device: The device to load the model on (e.g., 'cuda', 'cpu', 'mps').
                If None, it will automatically detect the best available device.
    """
    # Get the cloud storage service
    cloud_storage_service = get_cloud_storage_service()
    
    # Get cached cloud storage config
    cloud_storage_config = get_cloud_storage_config()
    
    # Initialize TTSService with the cloud storage config
    return TTSService(
        tts_provider="google",  # or "chatterbox" if you want to use the local model
        device=device,
        cloud_storage_config=cloud_storage_config
    )

@lru_cache(maxsize=1)
def get_cloud_storage_service() -> CloudStorageService:
    """
    Lazy load and cache the CloudStorageService instance
    """
    return CloudStorageService(
        bucket_name=GCS_BUCKET_NAME,
        credentials_path=GCS_CREDENTIALS_PATH
    )

@lru_cache(maxsize=1)
def get_embedding_cache_service() -> EmbeddingCacheService:
    """
    Lazy load and cache the EmbeddingCacheService instance
    """
    return EmbeddingCacheService()