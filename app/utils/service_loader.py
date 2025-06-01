from functools import lru_cache
import os
from typing import Optional
from app.services.GeminiService import GeminiService
from app.services.TTSService import TTSService
from app.services.CloudStorageService import CloudStorageService

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
def get_tts_service(device: Optional[str] = None) -> TTSService:
    """
    Lazy load and cache the TTSService instance
    
    Args:
        device: The device to load the model on (e.g., 'cuda', 'cpu', 'mps').
                If None, it will automatically detect the best available device.
    """
    return TTSService(device=device)

@lru_cache(maxsize=1)
def get_cloud_storage_service() -> CloudStorageService:
    """
    Lazy load and cache the CloudStorageService instance
    """
    return CloudStorageService(
        bucket_name=GCS_BUCKET_NAME,
        credentials_path=GCS_CREDENTIALS_PATH
    )