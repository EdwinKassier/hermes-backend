from functools import lru_cache
from app.services.GeminiService import GeminiService


@lru_cache(maxsize=1)
def get_gemini_service():
    """Lazy load and cache the GeminiService instance"""
    return GeminiService() 