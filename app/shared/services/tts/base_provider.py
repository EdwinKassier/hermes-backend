"""
Abstract base class for all TTS providers.

Defines the minimal provider interface expected by TTSService and
provider implementations under app.shared.services.tts.providers.*

All providers must implement generate_audio() and close().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTTSProvider(ABC):
    """
    Base interface for TTS providers.

    Implementations should be side-effect free on import and only allocate
    heavy resources in __init__ or lazily within generate_audio().
    """

    def __init__(self, cloud_storage_service: Optional[Any] = None) -> None:
        self.cloud_storage_service = cloud_storage_service

    @abstractmethod
    def generate_audio(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate audio for the given provider.

        Returns a dict with (at minimum):
          - local_path: str | None
          - sample_rate: int
          - cloud_url: str | None
          - audio_format: str
        Implementations may return additional keys as needed.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any provider resources. Default is a no-op."""
        return


