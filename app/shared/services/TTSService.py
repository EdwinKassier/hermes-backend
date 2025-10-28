"""
Text-to-Speech Service - Unified interface for multiple TTS providers.

This service provides a facade for multiple TTS provider implementations:
- ElevenLabs (default, lowest latency ~75ms)
- Google Cloud TTS

The implementation has been refactored into modular components:
- app.shared.services.tts.base_provider: Abstract base class
- app.shared.services.tts.providers.*: Provider implementations
- app.shared.services.tts.rate_limiter: Rate limiting
- app.shared.services.tts.circuit_breaker: Fault tolerance
- app.shared.services.tts.simple_cache: Response caching
"""

import logging
from typing import Any, Dict, Optional

from .tts.base_provider import BaseTTSProvider
from .tts.providers.elevenlabs_provider import (
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_VOICE_ID,
    ElevenLabsTTSProvider,
)
from .tts.providers.google_provider import GoogleTTSProvider

logger = logging.getLogger(__name__)

# Provider constants (for backward compatibility)
PROVIDER_GOOGLE = "google"
PROVIDER_ELEVENLABS = "elevenlabs"


class TTSService:
    """
    Text-to-Speech service facade providing a unified interface across providers.

    This service acts as a factory and facade, delegating actual TTS operations
    to provider-specific implementations.

    Supported providers:
    - elevenlabs: Lowest latency (~75ms), default
    - google: Google Cloud TTS

    Usage:
        # Via factory function (recommended)
        from app.shared.utils.service_loader import get_tts_service
        tts = get_tts_service()

        # Direct instantiation
        tts = TTSService(tts_provider="elevenlabs", elevenlabs_api_key="...")

        # Generate audio
        result = tts.generate_audio("Hello world!")
        # Returns: {local_path: str|None, sample_rate: int, cloud_url: str|None}

    Thread Safety:
        - ElevenLabs & Google providers: Thread-safe
    """

    def __init__(
        self,
        tts_provider: str = PROVIDER_ELEVENLABS,
        cloud_storage_config: Optional[Dict[str, Any]] = None,
        google_tts_credentials_path: Optional[str] = None,  # For Google
        elevenlabs_api_key: Optional[str] = None,  # For ElevenLabs
    ) -> None:
        """
        Initialize TTSService with specified provider.

        Args:
            tts_provider: Provider to use ('elevenlabs', 'google')
            cloud_storage_config: Config dict for CloudStorageService
            google_tts_credentials_path: Path to Google credentials JSON
            elevenlabs_api_key: ElevenLabs API key

        Raises:
            ValueError: If provider is unsupported
            ImportError: If provider dependencies not installed
        """
        self._provider_name = tts_provider.lower()
        logger.info(f"Initializing TTSService with provider: {self._provider_name}")

        # Initialize cloud storage if configured
        cloud_storage_service = None
        if cloud_storage_config:
            try:
                from .CloudStorageService import CloudStorageService

                cloud_storage_service = CloudStorageService(**cloud_storage_config)
                logger.info("CloudStorageService initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudStorageService: {e}")

        # Initialize provider
        if self._provider_name == PROVIDER_ELEVENLABS:
            self._provider: BaseTTSProvider = ElevenLabsTTSProvider(
                api_key=elevenlabs_api_key, cloud_storage_service=cloud_storage_service
            )
            # Expose provider attributes for backward compatibility
            self.elevenlabs_client = (
                self._provider.client if hasattr(self._provider, "client") else None
            )
            self.tts_model = None
            self.sample_rate = None
            self.device = None

        elif self._provider_name == PROVIDER_GOOGLE:
            self._provider: BaseTTSProvider = GoogleTTSProvider(
                credentials_path=google_tts_credentials_path,
                cloud_storage_service=cloud_storage_service,
            )
            # Expose provider attributes for backward compatibility
            self.google_tts_client = (
                self._provider.client if hasattr(self._provider, "client") else None
            )
            self.tts_model = None
            self.sample_rate = None
            self.device = None

        else:
            raise ValueError(
                f"Unsupported TTS provider: {self._provider_name}. "
                f"Choose from: {PROVIDER_ELEVENLABS}, {PROVIDER_GOOGLE}"
            )

        logger.info(f"TTSService initialized with {self._provider_name} provider")

    @property
    def tts_provider(self) -> str:
        """Get the current TTS provider name (read-only)."""
        return self._provider_name

    def generate_audio(
        self,
        text_input: str,
        # Common parameters
        output_filepath: Optional[str] = None,
        upload_to_cloud: bool = True,
        cloud_storage_service_override: Optional[Any] = None,
        cloud_destination_path: Optional[str] = None,
        # Google-specific (for backward compatibility)
        google_voice_params: Optional[Dict[str, Any]] = None,
        google_audio_config: Optional[Dict[str, Any]] = None,
        # ElevenLabs-specific (for backward compatibility)
        elevenlabs_voice_id: str = DEFAULT_VOICE_ID,
        elevenlabs_model_id: str = DEFAULT_MODEL_ID,
        elevenlabs_output_format: str = DEFAULT_OUTPUT_FORMAT,
        elevenlabs_voice_settings: Optional[Dict[str, Any]] = None,
        elevenlabs_optimize_streaming_latency: Optional[int] = None,
        elevenlabs_language_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate TTS audio using the configured provider.

        Args:
            text_input: Text to synthesize
            output_filepath: Optional local path for audio file
            upload_to_cloud: Whether to upload to cloud storage
            cloud_storage_service_override: Override cloud storage service
            cloud_destination_path: Cloud storage destination path

            # Provider-specific parameters (ignored if not applicable)
            google_voice_params: Google voice selection parameters
            google_audio_config: Google audio configuration
            elevenlabs_voice_id: ElevenLabs voice ID
            elevenlabs_model_id: ElevenLabs model ID
            elevenlabs_output_format: ElevenLabs output format
            elevenlabs_voice_settings: ElevenLabs voice settings
            elevenlabs_optimize_streaming_latency: ElevenLabs latency optimization
            elevenlabs_language_code: ElevenLabs language code

        Returns:
            Dict with keys:
                - local_path: str | None (path to local file)
                - sample_rate: int (audio sample rate in Hz)
                - cloud_url: str | None (signed URL if uploaded)
                - audio_format: str (audio format: 'wav', 'mp3', 'pcm', etc.)

        Raises:
            ValueError: If text is empty or parameters are invalid
            RuntimeError: If audio generation fails
        """
        # Delegate to provider based on type
        if self._provider_name == PROVIDER_GOOGLE:
            return self._provider.generate_audio(
                text=text_input,
                output_filepath=output_filepath,
                upload_to_cloud=upload_to_cloud,
                cloud_destination_path=cloud_destination_path,
                cloud_storage_service_override=cloud_storage_service_override,
                voice_params=google_voice_params,
                audio_config_params=google_audio_config,
            )

        elif self._provider_name == PROVIDER_ELEVENLABS:
            return self._provider.generate_audio(
                text=text_input,
                output_filepath=output_filepath,
                upload_to_cloud=upload_to_cloud,
                cloud_destination_path=cloud_destination_path,
                cloud_storage_service_override=cloud_storage_service_override,
                voice_id=elevenlabs_voice_id,
                model_id=elevenlabs_model_id,
                output_format=elevenlabs_output_format,
                voice_settings=elevenlabs_voice_settings,
                optimize_streaming_latency=elevenlabs_optimize_streaming_latency,
                language_code=elevenlabs_language_code,
            )

        else:
            raise RuntimeError(f"Unsupported provider: {self._provider_name}")

    def close(self) -> None:
        """
        Clean up provider resources.

        Call this when done with the service to free memory.
        """
        self._provider.close()
        logger.info(f"TTSService ({self._provider_name}) closed")

    def __enter__(self) -> "TTSService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TTSService(provider={self._provider_name!r})"


__all__ = [
    "TTSService",
    "PROVIDER_GOOGLE",
    "PROVIDER_ELEVENLABS",
]
