"""
Text-to-Speech Service - Unified interface for multiple TTS providers.

This service provides a facade for multiple TTS provider implementations:
- ElevenLabs (default, lowest latency ~75ms)
- Google Cloud TTS
- Chatterbox (ML-based voice cloning)

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
from .tts.providers.chatterbox_provider import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_CFGW,
    DEFAULT_EXAGGERATION,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    ChatterboxTTSProvider,
)
from .tts.providers.elevenlabs_provider import (
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_VOICE_ID,
    ElevenLabsTTSProvider,
)
from .tts.providers.google_provider import GoogleTTSProvider

logger = logging.getLogger(__name__)

# Provider constants (for backward compatibility)
PROVIDER_CHATTERBOX = "chatterbox"
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
    - chatterbox: ML-based voice cloning

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
        - Chatterbox provider: NOT thread-safe (use single worker!)
    """

    def __init__(
        self,
        tts_provider: str = PROVIDER_ELEVENLABS,
        device: Optional[str] = None,  # For Chatterbox
        cloud_storage_config: Optional[Dict[str, Any]] = None,
        google_tts_credentials_path: Optional[str] = None,  # For Google
        elevenlabs_api_key: Optional[str] = None,  # For ElevenLabs
    ) -> None:
        """
        Initialize TTSService with specified provider.

        Args:
            tts_provider: Provider to use ('elevenlabs', 'google', 'chatterbox')
            device: Device for Chatterbox ('cuda', 'cpu', 'mps')
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

        elif self._provider_name == PROVIDER_CHATTERBOX:
            self._provider: BaseTTSProvider = ChatterboxTTSProvider(
                device=device, cloud_storage_service=cloud_storage_service
            )
            # Expose provider attributes for backward compatibility
            self.tts_model = (
                self._provider.model if hasattr(self._provider, "model") else None
            )
            self.sample_rate = (
                self._provider.sample_rate
                if hasattr(self._provider, "sample_rate")
                else None
            )
            self.device = (
                self._provider.device if hasattr(self._provider, "device") else None
            )

        else:
            raise ValueError(
                f"Unsupported TTS provider: {self._provider_name}. "
                f"Choose from: {PROVIDER_ELEVENLABS}, {PROVIDER_GOOGLE}, {PROVIDER_CHATTERBOX}"
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
        # Chatterbox-specific (for backward compatibility)
        audio_prompt_path_input: str = DEFAULT_AUDIO_PROMPT,
        exaggeration_input: float = DEFAULT_EXAGGERATION,
        temperature_input: float = DEFAULT_TEMPERATURE,
        seed_num_input: int = DEFAULT_SEED,
        cfgw_input: float = DEFAULT_CFGW,
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
            audio_prompt_path_input: Chatterbox audio prompt path
            exaggeration_input: Chatterbox exaggeration (0.0-1.0)
            temperature_input: Chatterbox temperature
            seed_num_input: Chatterbox random seed (0 for random)
            cfgw_input: Chatterbox CFG weight
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
            FileNotFoundError: If Chatterbox audio prompt not found
        """
        # Delegate to provider based on type
        if self._provider_name == PROVIDER_CHATTERBOX:
            # Warn if non-Chatterbox params provided
            if google_voice_params or google_audio_config:
                logger.warning(
                    "[Chatterbox TTS] Google-specific parameters ignored for chatterbox provider"
                )

            return self._provider.generate_audio(
                text=text_input,
                output_filepath=output_filepath,
                upload_to_cloud=upload_to_cloud,
                cloud_destination_path=cloud_destination_path,
                cloud_storage_service_override=cloud_storage_service_override,
                audio_prompt_path=audio_prompt_path_input,
                exaggeration=exaggeration_input,
                temperature=temperature_input,
                seed=seed_num_input,
                cfg_weight=cfgw_input,
            )

        elif self._provider_name == PROVIDER_GOOGLE:
            # Warn if non-Google params provided
            if self._has_non_default_chatterbox_params(
                audio_prompt_path_input,
                exaggeration_input,
                temperature_input,
                seed_num_input,
                cfgw_input,
            ):
                logger.warning(
                    "[Google TTS] Chatterbox-specific parameters ignored for google provider"
                )

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
            # Warn if non-ElevenLabs params provided
            if self._has_non_default_chatterbox_params(
                audio_prompt_path_input,
                exaggeration_input,
                temperature_input,
                seed_num_input,
                cfgw_input,
            ):
                logger.warning(
                    "[ElevenLabs TTS] Chatterbox-specific parameters ignored for elevenlabs provider"
                )
            if google_voice_params or google_audio_config:
                logger.warning(
                    "[ElevenLabs TTS] Google-specific parameters ignored for elevenlabs provider"
                )

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

    @staticmethod
    def _has_non_default_chatterbox_params(
        audio_prompt_path_input: str,
        exaggeration_input: float,
        temperature_input: float,
        seed_num_input: int,
        cfgw_input: float,
    ) -> bool:
        """Check if non-default Chatterbox parameters were provided."""
        return any(
            [
                audio_prompt_path_input != DEFAULT_AUDIO_PROMPT,
                exaggeration_input != DEFAULT_EXAGGERATION,
                temperature_input != DEFAULT_TEMPERATURE,
                seed_num_input != DEFAULT_SEED,
                cfgw_input != DEFAULT_CFGW,
            ]
        )

    def close(self) -> None:
        """
        Clean up provider resources.

        Call this when done with the service to free memory,
        especially important for Chatterbox which loads large models.
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
    "PROVIDER_CHATTERBOX",
    "PROVIDER_GOOGLE",
    "PROVIDER_ELEVENLABS",
]
