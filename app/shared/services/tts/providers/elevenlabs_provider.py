"""ElevenLabs TTS Provider - Lowest latency option (~75ms)."""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from ..base_provider import BaseTTSProvider
from ..rate_limiter import RateLimiter
from ..circuit_breaker import CircuitBreaker
from ..simple_cache import cached_tts, _tts_cache

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
except ImportError:
    ElevenLabs = None
    VoiceSettings = None

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # Brian - conversational
DEFAULT_MODEL_ID = "eleven_flash_v2_5"  # Lowest latency (~75ms)
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class ElevenLabsTTSProvider(BaseTTSProvider):
    """
    ElevenLabs TTS provider - lowest latency option (~75ms).
    
    Features:
    - Lowest latency with eleven_flash_v2_5 model
    - High quality voice synthesis
    - Multiple output formats
    - Thread-safe for concurrent use
    
    Thread Safety: YES - HTTP client is thread-safe
    CPU Bound: NO - I/O-bound (network operations)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cloud_storage_service=None
    ):
        """
        Initialize ElevenLabs provider.
        
        Args:
            api_key: ElevenLabs API key (or use EL_API_KEY env var)
            cloud_storage_service: Optional cloud storage service
            
        Raises:
            ImportError: If elevenlabs package not installed
            ValueError: If API key not provided
        """
        super().__init__(cloud_storage_service)
        
        if ElevenLabs is None:
            raise ImportError(
                "ElevenLabs dependencies not installed. "
                "Install with: pip install elevenlabs"
            )
        
        # Get API key
        self.api_key = api_key or os.getenv("EL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not provided. "
                "Set EL_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        self.client = ElevenLabs(api_key=self.api_key)
        
        # Initialize rate limiter and circuit breaker
        self.rate_limiter = RateLimiter('elevenlabs')
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60,
            expected_exception=Exception
        )
        
        logger.info("ElevenLabs TTS provider initialized successfully")
    
    @property
    def provider_name(self) -> str:
        return "elevenlabs"
    
    @property
    def is_thread_safe(self) -> bool:
        return True
    
    @property
    def is_cpu_bound(self) -> bool:
        return False
    
    def _validate_text_input(self, text: str) -> None:
        """Validate text input for TTS generation."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            logger.warning(f"Text length {len(text)} exceeds recommended limit of 5000 characters")
    
    def _validate_output_filepath(self, output_filepath: Optional[str]) -> None:
        """Validate output filepath if provided."""
        if output_filepath:
            # Check if directory exists
            import os
            directory = os.path.dirname(output_filepath)
            if directory and not os.path.exists(directory):
                raise ValueError(f"Output directory does not exist: {directory}")
    
    @cached_tts(_tts_cache)
    def generate_audio(
        self,
        text: str,
        output_filepath: Optional[str] = None,
        upload_to_cloud: bool = True,
        cloud_destination_path: Optional[str] = None,
        cloud_storage_service_override=None,
        # ElevenLabs-specific parameters
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        voice_settings: Optional[Dict[str, Any]] = None,
        optimize_streaming_latency: Optional[int] = None,
        language_code: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audio using ElevenLabs.
        
        Args:
            text: Text to synthesize
            output_filepath: Local file path
            upload_to_cloud: Upload to cloud storage
            cloud_destination_path: Cloud storage path
            cloud_storage_service_override: Override cloud storage service
            voice_id: ElevenLabs voice ID
            model_id: Model ID (eleven_flash_v2_5 for lowest latency)
            output_format: Audio format (e.g., mp3_44100_128)
            voice_settings: Dict with stability, similarity_boost, style, use_speaker_boost
            optimize_streaming_latency: Optimization level 0-4
            language_code: Language code for multilingual models
            
        Returns:
            Dict with local_path, sample_rate, cloud_url
            
        Raises:
            ValueError: If text is empty or parameters invalid
            RuntimeError: If audio generation fails
            TimeoutError: If rate limited
        """
        # Validate inputs
        self._validate_text_input(text)
        self._validate_output_filepath(output_filepath)
        
        # Clean markdown
        cleaned_text = self._clean_text_with_logging(text)
        
        # Parse format for file extension and sample rate
        file_extension, sample_rate = self._parse_format(output_format)
        
        # Create temp file if needed
        if output_filepath is None:
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
                output_filepath = f.name
            temp_file = True
        else:
            temp_file = False
        
        try:
            # Apply rate limiting
            self.rate_limiter.acquire(timeout=10.0)
            
            # Generate with circuit breaker protection
            self.circuit_breaker.call(
                self._generate_audio_impl,
                cleaned_text,
                output_filepath,
                voice_id,
                model_id,
                output_format,
                voice_settings,
                optimize_streaming_latency,
                language_code
            )
            
            # Handle cloud upload
            return self._handle_cloud_upload(
                local_path=output_filepath,
                sample_rate=sample_rate,
                upload_to_cloud=upload_to_cloud,
                cloud_destination_path=cloud_destination_path,
                cloud_storage_service_override=cloud_storage_service_override,
                file_extension=file_extension,
                audio_format="mp3"  # ElevenLabs returns MP3
            )
            
        except Exception as e:
            # Cleanup temp file on error
            if temp_file and os.path.exists(output_filepath):
                try:
                    os.remove(output_filepath)
                except OSError:
                    pass
            raise
    
    def _generate_audio_impl(
        self,
        text: str,
        output_filepath: str,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_settings: Optional[Dict[str, Any]],
        optimize_streaming_latency: Optional[int],
        language_code: Optional[str]
    ) -> None:
        """
        Internal implementation of audio generation.
        
        Args:
            text: Cleaned text to synthesize
            output_filepath: Path to save audio
            voice_id: Voice ID
            model_id: Model ID
            output_format: Output format
            voice_settings: Voice settings dict
            optimize_streaming_latency: Latency optimization level
            language_code: Language code
            
        Raises:
            RuntimeError: If generation fails
        """
        logger.info(
            "[ElevenLabs TTS] Generating audio: text='%s...', voice=%s, model=%s, format=%s",
            text[:50], voice_id, model_id, output_format
        )
        
        # Prepare API parameters
        params = {
            "text": text,
            "voice_id": voice_id,
            "model_id": model_id,
            "output_format": output_format,
        }
        
        # Add optional parameters
        if voice_settings and VoiceSettings:
            try:
                params["voice_settings"] = VoiceSettings(**voice_settings)
                logger.debug("[ElevenLabs TTS] Using custom voice settings: %s", voice_settings)
            except TypeError as e:
                raise ValueError(
                    f"Invalid voice_settings: {e}. "
                    f"Valid keys: stability, similarity_boost, style, use_speaker_boost"
                ) from e
        
        if optimize_streaming_latency is not None:
            params["optimize_streaming_latency"] = optimize_streaming_latency
            
        if language_code:
            params["language_code"] = language_code
        
        try:
            # Generate audio (returns iterator)
            audio_generator = self.client.text_to_speech.convert(**params)
            
            # Write audio to file (streaming)
            bytes_written = 0
            with open(output_filepath, "wb") as audio_file:
                for chunk in audio_generator:
                    if chunk:  # Check if chunk is not empty
                        audio_file.write(chunk)
                        bytes_written += len(chunk)
            
            # Validate we received audio data
            if bytes_written == 0:
                raise RuntimeError("No audio data received from ElevenLabs API")
            
            logger.info(
                "[ElevenLabs TTS] Generated audio: %d bytes at %s",
                bytes_written, output_filepath
            )
            
        except Exception as e:
            logger.error("[ElevenLabs TTS] Error during audio generation: %s", e)
            raise RuntimeError(f"[ElevenLabs TTS] Audio generation failed: {e}") from e
    
    def close(self) -> None:
        """Clean up resources."""
        # ElevenLabs client doesn't require explicit cleanup
        pass
    
    @staticmethod
    def _parse_format(output_format: str) -> tuple:
        """
        Parse output format to extract extension and sample rate.
        
        Args:
            output_format: Format string like "mp3_44100_128" or "pcm_16000"
            
        Returns:
            Tuple of (file_extension, sample_rate)
        """
        # Determine file extension
        if output_format.startswith("mp3"):
            file_extension = ".mp3"
        elif output_format.startswith("pcm"):
            file_extension = ".wav"
        else:
            file_extension = ".wav"  # Default
        
        # Extract sample rate (e.g., "mp3_44100_128" -> 44100)
        parts = output_format.split("_")
        sample_rate = 44100  # Default
        if len(parts) >= 2 and parts[1].isdigit():
            sample_rate = int(parts[1])
        
        return file_extension, sample_rate

