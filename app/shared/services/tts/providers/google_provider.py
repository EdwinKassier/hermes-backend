"""Google Cloud Text-to-Speech Provider."""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from ..base_provider import BaseTTSProvider
from ..circuit_breaker import CircuitBreaker
from ..rate_limiter import RateLimiter
from ..simple_cache import _tts_cache, cached_tts

try:
    from google.cloud import texttospeech
    from google.oauth2 import service_account
except ImportError:
    texttospeech = None
    service_account = None

logger = logging.getLogger(__name__)


class GoogleTTSProvider(BaseTTSProvider):
    """
    Google Cloud Text-to-Speech provider.

    Features:
    - High quality voices (WaveNet, Neural2)
    - Multiple languages and voices
    - Reliable enterprise-grade service
    - Thread-safe for concurrent use

    Thread Safety: YES - gRPC client is thread-safe
    CPU Bound: NO - I/O-bound (network operations)
    """

    def __init__(
        self, credentials_path: Optional[str] = None, cloud_storage_service=None
    ):
        """
        Initialize Google TTS provider.

        Args:
            credentials_path: Path to Google Cloud service account JSON
            cloud_storage_service: Optional cloud storage service

        Raises:
            ImportError: If google-cloud-texttospeech not installed
            RuntimeError: If client initialization fails
        """
        super().__init__(cloud_storage_service)

        if texttospeech is None:
            raise ImportError(
                "Google Cloud TTS dependencies not installed. "
                "Install with: pip install google-cloud-texttospeech"
            )

        try:
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.client = texttospeech.TextToSpeechClient(credentials=credentials)
                logger.info(
                    f"Google TTS initialized with credentials from {credentials_path}"
                )
            else:
                self.client = texttospeech.TextToSpeechClient()
                if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    logger.warning(
                        "GOOGLE_APPLICATION_CREDENTIALS not set, "
                        "relying on default credentials"
                    )
                logger.info("Google TTS initialized with default credentials")

            # Initialize rate limiter and circuit breaker
            self.rate_limiter = RateLimiter("google")
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5, timeout=60, expected_exception=Exception
            )

            logger.info("Google TTS provider initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Google TTS client: %s", e)
            raise RuntimeError(f"Could not initialize Google TTS client: {e}") from e

    @property
    def provider_name(self) -> str:
        return "google"

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
            logger.warning(
                f"Text length {len(text)} exceeds recommended limit of 5000 characters"
            )

    def _validate_output_filepath(self, output_filepath: Optional[str]) -> None:
        """Validate output filepath if provided."""
        if output_filepath:
            # Check if directory exists
            import os

            directory = os.path.dirname(output_filepath)
            if directory and not os.path.exists(directory):
                raise ValueError(f"Output directory does not exist: {directory}")

    def _clean_text_with_logging(self, text: str) -> str:
        """
        Clean text for TTS generation, removing markdown and problematic characters.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text suitable for TTS
        """
        import re

        original_text = text

        # Remove markdown formatting
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
        text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
        text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_
        text = re.sub(r"`([^`]+)`", r"\1", text)  # `code`
        text = re.sub(r"```[^`]*```", "", text)  # ```code blocks```
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [link](url)
        text = re.sub(r"#{1,6}\s+", "", text)  # # headers

        # Remove problematic characters for speech
        text = re.sub(r"[•\-—–]+", ", ", text)  # bullet points and dashes
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        text = text.strip()

        if text != original_text:
            logger.debug(f"Text cleaned: '{original_text[:50]}...' -> '{text[:50]}...'")

        return text

    def _handle_cloud_upload(
        self,
        local_path: str,
        sample_rate: int,
        upload_to_cloud: bool,
        cloud_destination_path: Optional[str],
        cloud_storage_service_override,
        file_extension: str,
        audio_format: str,
    ) -> Dict[str, Any]:
        """
        Handle cloud upload and return result dict.

        Args:
            local_path: Local file path
            sample_rate: Audio sample rate
            upload_to_cloud: Whether to upload to cloud
            cloud_destination_path: Destination path in cloud storage
            cloud_storage_service_override: Override cloud storage service
            file_extension: File extension
            audio_format: Audio format (e.g., wav)

        Returns:
            Dict with local_path, sample_rate, and optionally cloud_url
        """
        result = {
            "local_path": local_path,
            "sample_rate": sample_rate,
            "audio_format": audio_format,
        }

        # Attempt cloud upload if requested and cloud storage is available
        if upload_to_cloud:
            cloud_service = cloud_storage_service_override or self.cloud_storage_service
            if cloud_service:
                try:
                    # Use upload_and_get_signed_url to get the actual URL
                    cloud_url = cloud_service.upload_and_get_signed_url(
                        local_file_path=local_path,
                        destination_blob_name=cloud_destination_path,
                    )
                    result["cloud_url"] = cloud_url
                    logger.info(f"Audio uploaded to cloud: {cloud_url}")
                except Exception as e:
                    logger.warning(
                        f"Cloud upload failed, using local file only: {str(e)}"
                    )
            else:
                logger.debug("Cloud storage not available, using local file only")

        return result

    @cached_tts(_tts_cache)
    def generate_audio(
        self,
        text: str,
        output_filepath: Optional[str] = None,
        upload_to_cloud: bool = True,
        cloud_destination_path: Optional[str] = None,
        cloud_storage_service_override=None,
        # Google-specific parameters
        voice_params: Optional[Dict[str, Any]] = None,
        audio_config_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate audio using Google Cloud TTS.

        Args:
            text: Text to synthesize
            output_filepath: Local file path
            upload_to_cloud: Upload to cloud storage
            cloud_destination_path: Cloud storage path
            cloud_storage_service_override: Override cloud storage service
            voice_params: Dict for VoiceSelectionParams (language_code, name, ssml_gender)
            audio_config_params: Dict for AudioConfig (sample_rate_hertz, etc.)

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

        # Prepare synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)

        # Default voice configuration
        final_voice_params = {
            "language_code": "en-gb",
            "name": "en-GB-Chirp3-HD-Enceladus",
            "ssml_gender": "MALE",
        }
        if voice_params:
            final_voice_params.update(voice_params)

        # Handle ssml_gender enum conversion
        if "ssml_gender" in final_voice_params and isinstance(
            final_voice_params["ssml_gender"], str
        ):
            try:
                final_voice_params["ssml_gender"] = texttospeech.SsmlVoiceGender[
                    final_voice_params["ssml_gender"].upper()
                ]
            except KeyError:
                logger.warning(
                    f"[Google TTS] Invalid ssml_gender: {final_voice_params['ssml_gender']}. "
                    f"Valid: {', '.join(g.name for g in texttospeech.SsmlVoiceGender)}"
                )
                # Let API handle invalid value

        voice = texttospeech.VoiceSelectionParams(**final_voice_params)

        # Default audio configuration
        final_audio_config = {
            "audio_encoding": texttospeech.AudioEncoding.LINEAR16,  # WAV
            "sample_rate_hertz": 24000,
        }
        if audio_config_params:
            final_audio_config.update(audio_config_params)
            # Force LINEAR16 for WAV output
            if (
                final_audio_config.get("audio_encoding")
                != texttospeech.AudioEncoding.LINEAR16
            ):
                logger.warning("[Google TTS] Forcing LINEAR16 encoding for WAV output")
                final_audio_config["audio_encoding"] = (
                    texttospeech.AudioEncoding.LINEAR16
                )

        audio_config = texttospeech.AudioConfig(**final_audio_config)
        sample_rate = final_audio_config["sample_rate_hertz"]

        # Create temp file if needed
        if output_filepath is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
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
                synthesis_input,
                voice,
                audio_config,
                output_filepath,
                cleaned_text,
                final_voice_params,
                sample_rate,
            )

            # Handle cloud upload
            return self._handle_cloud_upload(
                local_path=output_filepath,
                sample_rate=sample_rate,
                upload_to_cloud=upload_to_cloud,
                cloud_destination_path=cloud_destination_path,
                cloud_storage_service_override=cloud_storage_service_override,
                file_extension=".wav",
                audio_format="wav",  # Google returns LINEAR16 (WAV)
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
        synthesis_input,
        voice,
        audio_config,
        output_filepath: str,
        cleaned_text: str,
        final_voice_params: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Internal implementation of audio generation.

        Args:
            synthesis_input: Google SynthesisInput object
            voice: Google VoiceSelectionParams object
            audio_config: Google AudioConfig object
            output_filepath: Path to save audio
            cleaned_text: Cleaned text for logging
            final_voice_params: Voice params for logging
            sample_rate: Sample rate for logging

        Raises:
            RuntimeError: If generation fails
        """
        logger.info(
            "[Google TTS] Generating audio: text='%s...', voice=%s, rate=%dHz",
            cleaned_text[:50],
            final_voice_params,
            sample_rate,
        )

        try:
            # Synthesize speech
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Write to file
            with open(output_filepath, "wb") as f:
                f.write(response.audio_content)

            logger.info("[Google TTS] Generated audio at %s", output_filepath)

        except Exception as e:
            logger.error("[Google TTS] Error during audio generation: %s", e)
            raise RuntimeError(f"[Google TTS] Audio generation failed: {e}") from e

    def close(self) -> None:
        """Clean up resources."""
        # Google client doesn't require explicit cleanup
        pass
