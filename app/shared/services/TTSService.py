import logging
import os
import random
import re
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from app.shared.services.CloudStorageService import CloudStorageService
from google.cloud import texttospeech

if TYPE_CHECKING:
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS

logger = logging.getLogger(__name__)


@contextmanager
def _patched_torch_load_context(torch_module, map_location_device):
    """Patch torch.load to use a specific map_location within a context.

    This context manager ensures that any torch.load operations within its scope
    will use the specified device for tensor mapping if not explicitly overridden.

    Args:
        torch_module: The torch module (passed as parameter for lazy loading)
        map_location_device: The torch.device to use for mapping tensors
    """
    original_load = torch_module.load

    def patched_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location_device
        return original_load(*args, **kwargs)

    logger.info("Patching torch.load to use map_location=%s for this context.", map_location_device)
    torch_module.load = patched_load
    try:
        yield
    finally:
        logger.info("Restoring original torch.load.")
        torch_module.load = original_load

class TTSService:
    """Service for text-to-speech conversion using ChatterboxTTS or Google Cloud TTS.

    This service handles TTS model loading, audio generation, and optional
    cloud storage integration for the generated audio files.
    """

    def __init__(
        self,
        tts_provider: str = "google",
        device: Optional[str] = None, # For Chatterbox
        cloud_storage_config: Optional[Dict[str, Any]] = None,
        google_tts_credentials_path: Optional[str] = None, # For Google TTS
    ) -> None:
        """Initialize the TTSService.

        Args:
            tts_provider: The TTS provider to use ("chatterbox" or "google").
            device: For Chatterbox, the device to load the model on (e.g., "cuda", "cpu", "mps").
                   If None, it will check TTS_DEVICE env var, then auto-detect.
            cloud_storage_config: Configuration for CloudStorageService.
            google_tts_credentials_path: Optional path to Google Cloud service account JSON file.
                                         If None, relies on GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        self.tts_provider = tts_provider.lower()
        logger.info("Initializing TTSService with provider: %s", self.tts_provider)

        # Initialize cloud storage if configured
        self._init_cloud_storage(cloud_storage_config)

        if self.tts_provider == "chatterbox":
            # Lazy import ML dependencies (only when using chatterbox)
            try:
                import torch
                import torchaudio as ta
                from chatterbox.tts import ChatterboxTTS
                
                # Store references for use throughout the class
                self._torch = torch
                self._torchaudio = ta
                self._ChatterboxTTS = ChatterboxTTS
                
                self.device = self._get_configured_device(device)
                self.map_location = torch.device(self.device)
                logger.info("Chatterbox TTS will use device: %s", self.device)
                self._load_chatterbox_model()
                # self.sample_rate is set in _load_chatterbox_model
            except ImportError as e:
                raise ImportError(
                    f"Chatterbox TTS dependencies not installed: {e}. "
                    "Install with: pip install -r requirements-ml.txt"
                ) from e
        elif self.tts_provider == "google":
            self._init_google_tts_client(google_tts_credentials_path)
            # For Google, sample_rate is determined per request via audio_config
            self.tts_model = None # To avoid AttributeError if accessed by mistake
            self.sample_rate = None # Not fixed for the service with Google provider
            # Set these to None since we won't use them
            self._torch = None
            self._torchaudio = None
            self._ChatterboxTTS = None
        else:
            raise ValueError(
                f"Unsupported TTS provider: {self.tts_provider}. "
                "Choose 'chatterbox' or 'google'."
            )

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Clean markdown syntax from text while preserving the actual content.
        
        This method removes common markdown formatting while keeping the text
        content intact for better TTS processing.
        
        Args:
            text: The text containing markdown syntax to clean
            
        Returns:
            str: The cleaned text with markdown syntax removed
        """
        if not text:
            return text
            
        # Remove code blocks (```code``` or `code`)
        text = re.sub(r'```[\s\S]*?```', '', text)  # Multi-line code blocks
        text = re.sub(r'`[^`]*`', '', text)  # Inline code
        
        # Remove headers (# ## ### etc.)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold and italic formatting (**bold** or *italic*)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold alternative
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic alternative
        
        # Remove strikethrough (~~text~~)
        text = re.sub(r'~~([^~]+)~~', r'\1', text)
        
        # Remove links but keep the text ([text](url) -> text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove image syntax but keep alt text (![alt](url) -> alt)
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        
        # Remove any remaining leading exclamation marks from image processing
        text = re.sub(r'\s+!\s+', ' ', text)  # Remove standalone ! with spaces
        text = re.sub(r'^!\s+', '', text)  # Remove leading ! at start of line
        text = re.sub(r'\s+!([a-zA-Z])', r' \1', text)  # Remove ! before words
        
        # Remove blockquotes (> text)
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules (---, ***, ___)
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove list markers (- * + 1. 2. etc.)
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)  # Unordered lists
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)  # Ordered lists
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple blank lines to double line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        # Remove leading/trailing whitespace from each line
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Final cleanup of leading/trailing whitespace
        text = text.strip()
        
        return text

    def _get_configured_device(self, device: Optional[str]) -> str:
        """Determine the appropriate device for Chatterbox model execution.
        (Order: explicit param -> TTS_DEVICE env -> cuda -> mps -> cpu)
        """
        if device:
            logger.info("Using device from constructor parameter: %s", device)
            return device.lower()
        env_device = os.getenv("TTS_DEVICE")
        if env_device:
            logger.info("Using device from TTS_DEVICE env var: %s", env_device)
            return env_device.lower()
        if self._torch.cuda.is_available():
            logger.info("CUDA is available. Using 'cuda'.")
            return "cuda"
        if hasattr(self._torch.backends, 'mps') and self._torch.backends.mps.is_available():
            logger.info("MPS is available. Using 'mps'.")
            return "mps"
        logger.info("No GPU (CUDA/MPS) detected or specified. Using 'cpu'.")
        return "cpu"

    def _load_chatterbox_model(self) -> None:
        """Load the ChatterboxTTS model."""
        try:
            with _patched_torch_load_context(self._torch, self.map_location):
                self.tts_model = self._ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = self.tts_model.sr # Specific to Chatterbox
            logger.info("ChatterboxTTS model loaded successfully on %s. Sample rate: %s Hz.", self.device, self.sample_rate)
        except Exception as e:
            error_msg = f"Failed to load ChatterboxTTS model: {str(e)}"
            if "cuda" in self.device.lower() and not self._torch.cuda.is_available():
                error_msg += (
                    "\nCUDA was specified but torch.cuda.is_available() is False. "
                    "Ensure PyTorch is installed with CUDA support and CUDA drivers "
                    "are correctly set up."
                )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _init_google_tts_client(self, credentials_path: Optional[str] = None) -> None:
        """Initialize the Google Cloud TextToSpeechClient."""
        try:
            from google.cloud import texttospeech
            from google.oauth2 import service_account
            
            if credentials_path:
                # Pass credentials directly to client (no env mutation)
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.google_tts_client = texttospeech.TextToSpeechClient(
                    credentials=credentials
                )
                logger.info("Using Google Cloud credentials from: %s", credentials_path)
            else:
                # Uses Application Default Credentials from environment
                self.google_tts_client = texttospeech.TextToSpeechClient()
                if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    logger.warning(
                        "GOOGLE_APPLICATION_CREDENTIALS environment variable not set, "
                        "and no credentials_path provided. Relying on default credentials."
                    )
            logger.info("Google Cloud TextToSpeechClient initialized successfully.")
        except ImportError:
            logger.error(
                "google-cloud-texttospeech library not found. "
                "Please install it to use the Google TTS provider (pip install google-cloud-texttospeech)."
            )
            raise ImportError("google-cloud-texttospeech is required for Google TTS provider.")
        except Exception as e:
            logger.error("Failed to initialize Google Cloud TextToSpeechClient: %s", e)
            raise RuntimeError(f"Could not initialize Google TTS client: {e}") from e

    def _init_cloud_storage(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize cloud storage service."""
        if not config:
            self.cloud_storage_service_instance = None
            logger.info("CloudStorageService not configured.")
            return
        try:
            self.cloud_storage_service_instance = CloudStorageService(**config)
            logger.info("CloudStorageService initialized successfully.")
        except Exception as e:
            self.cloud_storage_service_instance = None
            logger.warning("Failed to initialize CloudStorageService: %s", str(e))

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility (primarily for Chatterbox)."""
        random.seed(seed)
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)
        logger.debug("Random seed set to %d", seed)

    def _handle_file_paths_and_cloud_upload(
        self,
        generated_local_path: str,
        generated_sample_rate: int,
        upload_to_cloud: bool,
        cloud_storage_service_override: Optional[CloudStorageService],
        cloud_destination_path: Optional[str],
        provider_name_for_log: str
    ) -> Dict[str, Any]:
        """Handles cloud upload and local file cleanup, returns result dict."""
        result = {
            'local_path': generated_local_path,
            'sample_rate': generated_sample_rate,
            'cloud_url': None
        }
        
        current_cloud_service = cloud_storage_service_override or self.cloud_storage_service_instance
        
        if upload_to_cloud and current_cloud_service:
            try:
                if cloud_destination_path is None:
                    cloud_destination_path = f"tts_output_{provider_name_for_log.lower()}/{uuid.uuid4()}.wav"
                
                logger.info(
                    "[%s TTS] Uploading to cloud storage: %s",
                    provider_name_for_log, cloud_destination_path
                )
                # First upload the file
                blob = current_cloud_service.upload_file(
                    local_file_path=generated_local_path,
                    destination_blob_name=cloud_destination_path
                )
                # Then generate a signed URL
                cloud_url = current_cloud_service.get_signed_url(
                    blob_name=cloud_destination_path,
                    expiration_seconds=86400  # 1 day
                )
                result['cloud_url'] = cloud_url
                logger.info(
                    "[%s TTS] Successfully uploaded to cloud storage: %s",
                    provider_name_for_log, cloud_url
                )
                
                # Clean up local file after successful upload, even if user specified path
                try:
                    os.remove(generated_local_path)
                    logger.info("[%s TTS] Local file cleaned up: %s", provider_name_for_log, generated_local_path)
                    result['local_path'] = None # Indicate local file is gone
                except OSError as cleanup_error:
                    logger.error(
                        "[%s TTS] Error cleaning up local file %s: %s",
                        provider_name_for_log, generated_local_path, cleanup_error
                    )
                    # Keep local_path in result if cleanup fails
                    
            except Exception as upload_error:
                logger.error(
                    "[%s TTS] Failed to upload to cloud storage: %s", 
                    provider_name_for_log, str(upload_error)
                )
                # Continue with local file path in result
        elif upload_to_cloud and not current_cloud_service:
            logger.warning(
                "[%s TTS] Upload to cloud requested but CloudStorageService is not configured/initialized.",
                provider_name_for_log
            )
            
        return result

    def _generate_audio_chatterbox(
        self,
        text_input: str,
        audio_prompt_path_input: str, # Required for Chatterbox
        output_filepath: Optional[str],
        exaggeration_input: float,
        temperature_input: float,
        seed_num_input: int,
        cfgw_input: float,
        upload_to_cloud: bool,
        cloud_storage_service_override: Optional[CloudStorageService],
        cloud_destination_path: Optional[str],
    ) -> Dict[str, Any]:
        """Generates audio using ChatterboxTTS."""
        if not self.tts_model:  # Should be loaded if provider is chatterbox
            logger.error("Chatterbox TTS model is not loaded.")
            raise RuntimeError("Chatterbox TTS model is not loaded. Initialization might have failed.")

        # Clean markdown from input text
        cleaned_text = self._clean_markdown(text_input)
        if cleaned_text != text_input:
            logger.info("[Chatterbox TTS] Markdown cleaned from input text")

        if seed_num_input != 0:
            self._set_seed(int(seed_num_input))
            logger.info("[Chatterbox TTS] Set generation seed to %s", seed_num_input)
        else:
            logger.info("[Chatterbox TTS] Using default random seed (seed_num_input was 0).")

        audio_path = Path(audio_prompt_path_input)
        if not audio_path.exists():
            error_msg = f"[Chatterbox TTS] Audio prompt file not found: {audio_prompt_path_input}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(
            "[Chatterbox TTS] Preparing conditionals from audio prompt: '%s' with exaggeration=%.2f",
            audio_prompt_path_input, exaggeration_input
        )
        try:
            self.tts_model.prepare_conditionals(str(audio_path), exaggeration=exaggeration_input)
        except Exception as e:
            error_msg = f"[Chatterbox TTS] Failed to prepare conditionals: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        temp_file_created = False
        actual_output_filepath = output_filepath
        if actual_output_filepath is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                actual_output_filepath = tmpfile.name
            temp_file_created = True
            logger.info("[Chatterbox TTS] Using temporary file for output: %s", actual_output_filepath)
        
        try:
            logger.info(
                "[Chatterbox TTS] Generating audio for text: '%s...' Temp: %.2f, CFGW: %.2f, Seed: %d",
                cleaned_text[:50], temperature_input, cfgw_input, seed_num_input
            )
            audio_data = self.tts_model.generate(
                cleaned_text, temperature=temperature_input, cfg_weight=cfgw_input
            )
            self._torchaudio.save(actual_output_filepath, audio_data, self.sample_rate) # self.sample_rate is from Chatterbox model
            logger.info("[Chatterbox TTS] Generated audio saved to: %s", actual_output_filepath)
            
            return self._handle_file_paths_and_cloud_upload(
                generated_local_path=actual_output_filepath,
                generated_sample_rate=self.sample_rate,
                upload_to_cloud=upload_to_cloud,
                cloud_storage_service_override=cloud_storage_service_override,
                cloud_destination_path=cloud_destination_path,
                provider_name_for_log="Chatterbox"
            )
            
        except Exception as e:
            logger.error("[Chatterbox TTS] Error during audio generation: %s", e)
            if temp_file_created and os.path.exists(actual_output_filepath):
                try:
                    os.remove(actual_output_filepath)
                    logger.info("[Chatterbox TTS] Cleaned up temporary file %s due to error.", actual_output_filepath)
                except OSError as cleanup_error:
                    logger.warning("[Chatterbox TTS] Failed to clean up temp file %s after error: %s", actual_output_filepath, cleanup_error)
            raise RuntimeError(f"[Chatterbox TTS] Audio generation failed: {e}") from e

    def _generate_audio_google(
        self,
        text_input: str,
        output_filepath: Optional[str],
        voice_params: Optional[Dict[str, Any]],
        audio_config_params: Optional[Dict[str, Any]],
        upload_to_cloud: bool,
        cloud_storage_service_override: Optional[CloudStorageService],
        cloud_destination_path: Optional[str],
    ) -> Dict[str, Any]:
        """Generates audio using Google Cloud TTS."""
        if not hasattr(self, 'google_tts_client') or not self.google_tts_client:
            logger.error("Google TTS client is not initialized.")
            raise RuntimeError("Google TTS client is not initialized. Initialization might have failed.")

        # Clean markdown from input text
        cleaned_text = self._clean_markdown(text_input)
        if cleaned_text != text_input:
            logger.info("[Google TTS] Markdown cleaned from input text")

        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)

        # Default voice and audio config for WAV output
        final_voice_params = {"language_code": "en-gb", "name": "en-GB-Chirp3-HD-Enceladus", "ssml_gender": "MALE"}
        if voice_params:
            final_voice_params.update(voice_params)
        
        # Ensure ssml_gender is of the correct type if provided
        if "ssml_gender" in final_voice_params and isinstance(final_voice_params["ssml_gender"], str):
            try:
                gender_enum_val = texttospeech.SsmlVoiceGender[final_voice_params["ssml_gender"].upper()]
                final_voice_params["ssml_gender"] = gender_enum_val
            except KeyError:
                logger.warning(
                    f"[Google TTS] Invalid ssml_gender string: {final_voice_params['ssml_gender']}. "
                    f"Using default or ignoring. Valid are: {', '.join(g.name for g in texttospeech.SsmlVoiceGender)}"
                )
                # Decide: remove or let API error? For now, let API error if it's truly invalid type.
                # If it was just a string version of a valid enum, we converted it.
                # If it's an invalid string, API will likely reject.
        voice = texttospeech.VoiceSelectionParams(**final_voice_params)
        
        final_audio_config = {
            "audio_encoding": texttospeech.AudioEncoding.LINEAR16, # For WAV
            "sample_rate_hertz": 24000, # A common high-quality rate for WAV
        }
        if audio_config_params:
            final_audio_config.update(audio_config_params)
            # Force LINEAR16 for WAV output if user tried to change encoding
            if final_audio_config.get("audio_encoding") != texttospeech.AudioEncoding.LINEAR16:
                logger.warning(
                    "[Google TTS] audio_config_params specified a non-LINEAR16 encoding. "
                    "Forcing LINEAR16 for WAV output."
                )
                final_audio_config["audio_encoding"] = texttospeech.AudioEncoding.LINEAR16
        
        audio_config = texttospeech.AudioConfig(**final_audio_config)
        current_sample_rate = final_audio_config["sample_rate_hertz"]

        logger.info(
            "[Google TTS] Generating audio: Text='%s...', Voice=%s, AudioConfig(encoding=%s, rate=%dHz)",
            cleaned_text[:50], final_voice_params, final_audio_config["audio_encoding"], current_sample_rate
        )

        response = self.google_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        temp_file_created = False
        actual_output_filepath = output_filepath
        if actual_output_filepath is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                actual_output_filepath = tmpfile.name
            temp_file_created = True
            logger.info("[Google TTS] Using temporary file for output: %s", actual_output_filepath)

        try:
            with open(actual_output_filepath, "wb") as out_file:
                out_file.write(response.audio_content)
            logger.info("[Google TTS] Generated audio saved to: %s", actual_output_filepath)

            return self._handle_file_paths_and_cloud_upload(
                generated_local_path=actual_output_filepath,
                generated_sample_rate=current_sample_rate,
                upload_to_cloud=upload_to_cloud,
                cloud_storage_service_override=cloud_storage_service_override,
                cloud_destination_path=cloud_destination_path,
                provider_name_for_log="Google"
            )

        except Exception as e:
            logger.error("[Google TTS] Error during audio generation or saving: %s", e)
            if temp_file_created and os.path.exists(actual_output_filepath):
                try:
                    os.remove(actual_output_filepath)
                    logger.info("[Google TTS] Cleaned up temporary file %s due to error.", actual_output_filepath)
                except OSError as cleanup_error:
                    logger.warning("[Google TTS] Failed to clean up temp file %s after error: %s", actual_output_filepath, cleanup_error)
            raise RuntimeError(f"[Google TTS] Audio generation failed: {e}") from e

    def generate_audio(
        self,
        text_input: str,
        # Common parameters
        output_filepath: Optional[str] = None,
        upload_to_cloud: bool = True,
        cloud_storage_service_override: Optional[CloudStorageService] = None,
        cloud_destination_path: Optional[str] = None,
        # Chatterbox-specific parameters
        audio_prompt_path_input: str = "data/tts_audio_prompts/TrainingShort.wav", # Default for Chatterbox
        exaggeration_input: float = 0.55,
        temperature_input: float = 0.45,
        seed_num_input: int = 4,
        cfgw_input: float = 0.5,
        # Google-specific parameters
        google_voice_params: Optional[Dict[str, Any]] = None,
        google_audio_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate TTS audio using the configured provider.

        Args:
            text_input: The text to synthesize.
            output_filepath: Local path to save the audio. If None, a temp file is used.
            upload_to_cloud: If True, attempts to upload to cloud storage.
            cloud_storage_service_override: Override internal CloudStorageService.
            cloud_destination_path: Cloud path. If None, a unique name is generated.
            
            For Chatterbox provider:
                audio_prompt_path_input: Path to reference audio for voice conditioning.
                exaggeration_input: Exaggeration parameter (0.0 to 1.0).
                temperature_input: Temperature for generation.
                seed_num_input: Random seed (0 for random).
                cfgw_input: CFG/Pace weight (0.0 to 1.0).
            
            For Google provider:
                google_voice_params: Dict for google.cloud.texttospeech.VoiceSelectionParams.
                                     Example: {"language_code": "en-US", "name": "en-US-Wavenet-D"}
                                     Can also include "ssml_gender" as a string like "NEUTRAL", "MALE", "FEMALE".
                google_audio_config: Dict for google.cloud.texttospeech.AudioConfig.
                                     Example: {"sample_rate_hertz": 48000}.
                                     audio_encoding will be forced to LINEAR16 (WAV).

        Returns:
            Dict containing:
                - local_path: Path to local file or None if deleted after cloud upload.
                - sample_rate: Audio sample rate in Hz.
                - cloud_url: URL to cloud storage or None if not uploaded/failed.
        Raises:
            RuntimeError: If TTS generation fails for the selected provider.
            FileNotFoundError: (Chatterbox) If audio prompt file not found.
        """
        if self.tts_provider == "chatterbox":
            if google_voice_params or google_audio_config:
                logger.warning(
                    "[Chatterbox TTS] Google-specific parameters (google_voice_params, google_audio_config) "
                    "were provided but will be ignored for the 'chatterbox' provider."
                )
            return self._generate_audio_chatterbox(
                text_input=text_input,
                audio_prompt_path_input=audio_prompt_path_input,
                output_filepath=output_filepath,
                exaggeration_input=exaggeration_input,
                temperature_input=temperature_input,
                seed_num_input=seed_num_input,
                cfgw_input=cfgw_input,
                upload_to_cloud=upload_to_cloud,
                cloud_storage_service_override=cloud_storage_service_override,
                cloud_destination_path=cloud_destination_path,
            )
        elif self.tts_provider == "google":
            chatterbox_params_provided = any([
                audio_prompt_path_input != "data/tts_audio_prompts/TrainingShort.wav", # Check if different from default
                exaggeration_input != 0.55,
                temperature_input != 0.45,
                seed_num_input != 4,
                cfgw_input != 0.5
            ])
            if chatterbox_params_provided:
                 logger.warning(
                    "[Google TTS] Chatterbox-specific parameters (audio_prompt_path_input, exaggeration_input, etc.) "
                    "were provided but will be ignored for the 'google' provider."
                )
            return self._generate_audio_google(
                text_input=text_input,
                output_filepath=output_filepath,
                voice_params=google_voice_params,
                audio_config_params=google_audio_config,
                upload_to_cloud=upload_to_cloud,
                cloud_storage_service_override=cloud_storage_service_override,
                cloud_destination_path=cloud_destination_path,
            )
        else:
            # This should theoretically be caught in __init__
            raise RuntimeError(f"TTS provider '{self.tts_provider}' is not supported or not configured correctly.")