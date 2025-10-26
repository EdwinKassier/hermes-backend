"""Chatterbox TTS Provider - ML-based voice cloning."""

import logging
import os
import random
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from ..base_provider import BaseTTSProvider

try:
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    torch = None
    ta = None
    ChatterboxTTS = None

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_AUDIO_PROMPT = "data/tts_audio_prompts/TrainingShort.wav"
DEFAULT_EXAGGERATION = 0.55
DEFAULT_TEMPERATURE = 0.45
DEFAULT_SEED = 4
DEFAULT_CFGW = 0.5


class ChatterboxRequestQueue:
    """
    In-process queue for Chatterbox requests.
    
    Serializes CPU-bound Chatterbox operations to prevent:
    1. Memory exhaustion (too many concurrent models)
    2. CPU contention (GIL thrashing)
    3. Model state corruption (race conditions)
    
    Note: This is per-worker. Use single worker for Chatterbox!
    """
    
    def __init__(self, max_queue_size: int = 10):
        """
        Initialize request queue.
        
        Args:
            max_queue_size: Maximum number of requests in queue
        """
        self.processing_lock = threading.Lock()
        self.max_queue_size = max_queue_size
        self.active_requests = 0
        logger.info(f"Chatterbox request queue initialized (max size: {max_queue_size})")
    
    def enqueue(
        self,
        func: Callable,
        *args,
        timeout: float = 300.0,
        **kwargs
    ) -> Any:
        """
        Queue a Chatterbox generation request.
        
        Args:
            func: Function to execute
            timeout: Maximum wait time
            *args, **kwargs: Function arguments
            
        Returns:
            Result from function
            
        Raises:
            RuntimeError: If queue is full
            TimeoutError: If queued too long
        """
        # Simple serialization: wait for lock
        logger.debug(f"Chatterbox request queued (active: {self.active_requests})")
        
        acquired = self.processing_lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Chatterbox request timed out after {timeout}s waiting in queue"
            )
        
        try:
            self.active_requests += 1
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Chatterbox request processed in {elapsed:.2f}s "
                f"(active: {self.active_requests})"
            )
            
            return result
            
        finally:
            self.active_requests -= 1
            self.processing_lock.release()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            'active_requests': self.active_requests,
            'max_queue_size': self.max_queue_size
        }


class ChatterboxTTSProvider(BaseTTSProvider):
    """
    Chatterbox TTS provider - ML-based voice cloning.
    
    WARNING: This provider is NOT thread-safe due to model state!
    
    Features:
    - Voice cloning from audio sample
    - Customizable voice characteristics
    - High quality audio output
    
    Thread Safety: NO - Model has mutable state
    CPU Bound: YES - Heavy CPU/GPU operations
    
    Limitations:
    - High memory usage (~2.5GB per model)
    - Slow generation (~2s per request)
    - NOT safe for concurrent use
    - Recommended: Single worker only
    """
    
    # Shared request queue across all instances in the worker
    _request_queue = None
    _queue_lock = threading.Lock()
    
    def __init__(
        self,
        device: Optional[str] = None,
        cloud_storage_service=None
    ):
        """
        Initialize Chatterbox provider.
        
        Args:
            device: Device for model ('cuda', 'cpu', 'mps'). Auto-detect if None.
            cloud_storage_service: Optional cloud storage service
            
        Raises:
            ImportError: If chatterbox dependencies not installed
            RuntimeError: If model loading fails
        """
        super().__init__(cloud_storage_service)
        
        if torch is None:
            raise ImportError(
                "Chatterbox dependencies not installed. "
                "Install with: pip install torch torchaudio chatterbox-tts"
            )
        
        # Store module references as instance variables
        self._torch = torch
        self._torchaudio = ta
        self._ChatterboxTTS = ChatterboxTTS
        
        # Cache CUDA availability
        self._cuda_available = torch.cuda.is_available()
        
        # Determine device
        self.device = self._get_device(device)
        self.map_location = torch.device(self.device)
        
        # Load model
        self._load_model()
        
        # Initialize queue once per worker
        with self._queue_lock:
            if ChatterboxTTSProvider._request_queue is None:
                ChatterboxTTSProvider._request_queue = ChatterboxRequestQueue(
                    max_queue_size=10
                )
        
        logger.info(f"Chatterbox TTS provider initialized on {self.device}")
        logger.warning(
            "⚠️  Chatterbox is NOT thread-safe! Use single worker configuration."
        )
    
    @property
    def provider_name(self) -> str:
        return "chatterbox"
    
    @property
    def is_thread_safe(self) -> bool:
        return False
    
    @property
    def is_cpu_bound(self) -> bool:
        return True
    
    def generate_audio(
        self,
        text: str,
        output_filepath: Optional[str] = None,
        upload_to_cloud: bool = True,
        cloud_destination_path: Optional[str] = None,
        cloud_storage_service_override=None,
        # Chatterbox-specific parameters
        audio_prompt_path: str = DEFAULT_AUDIO_PROMPT,
        exaggeration: float = DEFAULT_EXAGGERATION,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        cfg_weight: float = DEFAULT_CFGW,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audio using Chatterbox TTS.
        
        Args:
            text: Text to synthesize
            output_filepath: Local file path
            upload_to_cloud: Upload to cloud storage
            cloud_destination_path: Cloud storage path
            cloud_storage_service_override: Override cloud storage service
            audio_prompt_path: Path to audio prompt for voice conditioning
            exaggeration: Exaggeration parameter (0.0 to 1.0)
            temperature: Temperature for generation
            seed: Random seed (0 for random)
            cfg_weight: CFG/Pace weight (0.0 to 1.0)
            
        Returns:
            Dict with local_path, sample_rate, cloud_url
            
        Raises:
            ValueError: If text is empty or parameters invalid
            FileNotFoundError: If audio prompt not found
            RuntimeError: If audio generation fails
            TimeoutError: If queue timeout
        """
        # Queue the request to prevent concurrent model access
        if self._request_queue:
            logger.info(
                f"Queueing Chatterbox request "
                f"(stats: {self._request_queue.get_stats()})"
            )
            
            return self._request_queue.enqueue(
                self._generate_audio_impl,
                text,
                output_filepath,
                upload_to_cloud,
                cloud_destination_path,
                cloud_storage_service_override,
                audio_prompt_path,
                exaggeration,
                temperature,
                seed,
                cfg_weight,
                timeout=300.0  # 5 minute max
            )
        else:
            # Fallback if queue not initialized
            return self._generate_audio_impl(
                text,
                output_filepath,
                upload_to_cloud,
                cloud_destination_path,
                cloud_storage_service_override,
                audio_prompt_path,
                exaggeration,
                temperature,
                seed,
                cfg_weight
            )
    
    def _generate_audio_impl(
        self,
        text: str,
        output_filepath: Optional[str],
        upload_to_cloud: bool,
        cloud_destination_path: Optional[str],
        cloud_storage_service_override,
        audio_prompt_path: str,
        exaggeration: float,
        temperature: float,
        seed: int,
        cfg_weight: float
    ) -> Dict[str, Any]:
        """Internal implementation of audio generation."""
        # Validate inputs
        self._validate_text_input(text)
        self._validate_output_filepath(output_filepath)
        
        # Clean markdown
        cleaned_text = self._clean_text_with_logging(text)
        
        # Set seed if specified (0 = random)
        if seed != 0:
            self._set_seed(seed)
        
        # Validate audio prompt
        audio_path = Path(audio_prompt_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"[Chatterbox TTS] Audio prompt file not found: {audio_prompt_path}"
            )
        
        # Prepare conditionals
        try:
            self.model.prepare_conditionals(
                str(audio_path),
                exaggeration=exaggeration
            )
        except Exception as e:
            raise RuntimeError(f"[Chatterbox TTS] Failed to prepare conditionals: {e}") from e
        
        # Create temp file if needed
        if output_filepath is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_filepath = f.name
            temp_file = True
        else:
            temp_file = False
        
        try:
            logger.info(
                "[Chatterbox TTS] Generating audio: text='%s...', temp=%.2f, cfg=%.2f, seed=%d",
                cleaned_text[:50], temperature, cfg_weight, seed
            )
            
            # Generate audio
            audio_data = self.model.generate(
                cleaned_text,
                temperature=temperature,
                cfg_weight=cfg_weight
            )
            
            # Save to file
            self._torchaudio.save(output_filepath, audio_data, self.sample_rate)
            
            logger.info("[Chatterbox TTS] Generated audio at %s", output_filepath)
            
            # Handle cloud upload
            return self._handle_cloud_upload(
                local_path=output_filepath,
                sample_rate=self.sample_rate,
                upload_to_cloud=upload_to_cloud,
                cloud_destination_path=cloud_destination_path,
                cloud_storage_service_override=cloud_storage_service_override,
                file_extension=".wav",
                audio_format="wav"  # Chatterbox saves as WAV
            )
            
        except Exception as e:
            logger.error("[Chatterbox TTS] Error during audio generation: %s", e)
            if temp_file and os.path.exists(output_filepath):
                try:
                    os.remove(output_filepath)
                except OSError:
                    pass
            raise RuntimeError(f"[Chatterbox TTS] Audio generation failed: {e}") from e
    
    def close(self) -> None:
        """Clean up model from memory."""
        if hasattr(self, 'model') and self.model:
            logger.info("Cleaning up Chatterbox model from memory")
            del self.model
            self.model = None
            if self._cuda_available:
                self._torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine device for model execution."""
        if device:
            return device.lower()
        
        env_device = os.getenv("TTS_DEVICE")
        if env_device:
            return env_device.lower()
        
        if self._torch.cuda.is_available():
            return "cuda"
        
        if hasattr(self._torch.backends, 'mps') and self._torch.backends.mps.is_available():
            return "mps"
        
        return "cpu"
    
    def _load_model(self) -> None:
        """Load the Chatterbox model."""
        try:
            with self._patched_torch_load_context():
                self.model = self._ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = int(self.model.sr)  # Ensure int type
            logger.info(
                f"Chatterbox model loaded on {self.device}, "
                f"sample_rate={self.sample_rate}Hz"
            )
        except Exception as e:
            error_msg = f"Failed to load Chatterbox model: {e}"
            if "cuda" in self.device.lower() and not self._torch.cuda.is_available():
                error_msg += (
                    "\nCUDA specified but not available. "
                    "Ensure PyTorch is installed with CUDA support."
                )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        self._torch.manual_seed(seed)
        if self._cuda_available:
            self._torch.cuda.manual_seed_all(seed)
        logger.debug(f"Random seed set to {seed}")
    
    @contextmanager
    def _patched_torch_load_context(self):
        """Patch torch.load to use specific device mapping."""
        original_load = self._torch.load
        
        def patched_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = self.map_location
            return original_load(*args, **kwargs)
        
        logger.debug(f"Patching torch.load to use map_location={self.map_location}")
        self._torch.load = patched_load
        try:
            yield
        finally:
            logger.debug("Restoring original torch.load")
            self._torch.load = original_load

