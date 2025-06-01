import torchaudio as ta
import numpy as np
import torch
import random
import os
import logging
import tempfile
import time
import uuid
import io # Import io for BytesIO
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any

import librosa
import soundfile as sf
from pydantic import BaseModel

from app.services.CloudStorageService import CloudStorageService # Assuming this path is correct

# Assuming chatterbox is installed and its modules are available
# from chatterbox.vc import ChatterboxVC # Not directly used by TTSGenerator's core function
from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.tts import ChatterboxTTS

# Import FastAPI related components for the process_audio_upload function
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Store the original torch.load for potential patching and unpatching
_original_torch_load = torch.load

def _patch_torch_load(map_location_device: torch.device):
    """Patches torch.load to always use a specific map_location."""
    logger.info(f"Patching torch.load to always use map_location={map_location_device}")
    def patched_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location_device
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_load

def _unpatch_torch_load():
    """Restores the original torch.load function."""
    logger.info("Unpatching torch.load.")
    torch.load = _original_torch_load

class TTSService:
    """
    A class to encapsulate the Text-to-Speech (TTS) generation logic using ChatterboxTTS.
    """
    def __init__(self, device: Optional[str] = None):
        """
        Initializes the TTSService by loading the ChatterboxTTS model.

        Args:
            device (Optional[str]): The device to load the model on (e.g., "cuda", "cpu", "mps").
                                    If None, it will automatically detect CUDA, then MPS, else use CPU.
        """
        # 1. Determine the device based on availability and input
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Initializing TTSService on device: {self.device}")
        
        # 2. Define map_location for torch.load
        self.map_location = torch.device(self.device)

        # 3. Apply the torch.load patch BEFORE loading the model
        # This ensures that any model checkpoints loaded by ChatterboxTTS.from_pretrained
        # are mapped to the correct device (especially important for MPS).
        _patch_torch_load(self.map_location)

        # Load TTS model
        try:
            self.tts_model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = self.tts_model.sr # Store the model's sample rate
            logger.info("ChatterboxTTS model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ChatterboxTTS model: {e}")
            raise RuntimeError(f"Could not load TTS model. Ensure chatterbox is installed correctly and models are accessible. Error: {e}")
        finally:
            # 4. Optional: Unpatch torch.load if you want to restore its original behavior
            # after model loading. For a service where this is the primary loading
            # mechanism, leaving it patched might be acceptable. However, for
            # robustness in larger applications, unpatching is good.
            # I'll leave it unpatched here for simplicity as it only affects this service.
            # If your application loads other PyTorch models later via torch.load
            # that might need different map_location logic, then uncomment _unpatch_torch_load()
            # _unpatch_torch_load() # Uncomment if you want to revert the patch immediately after model load.
            pass # The patch remains active for the lifetime of the application unless explicitly unpatched.

    @staticmethod
    def _set_seed(seed: int):
        """Sets the random seed for reproducibility across torch, numpy, and random."""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        logger.debug(f"Random seed set to {seed}")

    def generate_audio(
        self,
        text_input: str,
        audio_prompt_path_input: str ="Training.wav",
        output_filepath: str = "output.wav",
        # Set default values for the parameters here
        exaggeration_input: float = 0.55,
        temperature_input: float = 0.45,
        seed_num_input: int = 4, # 0 for truly random
        cfgw_input: float = 0.5,
        upload_to_cloud: bool = True,
        cloud_storage_service: Optional[CloudStorageService] = None,
        cloud_destination_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates TTS audio using the ChatterboxTTS model with specified parameters
        and saves it to the given output file path.

        Args:
            text_input: The text to synthesize (max 300 characters recommended).
            audio_prompt_path_input: Path to the reference audio file for voice conditioning.
            output_filepath: The full path including filename where the generated .wav audio will be saved.
            exaggeration_input: Exaggeration parameter for voice conditioning (default: 0.55).
            temperature_input: Temperature parameter for generation (controls randomness) (default: 0.55).
            seed_num_input: Random seed (0 for truly random generation) (default: 4).
            cfgw_input: CFG/Pace weight for text-to-speech (default: 0.5).
            upload_to_cloud: If True, attempts to upload the generated audio to cloud storage.
            cloud_storage_service: An instance of CloudStorageService if upload_to_cloud is True.
            cloud_destination_path: The specific path/filename in cloud storage. If None, a unique name is generated.

        Returns:
            A dictionary containing 'local_path', 'sample_rate', and 'cloud_url'.

        Raises:
            RuntimeError: If the TTS model is not loaded or if generation fails.
        """
        if not self.tts_model:
            raise RuntimeError("TTS model is not loaded. Please ensure initialization was successful.")

        if seed_num_input != 0:
            self._set_seed(int(seed_num_input))
            logger.info(f"Set generation seed to {seed_num_input}")
        else:
            logger.info("Using random seed for audio generation (seed_num_input was 0).")
            
        self.cloud_storage_service = CloudStorageService()

        logger.info(f"Preparing conditionals from audio prompt: '{audio_prompt_path_input}' with exaggeration={exaggeration_input}")
        try:
            self.tts_model.prepare_conditionals(
                audio_prompt_path_input,
                exaggeration=exaggeration_input
            )
        except Exception as e:
            logger.error(f"Failed to prepare conditionals from prompt {audio_prompt_path_input}: {e}")
            raise RuntimeError(f"Could not process audio prompt: {e}")
        
        try:
            logger.info(f"Generating audio for text: '{text_input[:50]}...' "
                        f"Temp: {temperature_input}, CFGW: {cfgw_input}, Seed: {seed_num_input}")
            
            audio = self.tts_model.generate(
                text=text_input,
                exaggeration=exaggeration_input,
                temperature=temperature_input,
                cfg_weight=cfgw_input,
            )

            # Ensure audio tensor is on CPU before saving to avoid issues with non-CPU devices
            ta.save(output_filepath, audio.to('cpu'), sample_rate=self.sample_rate)
            logger.info(f"Generated audio saved to: {output_filepath}")
            
            result = {
                'local_path': output_filepath,
                'sample_rate': self.sample_rate,
                'cloud_url': None
            }
            
            # Upload to cloud storage if requested
            if upload_to_cloud and self.cloud_storage_service:
                try:
                    if not cloud_destination_path:
                        # Generate a unique filename if none provided
                        filename = f"tts_output_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
                        cloud_destination_path = f"tts_generated/{filename}"
                    
                    # Upload and get signed URL
                    signed_url = self.cloud_storage_service.upload_and_get_signed_url(
                        local_file_path=output_filepath,
                        destination_blob_name=cloud_destination_path,
                        expiration_seconds=86400  # 24 hours
                    )
                    
                    result['cloud_url'] = signed_url
                    logger.info(f"Audio uploaded to cloud storage: {signed_url}")
                    
                    # Clean up the local file if upload was successful
                    try:
                        os.remove(output_filepath)
                        logger.info(f"Local file cleaned up: {output_filepath}")
                        result['local_path'] = None # Indicate local file was removed
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up local file {output_filepath}: {cleanup_error}")
                        
                except Exception as upload_error:
                    logger.error(f"Failed to upload to cloud storage: {upload_error}")
                    # Don't fail the whole operation if upload fails, just log it.
            
            return result
            
        except Exception as e:
            logger.error(f"Error during audio generation: {e}")
            raise RuntimeError(f"Audio generation failed: {e}")