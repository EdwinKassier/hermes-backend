"""
Audio Processing Utilities for Prism Domain

Handles conversion between TTS output formats and Attendee's required
16-bit PCM format for WebSocket audio streaming.

Attendee Requirements (per documentation):
- Sample Rate: 16kHz
- Bit Depth: 16-bit
- Channels: Mono (1)
- Encoding: Base64-encoded PCM
"""
import base64
import io
import logging
from typing import List
from pydub import AudioSegment

from .constants import AUDIO_SAMPLE_RATE, AUDIO_BIT_DEPTH, AUDIO_CHANNELS
from .exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio format conversion for Attendee WebSocket streaming.
    
    Converts various audio formats (MP3, WAV, etc.) from TTS service
    to Attendee's required 16-bit PCM format.
    """
    
    @staticmethod
    def convert_to_pcm(
        audio_data: bytes,
        source_format: str = "mp3"
    ) -> bytes:
        """
        Convert audio to 16-bit PCM format required by Attendee.

        Args:
            audio_data: Raw audio bytes from TTS service
            source_format: Source audio format ("mp3", "wav", "ogg", etc.)

        Returns:
            bytes: Raw 16-bit PCM audio data

        Raises:
            AudioProcessingError: If conversion fails
        """
        try:
            # Load audio using pydub
            audio = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format=source_format
            )
            
            # Convert to required format
            audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE)  # 16kHz
            audio = audio.set_sample_width(AUDIO_BIT_DEPTH // 8)  # 16-bit = 2 bytes
            audio = audio.set_channels(AUDIO_CHANNELS)  # Mono
            
            # Export as raw PCM (s16le = signed 16-bit little-endian)
            pcm_buffer = io.BytesIO()
            audio.export(
                pcm_buffer,
                format="s16le"  # Raw PCM format, no codec parameter needed
            )

            pcm_data = pcm_buffer.getvalue()
            logger.info(
                "Converted %d bytes (%s) to %d bytes (PCM)",
                len(audio_data), source_format, len(pcm_data)
            )

            return pcm_data
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise AudioProcessingError(f"Failed to convert audio: {str(e)}")
    
    @staticmethod
    def encode_for_websocket(pcm_data: bytes) -> str:
        """
        Encode PCM audio as base64 for WebSocket transmission.
        
        Args:
            pcm_data: Raw 16-bit PCM audio bytes
        
        Returns:
            str: Base64-encoded audio string
        """
        return base64.b64encode(pcm_data).decode('utf-8')
    
    @staticmethod
    def decode_from_websocket(base64_audio: str) -> bytes:
        """
        Decode base64 audio from WebSocket (for incoming audio if needed).
        
        Args:
            base64_audio: Base64-encoded audio string
        
        Returns:
            bytes: Raw PCM audio bytes
        """
        try:
            return base64.b64decode(base64_audio)
        except Exception as e:
            logger.error(f"Base64 decode failed: {str(e)}")
            raise AudioProcessingError(f"Failed to decode audio: {str(e)}")
    
    @staticmethod
    def split_into_chunks(
        pcm_data: bytes,
        chunk_size: int = 4096
    ) -> List[bytes]:
        """
        Split PCM audio into chunks for streaming.
        
        Args:
            pcm_data: Raw PCM audio bytes
            chunk_size: Size of each chunk in bytes
        
        Returns:
            List[bytes]: List of audio chunks
        """
        chunks = []
        for i in range(0, len(pcm_data), chunk_size):
            chunks.append(pcm_data[i:i + chunk_size])
        return chunks
    
    @staticmethod
    def get_audio_duration(pcm_data: bytes) -> float:
        """
        Calculate duration of PCM audio in seconds.
        
        Args:
            pcm_data: Raw PCM audio bytes
        
        Returns:
            float: Duration in seconds
        """
        # Duration = num_samples / sample_rate
        # num_samples = num_bytes / (sample_width * channels)
        num_bytes = len(pcm_data)
        bytes_per_sample = (AUDIO_BIT_DEPTH // 8) * AUDIO_CHANNELS
        num_samples = num_bytes / bytes_per_sample
        duration = num_samples / AUDIO_SAMPLE_RATE
        return duration
    
    @classmethod
    def process_tts_audio(
        cls,
        tts_audio: bytes,
        source_format: str = "mp3"
    ) -> dict:
        """
        Complete processing pipeline: convert TTS audio to Attendee format.
        
        Args:
            tts_audio: Audio bytes from TTS service
            source_format: Source format ("mp3", "wav", etc.)
        
        Returns:
            dict: {
                "pcm_data": bytes,
                "base64_audio": str,
                "duration": float,
                "chunk_count": int
            }
        
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            # Convert to PCM
            pcm_data = cls.convert_to_pcm(tts_audio, source_format)
            
            # Encode for WebSocket
            base64_audio = cls.encode_for_websocket(pcm_data)
            
            # Get metadata
            duration = cls.get_audio_duration(pcm_data)
            chunks = cls.split_into_chunks(pcm_data)
            
            logger.info(
                f"Processed TTS audio: {len(tts_audio)} bytes -> "
                f"{len(pcm_data)} bytes PCM, {duration:.2f}s, {len(chunks)} chunks"
            )
            
            return {
                "pcm_data": pcm_data,
                "base64_audio": base64_audio,
                "duration": duration,
                "chunk_count": len(chunks)
            }
            
        except AudioProcessingError:
            raise
        except Exception as e:
            logger.error(f"TTS audio processing failed: {str(e)}")
            raise AudioProcessingError(f"Failed to process TTS audio: {str(e)}")

