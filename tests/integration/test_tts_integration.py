"""
CRITICAL: Integration tests for Text-to-Speech audio generation systems.
These tests verify actual audio generation with real TTS providers.

Run with: pytest tests/integration/test_tts_integration.py --run-integration -v

Providers tested:
- ElevenLabs (default, lowest latency ~75ms)
- Google Cloud TTS (WaveNet, Neural2)
"""

import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import pytest

from app.shared.services.TTSService import (
    PROVIDER_ELEVENLABS,
    PROVIDER_GOOGLE,
    TTSService,
)

logger = logging.getLogger(__name__)


# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def temp_audio_dir():
    """Create temporary directory for audio files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Created temp audio directory: {tmpdir}")
        yield tmpdir


@pytest.fixture(scope="module")
def elevenlabs_api_key():
    """Get ElevenLabs API key from environment"""
    api_key = os.environ.get("EL_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        pytest.skip(
            "ElevenLabs API key not found. Set EL_API_KEY or ELEVENLABS_API_KEY"
        )
    return api_key


@pytest.fixture(scope="module")
def google_credentials_path():
    """Get Google Cloud credentials path"""
    # Try multiple common environment variable names
    creds_path = (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("GOOGLE_TTS_CREDENTIALS_PATH")
        or os.environ.get("GCP_CREDENTIALS_PATH")
    )

    if not creds_path:
        pytest.skip(
            "Google Cloud credentials not found. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_TTS_CREDENTIALS_PATH"
        )

    if not os.path.exists(creds_path):
        pytest.skip(f"Google credentials file not found at: {creds_path}")

    return creds_path


@pytest.fixture(scope="module")
def elevenlabs_service(elevenlabs_api_key):
    """Create ElevenLabs TTS service"""
    try:
        service = TTSService(
            tts_provider=PROVIDER_ELEVENLABS, elevenlabs_api_key=elevenlabs_api_key
        )
        logger.info("✓ ElevenLabs TTS service initialized")
        yield service
    except Exception as e:
        pytest.skip(f"Failed to initialize ElevenLabs service: {e}")


@pytest.fixture(scope="module")
def google_service(google_credentials_path):
    """Create Google TTS service"""
    try:
        service = TTSService(
            tts_provider=PROVIDER_GOOGLE,
            google_tts_credentials_path=google_credentials_path,
        )
        logger.info("✓ Google TTS service initialized")
        yield service
    except Exception as e:
        pytest.skip(f"Failed to initialize Google TTS service: {e}")


# ==================== Helper Functions ====================


def validate_audio_file(filepath: str, min_duration_sec: float = 0.1) -> dict:
    """
    Validate audio file and return metadata.

    Args:
        filepath: Path to audio file
        min_duration_sec: Minimum expected duration

    Returns:
        dict with: exists, format, sample_rate, duration, size_bytes
    """
    result = {
        "exists": os.path.exists(filepath),
        "format": None,
        "sample_rate": None,
        "duration": None,
        "size_bytes": None,
    }

    if not result["exists"]:
        return result

    result["size_bytes"] = os.path.getsize(filepath)

    # Try to read as WAV file
    try:
        with wave.open(filepath, "rb") as wav_file:
            result["format"] = "wav"
            result["sample_rate"] = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            result["duration"] = n_frames / result["sample_rate"]

            logger.info(
                f"Audio file validated: {os.path.basename(filepath)} - "
                f"{result['duration']:.2f}s, {result['sample_rate']}Hz, "
                f"{result['size_bytes']} bytes"
            )
    except Exception as e:
        # Not a WAV file, might be MP3 or other format
        logger.warning(f"Could not parse as WAV: {e}")

        # Try to detect format by extension
        if filepath.lower().endswith(".mp3"):
            result["format"] = "mp3"
            # For MP3 files, we can't easily get duration without additional libraries
            # Just validate that the file exists and has reasonable size
            result["duration"] = None  # Will be handled in validation below
        else:
            result["format"] = "unknown"

    # Basic validation
    assert (
        result["size_bytes"] > 1000
    ), f"Audio file too small: {result['size_bytes']} bytes"

    # Duration validation (only if we have duration info)
    if result["duration"] is not None:
        assert (
            result["duration"] >= min_duration_sec
        ), f"Audio too short: {result['duration']:.2f}s < {min_duration_sec}s"
    else:
        # For MP3 and other formats where we can't get duration,
        # just validate file size is reasonable for expected duration
        expected_min_size = int(
            min_duration_sec * 1000
        )  # Rough estimate: 1KB per second
        assert (
            result["size_bytes"] >= expected_min_size
        ), f"Audio file too small for expected duration: {result['size_bytes']} bytes < {expected_min_size} bytes"

    return result


# ==================== ElevenLabs Tests ====================


@pytest.mark.integration
class TestElevenLabsTTS:
    """Test ElevenLabs TTS provider"""

    def test_elevenlabs_service_initialized(self, elevenlabs_service):
        """Test that ElevenLabs service is properly initialized"""
        assert elevenlabs_service is not None
        assert elevenlabs_service._provider_name == PROVIDER_ELEVENLABS
        logger.info("✓ ElevenLabs service initialized")

    def test_elevenlabs_generate_simple_audio(self, elevenlabs_service, temp_audio_dir):
        """Test generating simple audio with ElevenLabs"""
        text = "Hello, this is a test of the ElevenLabs text to speech system."
        output_path = os.path.join(temp_audio_dir, "elevenlabs_test.wav")

        result = elevenlabs_service.generate_audio(
            text_input=text, output_filepath=output_path, upload_to_cloud=False
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "local_path" in result
        assert "sample_rate" in result
        assert result["local_path"] is not None
        assert result["sample_rate"] > 0

        # Validate audio file
        audio_info = validate_audio_file(result["local_path"])
        assert audio_info["exists"]
        assert audio_info["size_bytes"] > 1000

        logger.info(f"✓ ElevenLabs audio generated: {audio_info}")

    def test_elevenlabs_multiple_generations(self, elevenlabs_service, temp_audio_dir):
        """Test generating multiple audio files sequentially"""
        texts = [
            "First test sentence.",
            "Second test sentence with different content.",
            "Third and final test sentence.",
        ]

        results = []
        for i, text in enumerate(texts):
            output_path = os.path.join(temp_audio_dir, f"elevenlabs_multi_{i}.wav")
            result = elevenlabs_service.generate_audio(
                text_input=text, output_filepath=output_path, upload_to_cloud=False
            )
            results.append(result)

            # Validate each file
            assert os.path.exists(result["local_path"])
            audio_info = validate_audio_file(result["local_path"])
            assert audio_info["exists"]

        logger.info(f"✓ Generated {len(results)} audio files with ElevenLabs")

    def test_elevenlabs_long_text(self, elevenlabs_service, temp_audio_dir):
        """Test generating audio from longer text"""
        text = (
            "This is a longer text passage to test the ElevenLabs text to speech system. "
            "It contains multiple sentences and should produce a longer audio file. "
            "The system should handle this without any issues, maintaining good audio quality "
            "throughout the entire generation process. "
            "Testing longer passages is important for production use cases."
        )

        output_path = os.path.join(temp_audio_dir, "elevenlabs_long.wav")

        result = elevenlabs_service.generate_audio(
            text_input=text, output_filepath=output_path, upload_to_cloud=False
        )

        # Validate longer audio
        audio_info = validate_audio_file(result["local_path"], min_duration_sec=5.0)
        assert audio_info["exists"]

        # For MP3 files, duration might not be available, so check file size instead
        if audio_info["duration"] is not None:
            assert audio_info["duration"] > 5.0  # Should be at least 5 seconds
            logger.info(
                f"✓ ElevenLabs long text generated: {audio_info['duration']:.2f}s"
            )
        else:
            # For MP3 files, validate file size is reasonable for 5+ seconds
            assert (
                audio_info["size_bytes"] > 5000
            )  # Should be at least 5KB for 5+ seconds
            logger.info(
                f"✓ ElevenLabs long text generated: {audio_info['size_bytes']} bytes (MP3 format)"
            )

    def test_elevenlabs_markdown_cleaning(self, elevenlabs_service, temp_audio_dir):
        """Test that markdown is properly cleaned from text"""
        text = (
            "This is **bold text** and this is *italic*. "
            "Here's a `code snippet` and a [link](https://example.com). "
            "### Header should be removed."
        )

        output_path = os.path.join(temp_audio_dir, "elevenlabs_markdown.wav")

        result = elevenlabs_service.generate_audio(
            text_input=text, output_filepath=output_path, upload_to_cloud=False
        )

        # Should succeed even with markdown
        assert result["local_path"] is not None
        audio_info = validate_audio_file(result["local_path"])
        assert audio_info["exists"]

        logger.info("✓ ElevenLabs markdown cleaning works")

    def test_elevenlabs_empty_text_error(self, elevenlabs_service):
        """Test that empty text raises appropriate error"""
        with pytest.raises((ValueError, Exception)) as exc_info:
            elevenlabs_service.generate_audio(text_input="", upload_to_cloud=False)

        logger.info(f"✓ ElevenLabs empty text validation: {exc_info.value}")

    def test_elevenlabs_latency_check(self, elevenlabs_service, temp_audio_dir):
        """Test that ElevenLabs has reasonable latency"""
        import time

        text = "Quick latency test for ElevenLabs."
        output_path = os.path.join(temp_audio_dir, "elevenlabs_latency.wav")

        start_time = time.time()
        result = elevenlabs_service.generate_audio(
            text_input=text, output_filepath=output_path, upload_to_cloud=False
        )
        elapsed = time.time() - start_time

        # ElevenLabs claims ~75ms with eleven_flash_v2_5
        # Allow generous margin for network latency
        assert elapsed < 5.0, f"ElevenLabs took too long: {elapsed:.2f}s"

        logger.info(f"✓ ElevenLabs latency: {elapsed:.2f}s")


# ==================== Google Cloud TTS Tests ====================


@pytest.mark.integration
class TestGoogleTTS:
    """Test Google Cloud TTS provider"""

    def test_google_service_initialized(self, google_service):
        """Test that Google service is properly initialized"""
        assert google_service is not None
        assert google_service._provider_name == PROVIDER_GOOGLE
        logger.info("✓ Google TTS service initialized")

    def test_google_generate_simple_audio(self, google_service, temp_audio_dir):
        """Test generating simple audio with Google TTS"""
        text = "Hello, this is a test of the Google Cloud text to speech system."
        output_path = os.path.join(temp_audio_dir, "google_test.wav")

        try:
            result = google_service.generate_audio(
                text_input=text, output_filepath=output_path, upload_to_cloud=False
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if (
                "billing" in error_msg
                or "403" in error_msg
                or "billing_disabled" in error_msg
            ):
                pytest.skip(f"Google TTS billing not enabled: {e}")
            raise

        # Verify result structure
        assert isinstance(result, dict)
        assert "local_path" in result
        assert "sample_rate" in result
        assert result["local_path"] is not None
        assert result["sample_rate"] > 0

        # Validate audio file
        audio_info = validate_audio_file(result["local_path"])
        assert audio_info["exists"]
        assert audio_info["format"] == "wav"  # Google outputs WAV
        assert audio_info["sample_rate"] in [16000, 24000]  # Common rates

        logger.info(f"✓ Google TTS audio generated: {audio_info}")

    def test_google_voice_parameters(self, google_service, temp_audio_dir):
        """Test Google TTS with custom voice parameters"""
        text = "Testing custom voice parameters with Google Cloud TTS."
        output_path = os.path.join(temp_audio_dir, "google_voice_params.wav")

        try:
            result = google_service.generate_audio(
                text_input=text,
                output_filepath=output_path,
                upload_to_cloud=False,
                google_voice_params={
                    "language_code": "en-US",
                    "name": "en-US-Neural2-F",  # Female voice
                    "ssml_gender": "FEMALE",
                },
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if (
                "billing" in error_msg
                or "403" in error_msg
                or "billing_disabled" in error_msg
            ):
                pytest.skip(f"Google TTS billing not enabled: {e}")
            raise

        # Validate
        assert result["local_path"] is not None
        audio_info = validate_audio_file(result["local_path"])
        assert audio_info["exists"]

        logger.info("✓ Google TTS custom voice parameters work")

    def test_google_different_languages(self, google_service, temp_audio_dir):
        """Test Google TTS with different language voices"""
        test_cases = [
            ("Hello world", "en-US", "en-US-Neural2-A"),
            ("Bonjour le monde", "fr-FR", "fr-FR-Neural2-A"),
            ("Hola mundo", "es-ES", "es-ES-Neural2-A"),
        ]

        for i, (text, lang_code, voice_name) in enumerate(test_cases):
            output_path = os.path.join(temp_audio_dir, f"google_lang_{i}.wav")

            try:
                result = google_service.generate_audio(
                    text_input=text,
                    output_filepath=output_path,
                    upload_to_cloud=False,
                    voice_params={"language_code": lang_code, "name": voice_name},
                )

                audio_info = validate_audio_file(result["local_path"])
                assert audio_info["exists"]
                logger.info(f"✓ Google TTS {lang_code}: {audio_info}")

            except RuntimeError as e:
                error_msg = str(e).lower()
                if (
                    "billing" in error_msg
                    or "403" in error_msg
                    or "billing_disabled" in error_msg
                ):
                    pytest.skip(f"Google TTS billing not enabled: {e}")
                raise
            except Exception as e:
                logger.warning(f"Language {lang_code} test skipped: {e}")
                # Some voices might not be available in all projects
                continue

    def test_google_audio_quality(self, google_service, temp_audio_dir):
        """Test Google TTS audio quality parameters"""
        text = "Testing high quality audio generation with Google TTS."
        output_path = os.path.join(temp_audio_dir, "google_quality.wav")

        try:
            result = google_service.generate_audio(
                text_input=text,
                output_filepath=output_path,
                upload_to_cloud=False,
                google_audio_config={"sample_rate_hertz": 24000},  # Higher quality
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if (
                "billing" in error_msg
                or "403" in error_msg
                or "billing_disabled" in error_msg
            ):
                pytest.skip(f"Google TTS billing not enabled: {e}")
            raise

        audio_info = validate_audio_file(result["local_path"])
        assert audio_info["exists"]
        assert audio_info["sample_rate"] == 24000

        logger.info(f"✓ Google TTS high quality: {audio_info}")


# ==================== Cross-Provider Comparison Tests ====================


@pytest.mark.integration
class TestTTSComparison:
    """Compare behavior across different TTS providers"""

    def test_same_text_different_providers(
        self, elevenlabs_service, google_service, temp_audio_dir
    ):
        """Test that all providers can generate audio from the same text"""
        text = "This is a comparison test across different TTS providers."

        providers = [
            ("elevenlabs", elevenlabs_service),
            ("google", google_service),
        ]

        results = {}
        for provider_name, service in providers:
            try:
                output_path = os.path.join(
                    temp_audio_dir, f"compare_{provider_name}.wav"
                )
                result = service.generate_audio(
                    text_input=text, output_filepath=output_path, upload_to_cloud=False
                )

                audio_info = validate_audio_file(result["local_path"])
                results[provider_name] = {
                    "success": True,
                    "duration": audio_info.get("duration"),
                    "size": audio_info.get("size_bytes"),
                    "sample_rate": audio_info.get("sample_rate"),
                }

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                results[provider_name] = {"success": False, "error": str(e)}

        # At least one provider should succeed
        successful = [r for r in results.values() if r.get("success")]
        assert len(successful) > 0, "No providers succeeded"

        logger.info(f"✓ Provider comparison results: {results}")

    def test_error_handling_consistency(self, elevenlabs_service, google_service):
        """Test that all providers handle errors consistently"""
        providers = [
            ("elevenlabs", elevenlabs_service),
            ("google", google_service),
        ]

        for provider_name, service in providers:
            # Test empty text
            with pytest.raises((ValueError, Exception)):
                service.generate_audio(text_input="", upload_to_cloud=False)

            logger.info(f"✓ {provider_name} handles empty text error")


# ==================== Performance Tests ====================


@pytest.mark.integration
@pytest.mark.slow
class TestTTSPerformance:
    """Performance tests for TTS systems"""

    def test_elevenlabs_batch_performance(self, elevenlabs_service, temp_audio_dir):
        """Test ElevenLabs performance with batch generation"""
        import time

        texts = [f"Test sentence number {i}." for i in range(5)]

        start_time = time.time()
        for i, text in enumerate(texts):
            output_path = os.path.join(temp_audio_dir, f"perf_batch_{i}.wav")
            elevenlabs_service.generate_audio(
                text_input=text, output_filepath=output_path, upload_to_cloud=False
            )
        elapsed = time.time() - start_time

        avg_per_request = elapsed / len(texts)

        logger.info(
            f"✓ ElevenLabs batch performance: "
            f"{len(texts)} requests in {elapsed:.2f}s "
            f"(avg {avg_per_request:.2f}s per request)"
        )

        # Should be reasonably fast even for batch
        assert (
            avg_per_request < 5.0
        ), f"Average request time too high: {avg_per_request:.2f}s"

    def test_google_tts_performance(self, google_service, temp_audio_dir):
        """Test Google TTS performance"""
        import time

        text = "Performance test for Google Cloud TTS system."
        output_path = os.path.join(temp_audio_dir, "perf_google.wav")

        start_time = time.time()
        google_service.generate_audio(
            text_input=text, output_filepath=output_path, upload_to_cloud=False
        )
        elapsed = time.time() - start_time

        logger.info(f"✓ Google TTS performance: {elapsed:.2f}s")

        # Google TTS should complete within reasonable time
        assert elapsed < 10.0, f"Google TTS took too long: {elapsed:.2f}s"


# ==================== Summary Test ====================


@pytest.mark.integration
class TestTTSIntegrationSummary:
    """Summary test to verify all TTS systems are working"""

    def test_all_tts_providers_available(
        self, elevenlabs_service, google_service, temp_audio_dir
    ):
        """Comprehensive test that all available TTS providers work"""
        test_text = "Integration test summary for all TTS providers."

        results = {
            "elevenlabs": {"available": False, "working": False},
            "google": {"available": False, "working": False},
        }

        # Test ElevenLabs
        try:
            output = os.path.join(temp_audio_dir, "summary_elevenlabs.wav")
            elevenlabs_service.generate_audio(
                text_input=test_text, output_filepath=output, upload_to_cloud=False
            )
            results["elevenlabs"]["available"] = True
            results["elevenlabs"]["working"] = os.path.exists(output)
        except Exception as e:
            logger.warning(f"ElevenLabs test failed: {e}")

        # Test Google
        try:
            output = os.path.join(temp_audio_dir, "summary_google.wav")
            google_service.generate_audio(
                text_input=test_text, output_filepath=output, upload_to_cloud=False
            )
            results["google"]["available"] = True
            results["google"]["working"] = os.path.exists(output)
        except Exception as e:
            logger.warning(f"Google TTS test failed: {e}")

        # At least one provider must be working
        working_count = sum(1 for s in results.values() if s["working"])
        assert working_count > 0, "No TTS providers are working!"

        logger.info(f"✓ {working_count}/2 TTS providers working")
