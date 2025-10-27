"""
Unit tests for TTS error handling in Hermes service.
Tests specifically designed to catch production errors from Cloud Run logs.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.hermes.services import HermesService
from app.hermes.models import UserIdentity, ResponseMode
from app.hermes.exceptions import TTSServiceError


class TestTTSCloudURLErrorHandling:
    """Tests for handling missing cloud_url in TTS responses"""
    
    def test_tts_result_without_cloud_url(self):
        """
        Test that service handles TTS result without cloud_url key.
        
        This catches the production error:
        ERROR: TTS generation failed: 'cloud_url'
        
        When cloud upload is disabled or fails, the TTS service returns
        a dict without the 'cloud_url' key. The service should handle
        this gracefully by returning None for cloud_url.
        """
        service = HermesService()
        
        # Mock TTS service to return result without cloud_url
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.return_value = {
            'local_path': '/tmp/audio.mp3',
            'sample_rate': 44100,
            'audio_format': 'mp3'
            # NOTE: No 'cloud_url' key!
        }
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        # Should return (None, provider) without error
        cloud_url, provider = service.generate_tts("Test text")
        assert cloud_url is None
        assert provider == 'elevenlabs'
    
    def test_tts_result_with_none_cloud_url(self):
        """
        Test that service handles TTS result with cloud_url=None.
        
        When cloud upload is disabled, cloud_url is explicitly set to None.
        """
        service = HermesService()
        
        # Mock TTS service to return result with None cloud_url
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.return_value = {
            'local_path': '/tmp/audio.mp3',
            'sample_rate': 44100,
            'audio_format': 'mp3',
            'cloud_url': None  # Explicitly None
        }
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        # Should return (None, 'elevenlabs') without error
        cloud_url, provider = service.generate_tts("Test text")
        assert cloud_url is None
        assert provider == 'elevenlabs'
    
    def test_process_request_tts_mode_without_cloud_url(self):
        """
        Test full process_request flow with TTS mode when cloud_url is missing.
        
        This simulates the exact production scenario from Cloud Run logs:
        - Request with response_mode=tts
        - TTS generation succeeds (audio file created)
        - But cloud_url is missing from result
        - Request should succeed with audio_url=None
        """
        service = HermesService()
        
        # Mock Gemini service
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response_with_rag.return_value = "AI response text"
        service._gemini_service = mock_gemini
        
        # Mock TTS service - returns result without cloud_url
        mock_tts = Mock()
        mock_tts.generate_audio.return_value = {
            'local_path': '/tmp/tmpvxixpzo8.mp3',  # From actual logs
            'sample_rate': 44100,
            'audio_format': 'mp3'
            # Missing 'cloud_url' - now handled gracefully
        }
        mock_tts.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts
        
        user_identity = UserIdentity(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        # Should succeed with audio_url=None (not crash with KeyError)
        result = service.process_request(
            text="Where did he go to school",  # From actual logs
            user_identity=user_identity,
            response_mode=ResponseMode.TTS,
            persona="hermes"
        )
        
        # Verify result
        assert result.audio_url is None  # Cloud URL not available
        assert result.tts_provider == 'elevenlabs'
        assert result.response_mode == ResponseMode.TTS
    
    def test_tts_result_with_valid_cloud_url(self):
        """Test normal case where cloud_url is present and valid"""
        service = HermesService()
        
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.return_value = {
            'local_path': '/tmp/audio.mp3',
            'sample_rate': 44100,
            'audio_format': 'mp3',
            'cloud_url': 'https://storage.googleapis.com/bucket/audio.mp3'
        }
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        # Should return (cloud_url, provider) successfully
        cloud_url, provider = service.generate_tts("Test text")
        assert cloud_url == 'https://storage.googleapis.com/bucket/audio.mp3'
        assert provider == 'elevenlabs'
    
    def test_process_request_tts_mode_with_valid_cloud_url(self):
        """Test full process_request flow with TTS mode when cloud_url is valid"""
        service = HermesService()
        
        # Mock Gemini service
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response_with_rag.return_value = "AI response"
        service._gemini_service = mock_gemini
        
        # Mock TTS service - returns complete result
        mock_tts = Mock()
        mock_tts.generate_audio.return_value = {
            'local_path': '/tmp/audio.mp3',
            'sample_rate': 44100,
            'audio_format': 'mp3',
            'cloud_url': 'https://storage.googleapis.com/bucket/audio.mp3'
        }
        mock_tts.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts
        
        user_identity = UserIdentity(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        result = service.process_request(
            text="Test text",
            user_identity=user_identity,
            response_mode=ResponseMode.TTS,
            persona="hermes"
        )
        
        # Should succeed with audio_url
        assert result.audio_url == 'https://storage.googleapis.com/bucket/audio.mp3'
        assert result.tts_provider == 'elevenlabs'
        assert result.response_mode == ResponseMode.TTS


class TestTTSServiceErrorScenarios:
    """Test various TTS service error scenarios"""
    
    def test_tts_service_returns_invalid_dict(self):
        """Test handling of completely invalid TTS response"""
        service = HermesService()
        
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.return_value = {}  # Empty dict
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        # Empty dict is valid (just no cloud_url)
        cloud_url, provider = service.generate_tts("Test text")
        assert cloud_url is None
        assert provider == 'elevenlabs'
    
    def test_tts_service_returns_none(self):
        """Test handling when TTS returns None"""
        service = HermesService()
        
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.return_value = None
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        with pytest.raises(TTSServiceError):
            service.generate_tts("Test text")
    
    def test_tts_service_raises_exception(self):
        """Test handling when TTS service raises exception"""
        service = HermesService()
        
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.side_effect = TTSServiceError("TTS failed")
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        with pytest.raises(TTSServiceError) as exc_info:
            service.generate_tts("Test text")
        
        assert "TTS failed" in str(exc_info.value)
    
    def test_tts_with_missing_local_path(self):
        """Test TTS response without local_path"""
        service = HermesService()
        
        mock_tts_service = Mock()
        mock_tts_service.generate_audio.return_value = {
            'sample_rate': 44100,
            'cloud_url': 'https://example.com/audio.mp3'
            # Missing 'local_path'
        }
        mock_tts_service.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts_service
        
        # Should still work if cloud_url is present
        cloud_url, provider = service.generate_tts("Test text")
        assert cloud_url == 'https://example.com/audio.mp3'


class TestProductionLogErrors:
    """
    Tests specifically designed to catch errors from production Cloud Run logs.
    
    Based on actual log entries:
    - ERROR:app.hermes.services:TTS generation failed: 'cloud_url'
    - ERROR:app.hermes.services:Service error processing request: Failed to generate audio: 'cloud_url'
    - ERROR:hermes:Hermes error: Failed to generate audio: 'cloud_url'
    """
    
    def test_cloud_run_tts_error_scenario(self):
        """
        Reproduce exact error from Cloud Run logs dated 2025-10-26T20:28:35.
        
        Context from logs:
        - Request: GET /api/v1/hermes/process_request?request_text=Where+did+he+go+to+school&response_mode=tts&persona=hermes
        - TTS generated audio: 110804 bytes at /tmp/tmpvxixpzo8.mp3
        - Error: TTS generation failed: 'cloud_url'
        - HTTP response: 400
        
        The error was caused by accessing cloud_url with dict[] instead of .get().
        After fix, this should succeed with audio_url=None.
        """
        service = HermesService()
        
        # Mock successful Gemini response
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response_with_rag.return_value = (
            "Edwin Kassier attended the University of Cape Town..."
        )
        service._gemini_service = mock_gemini
        
        # Mock TTS - audio generated but no cloud_url (production scenario)
        mock_tts = Mock()
        mock_tts.generate_audio.return_value = {
            'local_path': '/tmp/tmpvxixpzo8.mp3',
            'sample_rate': 44100,
            'audio_format': 'mp3'
            # cloud_url is missing - this was causing KeyError
        }
        mock_tts.tts_provider = 'elevenlabs'
        service._tts_service = mock_tts
        
        user_identity = UserIdentity(
            user_id="anonymous",
            session_id="prod_session",
            ip_address="169.254.169.126",
            user_agent="Portfolio-Chat-Proxy"
        )
        
        # After fix, should succeed (not raise KeyError!)
        result = service.process_request(
            text="Where did he go to school",
            user_identity=user_identity,
            response_mode=ResponseMode.TTS,
            persona="hermes"
        )
        
        # Verify result
        assert result.audio_url is None  # Cloud URL not available
        assert result.tts_provider == 'elevenlabs'
        assert result.message == "Edwin Kassier attended the University of Cape Town..."
    
    def test_cloud_run_environment_without_cloud_storage(self):
        """
        Test scenario where Cloud Run environment has no cloud storage configured.
        
        In Cloud Run, if CloudStorageService is not initialized or
        upload_to_cloud defaults to False, cloud_url won't be in response.
        """
        service = HermesService()
        
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response_with_rag.return_value = "Response"
        service._gemini_service = mock_gemini
        
        # TTS works but no cloud storage available
        mock_tts = Mock()
        mock_tts.generate_audio.return_value = {
            'local_path': '/tmp/audio.mp3',
            'sample_rate': 24000,
            'audio_format': 'wav',
            'cloud_url': None  # Cloud storage not configured
        }
        mock_tts.tts_provider = 'google'
        service._tts_service = mock_tts
        
        user_identity = UserIdentity(
            user_id="test",
            session_id="test",
            ip_address="127.0.0.1",
            user_agent="test"
        )
        
        # Should handle None cloud_url gracefully
        cloud_url, provider = service.generate_tts("Test")
        assert cloud_url is None
        assert provider == 'google'

