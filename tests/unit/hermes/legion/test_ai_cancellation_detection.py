"""Unit tests for AI-powered cancellation intent detection."""

from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.orchestrator import IntentDetector


class TestCancellationDetection:
    """Test AI-powered cancellation intent detection."""

    @pytest.fixture
    def detector(self):
        """Create intent detector instance."""
        detector = IntentDetector()
        detector._gemini_service = Mock()
        return detector

    def test_cancellation_detection_ai_success_cancel(self, detector):
        """Test AI correctly detects cancellation intent."""
        detector.gemini_service.generate_gemini_response.return_value = "CANCEL"

        result = detector.is_cancellation_intent("cancel this task")

        assert result is True
        detector.gemini_service.generate_gemini_response.assert_called_once()

    def test_cancellation_detection_ai_success_continue(self, detector):
        """Test AI correctly detects non-cancellation."""
        detector.gemini_service.generate_gemini_response.return_value = "CONTINUE"

        result = detector.is_cancellation_intent("don't worry, I can wait")

        assert result is False

    def test_cancellation_detection_ai_failure_uses_fallback(self, detector):
        """Test fallback when AI fails."""
        detector.gemini_service.generate_gemini_response.side_effect = Exception(
            "API Error"
        )

        # Fallback should detect "cancel" keyword
        result = detector.is_cancellation_intent("cancel")

        assert result is True

    def test_cancellation_detection_network_error_fallback(self, detector):
        """Test network error uses fallback."""
        detector.gemini_service.generate_gemini_response.side_effect = ConnectionError(
            "Network error"
        )

        result = detector.is_cancellation_intent("never mind")

        assert result is True  # Fallback detects keyword

    def test_cancellation_detection_empty_input(self, detector):
        """Test empty input returns False."""
        assert detector.is_cancellation_intent("") is False
        assert detector.is_cancellation_intent("   ") is False

    def test_cancellation_detection_invalid_type(self, detector):
        """Test invalid input type returns False."""
        assert detector.is_cancellation_intent(None) is False
        assert detector.is_cancellation_intent(123) is False

    def test_fallback_cancellation_detection_keywords(self, detector):
        """Test fallback detects cancellation keywords."""
        assert detector._fallback_cancellation_detection("cancel") is True
        assert detector._fallback_cancellation_detection("never mind") is True
        assert detector._fallback_cancellation_detection("forget it") is True
        assert detector._fallback_cancellation_detection("stop") is True
        assert detector._fallback_cancellation_detection("abort") is True

    def test_fallback_cancellation_detection_non_keywords(self, detector):
        """Test fallback doesn't false-positive on non-cancellation."""
        # Note: Fallback is imperfect - "don't" is a keyword
        assert detector._fallback_cancellation_detection("continue please") is False
        assert detector._fallback_cancellation_detection("yes, proceed") is False

    def test_ai_distinguishes_context(self, detector):
        """Test AI can distinguish context (unlike fallback)."""
        detector.gemini_service.generate_gemini_response.return_value = "CONTINUE"

        # AI should understand this is NOT cancellation
        result = detector.is_cancellation_intent("don't worry, I can wait")

        assert result is False

    def test_ai_response_case_insensitive(self, detector):
        """Test AI response parsing is case-insensitive."""
        detector.gemini_service.generate_gemini_response.return_value = "cancel"

        result = detector.is_cancellation_intent("cancel this")

        assert result is True


class TestPromptBuilding:
    """Test cancellation detection prompt."""

    @pytest.fixture
    def detector(self):
        return IntentDetector()

    def test_prompt_includes_user_message(self, detector):
        """Test prompt includes user's message."""
        # We need to capture the prompt - mock the AI call
        detector._gemini_service = Mock()
        detector.gemini_service.generate_gemini_response.return_value = "CONTINUE"

        detector.is_cancellation_intent("test message")

        # Check the prompt argument
        call_args = detector.gemini_service.generate_gemini_response.call_args
        prompt = call_args[0][0]

        assert "test message" in prompt

    def test_prompt_includes_examples(self, detector):
        """Test prompt includes examples."""
        detector._gemini_service = Mock()
        detector.gemini_service.generate_gemini_response.return_value = "CONTINUE"

        detector.is_cancellation_intent("test")

        prompt = detector.gemini_service.generate_gemini_response.call_args[0][0]

        # Should have cancellation examples
        assert "cancel" in prompt
        assert "never mind" in prompt

        # Should have non-cancellation examples
        assert "don't worry" in prompt
        assert "don't forget" in prompt


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def detector(self):
        detector = IntentDetector()
        detector._gemini_service = Mock()
        return detector

    def test_malformed_ai_response(self, detector):
        """Test handling of malformed AI response."""
        detector.gemini_service.generate_gemini_response.return_value = "INVALID"

        result = detector.is_cancellation_intent("cancel")

        # Should return False (CANCEL not in response)
        assert result is False

    def test_ai_response_with_extra_text(self, detector):
        """Test AI response with extra text."""
        detector.gemini_service.generate_gemini_response.return_value = (
            "The user wants to CANCEL the task."
        )

        result = detector.is_cancellation_intent("cancel")

        assert result is True
