"""
Integration tests for Prism session creation.

These tests verify that Prism sessions can be created without signal registration errors,
particularly in multi-threaded environments like Cloud Run.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.prism.services import PrismService, get_prism_service
from app.prism.session_store import PrismSession


class TestPrismSessionCreation:
    """Test Prism session creation in various contexts."""

    def test_prism_service_instantiation_integration(self):
        """Test that PrismService can be instantiated without signal errors."""
        # This should work without any signal registration errors
        service = PrismService()

        # Verify service is properly initialized
        assert service is not None
        assert hasattr(service, "executor")
        assert hasattr(service, "attendee_client")
        assert hasattr(service, "audio_processor")
        assert hasattr(service, "identity_service")

        # Test basic functionality
        system_metrics = service.get_system_metrics()
        assert isinstance(system_metrics, dict)
        assert "sessions" in system_metrics

        # Cleanup
        service.cleanup()

    def test_get_prism_service_singleton_integration(self):
        """Test that get_prism_service returns singleton without signal errors."""
        # First call should create the service
        service1 = get_prism_service()
        assert service1 is not None

        # Second call should return the same instance
        service2 = get_prism_service()
        assert service2 is service1

        # Verify service is functional
        health_metrics = service1.get_health_metrics()
        assert isinstance(health_metrics, dict)

    def test_prism_session_creation_integration(self):
        """Test that PrismSession can be created without signal errors."""
        service = get_prism_service()

        # Create a test session
        session_id = "test_session_integration"
        session = PrismSession(
            session_id=session_id,
            user_id="test_user",
            meeting_url="https://example.com/meeting",
        )

        # Verify session is properly initialized
        assert session.session_id == session_id
        assert session.user_id == "test_user"
        assert session.status.value == "created"

        # Test session operations
        from datetime import datetime

        from app.prism.models import TranscriptEntry

        transcript = TranscriptEntry(
            speaker="user",
            text="Hello, this is a test transcript",
            timestamp=datetime.utcnow(),
        )
        session.transcript_history.append(transcript)
        assert len(session.transcript_history) == 1

        session.status = "closed"
        assert session.status.value == "closed"

    def test_multiple_threads_create_sessions(self):
        """Test that multiple threads can create sessions simultaneously."""
        service = get_prism_service()
        results = []
        exceptions = []

        def create_session(thread_id):
            try:
                session_id = f"test_session_{thread_id}"
                session = PrismSession(
                    session_id=session_id,
                    user_id=f"user_{thread_id}",
                    meeting_url="https://example.com/meeting",
                )
                results.append(
                    {"thread_id": thread_id, "session_id": session_id, "success": True}
                )
            except Exception as e:
                exceptions.append({"thread_id": thread_id, "exception": e})

        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_session, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions in worker threads: {exceptions}"
        assert len(results) == 5, f"Expected 5 successful results, got {len(results)}"

        # Verify all sessions were created successfully
        for result in results:
            assert result["success"], f"Thread {result['thread_id']} failed"

    def test_prism_service_operations_thread_safety(self):
        """Test that PrismService operations are thread-safe."""
        service = get_prism_service()
        results = []
        exceptions = []

        def perform_operations(thread_id):
            try:
                # Test various service operations
                system_metrics = service.get_system_metrics()
                assert isinstance(system_metrics, dict)

                # Test session creation
                session_id = f"test_session_{thread_id}"
                session = PrismSession(
                    session_id=session_id,
                    user_id=f"user_{thread_id}",
                    meeting_url="https://example.com/meeting",
                )

                results.append({"thread_id": thread_id, "success": True})
            except Exception as e:
                exceptions.append({"thread_id": thread_id, "exception": e})

        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=perform_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions in worker threads: {exceptions}"
        assert len(results) == 5, f"Expected 5 successful results, got {len(results)}"

    def test_signal_registration_error_simulation(self):
        """Test that the service works even when signal registration would fail."""
        # This test simulates the original error condition
        signal_calls = []

        def mock_signal(signum, handler):
            signal_calls.append((signum, handler))
            raise ValueError("signal only works in main thread of the main interpreter")

        with patch("signal.signal", side_effect=mock_signal):
            # Service should still work even if signal registration fails
            service = PrismService()
            assert service is not None

            # Verify no signal registration was attempted
            assert (
                len(signal_calls) == 0
            ), f"Signal registration was attempted: {signal_calls}"

            # Test service functionality
            system_metrics = service.get_system_metrics()
            assert isinstance(system_metrics, dict)

            service.cleanup()

    def test_gunicorn_worker_context_simulation(self):
        """Test that the service works in a simulated Gunicorn worker context."""
        # Simulate the conditions that caused the original error
        results = []
        exceptions = []

        def simulate_worker_thread():
            try:
                # This simulates what happens when a request comes in
                # and get_prism_service() is called from a worker thread
                service = get_prism_service()

                # Test that the service is functional
                system_metrics = service.get_system_metrics()
                assert isinstance(system_metrics, dict)

                # Test session creation
                session = PrismSession(
                    session_id="worker_test_session",
                    user_id="worker_user",
                    meeting_url="https://example.com/meeting",
                )

                results.append({"success": True})

            except Exception as e:
                exceptions.append({"exception": e})

        # Start worker thread (simulating Gunicorn worker)
        thread = threading.Thread(target=simulate_worker_thread)
        thread.start()
        thread.join(timeout=10)

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exception in worker thread: {exceptions}"
        assert len(results) == 1, f"Expected 1 successful result, got {len(results)}"
        assert results[0]["success"]

    def test_cleanup_after_signal_error_simulation(self):
        """Test that cleanup works even after signal registration errors."""
        service = PrismService()

        # Verify service is functional
        assert service is not None
        assert hasattr(service, "executor")

        # Test cleanup
        service.cleanup()

        # Verify executor was shut down
        assert service.executor._shutdown

    def test_concurrent_service_instantiation(self):
        """Test concurrent service instantiation from multiple threads."""
        results = []
        exceptions = []

        def instantiate_service(thread_id):
            try:
                service = PrismService()
                results.append(
                    {"thread_id": thread_id, "service_id": id(service), "success": True}
                )
                service.cleanup()
            except Exception as e:
                exceptions.append({"thread_id": thread_id, "exception": e})

        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=instantiate_service, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions in worker threads: {exceptions}"
        assert len(results) == 3, f"Expected 3 successful results, got {len(results)}"

        # Verify all services were created successfully
        for result in results:
            assert result["success"], f"Thread {result['thread_id']} failed"
            assert result["service_id"] is not None
