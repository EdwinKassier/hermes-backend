"""
Simple unit tests for signal handling fix.

These tests verify that the signal registration fix works without requiring Redis.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from app.prism.services import PrismService, get_prism_service

pytestmark = pytest.mark.redis


class TestSignalHandlingSimple:
    """Test signal handling in multi-threaded contexts without Redis dependency."""

    def test_prism_service_instantiation_without_redis(self):
        """Test that PrismService can be instantiated without Redis (mocked)."""
        with patch("app.prism.services.RedisSessionStore") as mock_redis:
            mock_redis.return_value = MagicMock()

            # This should work without signal registration errors
            service = PrismService()
            assert service is not None
            assert hasattr(service, "executor")
            service.cleanup()

    def test_prism_service_instantiation_from_worker_thread_without_redis(self):
        """Test that PrismService can be instantiated from worker thread without Redis."""
        result = {}
        exception = None

        def worker_thread():
            nonlocal exception
            try:
                with patch("app.prism.services.RedisSessionStore") as mock_redis:
                    mock_redis.return_value = MagicMock()
                    # This should work without signal registration errors
                    service = PrismService()
                    result["service"] = service
                    result["success"] = True
                    service.cleanup()
            except Exception as e:
                exception = e
                result["success"] = False

        # Start worker thread
        thread = threading.Thread(target=worker_thread)
        thread.start()
        thread.join(timeout=5)

        # Verify no exception occurred
        assert exception is None, f"Exception in worker thread: {exception}"
        assert result.get(
            "success", False
        ), "Service instantiation failed in worker thread"
        assert "service" in result

    def test_get_prism_service_from_worker_thread_without_redis(self):
        """Test that get_prism_service works from worker thread without Redis."""
        result = {}
        exception = None

        def worker_thread():
            nonlocal exception
            try:
                with patch("app.prism.services.RedisSessionStore") as mock_redis:
                    mock_redis.return_value = MagicMock()
                    # This should work without signal registration errors
                    service = get_prism_service()
                    result["service"] = service
                    result["success"] = True
            except Exception as e:
                exception = e
                result["success"] = False

        # Start worker thread
        thread = threading.Thread(target=worker_thread)
        thread.start()
        thread.join(timeout=5)

        # Verify no exception occurred
        assert exception is None, f"Exception in worker thread: {exception}"
        assert result.get("success", False), "get_prism_service failed in worker thread"
        assert "service" in result

    def test_signal_registration_not_called_in_worker_thread(self):
        """Test that signal registration is not attempted in worker threads."""
        signal_calls = []

        def mock_signal(signum, handler):
            signal_calls.append((signum, handler))
            raise ValueError("signal only works in main thread of the main interpreter")

        with patch("signal.signal", side_effect=mock_signal):
            result = {}
            exception = None

            def worker_thread():
                nonlocal exception
                try:
                    with patch("app.prism.services.RedisSessionStore") as mock_redis:
                        mock_redis.return_value = MagicMock()
                        service = PrismService()
                        result["service"] = service
                        result["success"] = True
                        service.cleanup()
                except Exception as e:
                    exception = e
                    result["success"] = False

            # Start worker thread
            thread = threading.Thread(target=worker_thread)
            thread.start()
            thread.join(timeout=5)

            # Verify no signal registration was attempted
            assert (
                len(signal_calls) == 0
            ), f"Signal registration was attempted: {signal_calls}"
            # Verify service was created successfully despite signal mock
            assert result.get("success", False), "Service creation failed"
            assert "service" in result

    def test_gunicorn_when_ready_signal_registration(self):
        """Test that Gunicorn when_ready hook can register signals."""
        # Import the function directly from the module
        import importlib.util
        import os

        gunicorn_conf_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "gunicorn.conf.py"
        )
        spec = importlib.util.spec_from_file_location(
            "gunicorn.conf", gunicorn_conf_path
        )
        gunicorn_conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gunicorn_conf)
        when_ready = gunicorn_conf.when_ready

        # Mock server object
        mock_server = MagicMock()
        mock_server.log = MagicMock()

        # Mock threading to simulate main thread
        import threading

        main_thread = threading.main_thread()
        with patch("threading.current_thread", return_value=main_thread):
            # Mock signal registration
            with patch("signal.signal") as mock_signal:
                # Call when_ready hook
                when_ready(mock_server)

                # Verify signal registration was attempted
                assert mock_signal.call_count == 2  # SIGTERM and SIGINT
                mock_server.log.info.assert_called_with(
                    "Signal handlers registered successfully"
                )

    def test_gunicorn_when_ready_skips_signal_registration_in_worker_thread(self):
        """Test that Gunicorn when_ready skips signal registration in worker threads."""
        # Import the function directly from the module
        import importlib.util
        import os

        gunicorn_conf_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "gunicorn.conf.py"
        )
        spec = importlib.util.spec_from_file_location(
            "gunicorn.conf", gunicorn_conf_path
        )
        gunicorn_conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gunicorn_conf)
        when_ready = gunicorn_conf.when_ready

        # Mock server object
        mock_server = MagicMock()
        mock_server.log = MagicMock()

        # Mock threading to simulate worker thread
        import threading

        worker_thread = threading.Thread()
        with patch("threading.current_thread", return_value=worker_thread):
            # Mock signal registration
            with patch("signal.signal") as mock_signal:
                # Call when_ready hook
                when_ready(mock_server)

                # Verify signal registration was not attempted
                assert mock_signal.call_count == 0
                mock_server.log.warning.assert_called_with(
                    "Not in main thread, skipping signal registration"
                )

    def test_prism_service_cleanup_works(self):
        """Test that PrismService cleanup works correctly."""
        with patch("app.prism.services.RedisSessionStore") as mock_redis:
            mock_redis.return_value = MagicMock()
            service = PrismService()

            # Verify service has required attributes
            assert hasattr(service, "executor")
            assert hasattr(service, "cleanup")

            # Test cleanup
            service.cleanup()

            # Verify executor was shut down
            assert service.executor._shutdown

    def test_singleton_behavior_across_threads(self):
        """Test that get_prism_service returns the same instance across threads."""
        results = []

        def worker_thread(thread_id):
            with patch("app.prism.services.RedisSessionStore") as mock_redis:
                mock_redis.return_value = MagicMock()
                service = get_prism_service()
                results.append({"thread_id": thread_id, "service_id": id(service)})

        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)

        # Verify all threads got the same service instance
        assert len(results) == 3
        service_ids = [result["service_id"] for result in results]
        assert len(set(service_ids)) == 1, f"Different service instances: {service_ids}"

    def test_prism_service_thread_safety(self):
        """Test that PrismService operations are thread-safe."""
        with patch("app.prism.services.RedisSessionStore") as mock_redis:
            mock_redis.return_value = MagicMock()
            service = get_prism_service()
            results = []
            exceptions = []

            def worker_operation(thread_id):
                try:
                    # Test various service operations
                    service.get_session_metrics("test_session")
                    service.get_system_metrics()
                    results.append({"thread_id": thread_id, "success": True})
                except Exception as e:
                    exceptions.append({"thread_id": thread_id, "exception": e})

            # Start multiple worker threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_operation, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)

            # Verify no exceptions occurred
            assert len(exceptions) == 0, f"Exceptions in worker threads: {exceptions}"
            assert (
                len(results) == 5
            ), f"Expected 5 successful results, got {len(results)}"

            # Cleanup
            service.cleanup()
