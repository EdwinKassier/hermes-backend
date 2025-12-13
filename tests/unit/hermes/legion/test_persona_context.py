"""Unit tests for LegionPersonaContext."""

import asyncio

from app.hermes.legion.utils.persona_context import (
    LegionPersonaContext,
    get_current_legion_persona,
    with_legion_persona,
)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestLegionPersonaContext:
    """Test cases for LegionPersonaContext."""

    def test_context_default_value(self):
        """Test that default persona is 'legion' when no context is set."""
        # Should return default when no context is active
        assert get_current_legion_persona() == "legion"

    def test_context_basic_usage(self):
        """Test basic context manager usage."""
        # Outside context
        assert get_current_legion_persona() == "legion"

        # Inside context
        with LegionPersonaContext("test_persona"):
            assert get_current_legion_persona() == "test_persona"

        # Back outside context
        assert get_current_legion_persona() == "legion"

    def test_context_nesting(self):
        """Test nested contexts."""
        with LegionPersonaContext("outer_persona"):
            assert get_current_legion_persona() == "outer_persona"

            with LegionPersonaContext("inner_persona"):
                assert get_current_legion_persona() == "inner_persona"

            # Should restore outer context
            assert get_current_legion_persona() == "outer_persona"

    def test_context_exception_handling(self):
        """Test context cleanup on exceptions."""
        try:
            with LegionPersonaContext("exception_persona"):
                assert get_current_legion_persona() == "exception_persona"
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Context should be cleaned up despite exception
        assert get_current_legion_persona() == "legion"

    def test_async_context(self):
        """Test context manager in async functions."""
        # Outside context
        assert get_current_legion_persona() == "legion"

        # Inside async context
        async def test_async():
            with LegionPersonaContext("async_persona"):
                await asyncio.sleep(0.01)  # Simulate async work
                assert get_current_legion_persona() == "async_persona"
                return "done"

        result = run_async_test(test_async())
        assert result == "done"

        # Context should be restored
        assert get_current_legion_persona() == "legion"

    def test_concurrent_contexts(self):
        """Test that contexts are isolated between concurrent tasks."""
        results = []

        async def task1():
            with LegionPersonaContext("persona1"):
                await asyncio.sleep(0.01)
                results.append(("task1", get_current_legion_persona()))
            results.append(("task1_done", get_current_legion_persona()))

        async def task2():
            with LegionPersonaContext("persona2"):
                await asyncio.sleep(0.01)
                results.append(("task2", get_current_legion_persona()))
            results.append(("task2_done", get_current_legion_persona()))

        run_async_test(asyncio.gather(task1(), task2()))

        # Check results - each task should see its own persona
        task_results = {key: value for key, value in results}
        assert task_results["task1"] == "persona1"
        assert task_results["task2"] == "persona2"
        assert task_results["task1_done"] == "legion"
        assert task_results["task2_done"] == "legion"

    def test_context_manager_as_decorator(self):
        """Test the with_legion_persona decorator."""

        @with_legion_persona("decorated_persona")
        async def decorated_function():
            return get_current_legion_persona()

        # Run in event loop
        result = run_async_test(decorated_function())
        assert result == "decorated_persona"

        # Context should be restored after function completes
        assert get_current_legion_persona() == "legion"

    def test_context_reentrancy(self):
        """Test that the same context can be entered multiple times."""
        with LegionPersonaContext("reusable_persona"):
            assert get_current_legion_persona() == "reusable_persona"

            # Enter again with same persona
            with LegionPersonaContext("reusable_persona"):
                assert get_current_legion_persona() == "reusable_persona"

            # Still same persona
            assert get_current_legion_persona() == "reusable_persona"

    def test_context_different_instances(self):
        """Test that different context instances work correctly."""
        context1 = LegionPersonaContext("persona_a")
        context2 = LegionPersonaContext("persona_b")

        with context1:
            assert get_current_legion_persona() == "persona_a"

            with context2:
                assert get_current_legion_persona() == "persona_b"

            assert get_current_legion_persona() == "persona_a"

    def test_context_empty_string(self):
        """Test context with empty string (edge case)."""
        with LegionPersonaContext(""):
            assert get_current_legion_persona() == ""

        assert get_current_legion_persona() == "legion"

    def test_context_none_value(self):
        """Test that None gets converted to default."""
        # This tests the internal behavior - None should not be returned
        # The context variable defaults to None, but our getter handles it
        assert get_current_legion_persona() == "legion"
