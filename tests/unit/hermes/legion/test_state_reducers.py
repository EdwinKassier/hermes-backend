"""Tests for LangGraph state reducers."""

import pytest

from app.hermes.legion.state.graph_state import deep_merge_dicts, merge_dicts


class TestMergeDicts:
    """Tests for the shallow merge_dicts reducer."""

    def test_merge_empty_dicts(self):
        """Empty dicts should produce empty result."""
        result = merge_dicts({}, {})
        assert result == {}

    def test_merge_with_first_empty(self):
        """Merging into empty dict returns second dict."""
        result = merge_dicts({}, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_with_second_empty(self):
        """Merging empty into dict returns first dict."""
        result = merge_dicts({"a": 1, "b": 2}, {})
        assert result == {"a": 1, "b": 2}

    def test_merge_disjoint_keys(self):
        """Merging dicts with no overlap combines all keys."""
        result = merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_overlapping_keys(self):
        """Second dict values override first for overlapping keys."""
        result = merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_shallow_merge_replaces_nested(self):
        """Shallow merge replaces nested dicts entirely."""
        result = merge_dicts({"nested": {"a": 1, "b": 2}}, {"nested": {"c": 3}})
        # Shallow merge: nested dict is replaced, not merged
        assert result == {"nested": {"c": 3}}


class TestDeepMergeDicts:
    """Tests for the deep_merge_dicts reducer."""

    def test_deep_merge_empty_dicts(self):
        """Empty dicts should produce empty result."""
        result = deep_merge_dicts({}, {})
        assert result == {}

    def test_deep_merge_with_first_empty(self):
        """Merging into empty dict returns second dict."""
        result = deep_merge_dicts({}, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_with_second_empty(self):
        """Merging empty into dict returns first dict."""
        result = deep_merge_dicts({"a": 1, "b": 2}, {})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_disjoint_keys(self):
        """Merging dicts with no overlap combines all keys."""
        result = deep_merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_overlapping_keys(self):
        """Second dict values override first for overlapping non-dict keys."""
        result = deep_merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested_dicts(self):
        """Deep merge preserves and merges nested dicts."""
        result = deep_merge_dicts({"nested": {"a": 1, "b": 2}}, {"nested": {"c": 3}})
        # Deep merge: nested dicts are merged
        assert result == {"nested": {"a": 1, "b": 2, "c": 3}}

    def test_deep_merge_nested_override(self):
        """Deep merge overrides nested keys in conflict."""
        result = deep_merge_dicts(
            {"nested": {"a": 1, "b": 2}}, {"nested": {"b": 99, "c": 3}}
        )
        assert result == {"nested": {"a": 1, "b": 99, "c": 3}}

    def test_deep_merge_multi_level(self):
        """Deep merge works for deeply nested structures."""
        result = deep_merge_dicts(
            {"l1": {"l2": {"l3": {"a": 1}}}}, {"l1": {"l2": {"l3": {"b": 2}}}}
        )
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}

    def test_deep_merge_mixed_types(self):
        """Deep merge replaces dict with non-dict and vice versa."""
        # Dict replaced by non-dict
        result = deep_merge_dicts({"key": {"nested": 1}}, {"key": "string_value"})
        assert result == {"key": "string_value"}

        # Non-dict replaced by dict
        result = deep_merge_dicts({"key": "string_value"}, {"key": {"nested": 1}})
        assert result == {"key": {"nested": 1}}

    def test_deep_merge_preserves_original(self):
        """Deep merge does not mutate original dictionaries."""
        original_a = {"nested": {"a": 1}}
        original_b = {"nested": {"b": 2}}

        result = deep_merge_dicts(original_a, original_b)

        # Original dicts unchanged
        assert original_a == {"nested": {"a": 1}}
        assert original_b == {"nested": {"b": 2}}
        # Result is merged
        assert result == {"nested": {"a": 1, "b": 2}}

    def test_deep_merge_worker_results(self):
        """Simulate parallel worker result merging."""
        worker1_result = {
            "worker_1": {
                "result": "Research findings...",
                "status": "success",
            }
        }
        worker2_result = {
            "worker_2": {
                "result": "Analysis results...",
                "status": "success",
            }
        }
        worker3_result = {
            "worker_3": {
                "result": "Code implementation...",
                "status": "success",
            }
        }

        # Simulate sequential merges as workers complete
        merged = deep_merge_dicts({}, worker1_result)
        merged = deep_merge_dicts(merged, worker2_result)
        merged = deep_merge_dicts(merged, worker3_result)

        assert len(merged) == 3
        assert "worker_1" in merged
        assert "worker_2" in merged
        assert "worker_3" in merged
        assert merged["worker_1"]["status"] == "success"
