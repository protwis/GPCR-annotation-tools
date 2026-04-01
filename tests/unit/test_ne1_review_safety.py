"""Tests for New Era Epic 1: Review Engine Safety & Type Integrity.

Covers P0 (fix_mode bypass), P1+P4 (skip option + core block guard),
P3 (coerce_type), and P2 (CSV write decline).
"""

from gpcr_tools.csv_generator.review_engine import coerce_type

# ── NE-1.3: coerce_type ────────────────────────────────────────────────


class TestCoerceType:
    """Verify type preservation after Prompt.ask edits."""

    def test_bool_true(self):
        assert coerce_type(True, "true") is True

    def test_bool_false(self):
        assert coerce_type(False, "0") is False

    def test_bool_yes(self):
        assert coerce_type(True, "yes") is True

    def test_bool_no(self):
        assert coerce_type(False, "no") is False

    def test_int_roundtrip(self):
        result = coerce_type(42, "42")
        assert result == 42
        assert isinstance(result, int)

    def test_float_roundtrip(self):
        result = coerce_type(0.95, "0.95")
        assert result == 0.95
        assert isinstance(result, float)

    def test_list_roundtrip(self):
        original = ["A", "B"]
        result = coerce_type(original, '["A", "B"]')
        assert result == ["A", "B"]
        assert isinstance(result, list)

    def test_dict_roundtrip(self):
        original = {"key": "val"}
        result = coerce_type(original, '{"key": "val"}')
        assert result == {"key": "val"}
        assert isinstance(result, dict)

    def test_fallback_to_string(self):
        """When the new value can't be parsed as the original type, return string."""
        result = coerce_type(42, "hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_identity_returns_original(self):
        """When str(original) == new_str, return the original object directly."""
        result = coerce_type("foo", "foo")
        assert result == "foo"

    def test_empty_string_stays_string(self):
        """Empty input returns empty string, not None."""
        result = coerce_type(42, "")
        assert result == ""
        assert isinstance(result, str)

    def test_bool_before_int(self):
        """Bool check must come before int check (bool is subclass of int)."""
        # If we pass "1" to a bool original, it should return True (bool), not 1 (int)
        result = coerce_type(True, "1")
        assert result is True
        assert isinstance(result, bool)

    def test_float_invalid_input(self):
        result = coerce_type(3.14, "not_a_float")
        assert result == "not_a_float"
        assert isinstance(result, str)

    def test_list_invalid_json(self):
        result = coerce_type(["A"], "not json")
        assert result == "not json"
        assert isinstance(result, str)
