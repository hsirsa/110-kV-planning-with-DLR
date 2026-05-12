import math

import pytest

from dlr.network import first_valid_value, safe_name


class TestSafeName:
    def test_none_returns_fallback(self):
        assert safe_name(None, "fallback") == "fallback"

    def test_blank_string_returns_fallback(self):
        assert safe_name("   ", "fallback") == "fallback"

    def test_empty_string_returns_fallback(self):
        assert safe_name("", "fallback") == "fallback"

    def test_strips_whitespace(self):
        assert safe_name("  bus_1  ", "fb") == "bus_1"

    def test_non_string_is_converted(self):
        assert safe_name(42, "fb") == "42"

    def test_valid_string_returned_unchanged(self):
        assert safe_name("HV1 Bus 5", "fb") == "HV1 Bus 5"


class TestFirstValidValue:
    def test_skips_none(self):
        assert first_valid_value(None, None, 5.0) == 5.0

    def test_skips_nan(self):
        assert first_valid_value(float("nan"), 3.0) == pytest.approx(3.0)

    def test_returns_first_non_none(self):
        assert first_valid_value(1.0, 2.0) == pytest.approx(1.0)

    def test_all_none_returns_none(self):
        assert first_valid_value(None, None) is None

    def test_zero_is_valid(self):
        assert first_valid_value(None, 0.0) == pytest.approx(0.0)

    def test_false_is_valid(self):
        assert first_valid_value(None, False) is False

    def test_empty_string_is_valid(self):
        # Only None and float NaN are skipped — empty string is a valid value
        assert first_valid_value(None, "") == ""

    def test_integer_returned(self):
        assert first_valid_value(None, 7) == 7
