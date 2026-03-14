"""Unit tests for natural_sort.py.

Frame sequences are named with numeric suffixes (frame_1, frame_10,
frame_100). Lexicographic sort produces the wrong order (frame_1,
frame_10, frame_100 becomes frame_1, frame_100, frame_10). natural_sort_key
fixes this. A regression here would silently process frames out of order,
corrupting the output.
"""

from __future__ import annotations

from corridorkey.natural_sort import natsorted, natural_sort_key


class TestNaturalSortKey:
    """natural_sort_key - numeric token extraction and ordering."""

    def test_numeric_order(self):
        """Numeric suffixes must sort by value, not lexicographically."""
        items = ["frame_10", "frame_2", "frame_1"]
        assert sorted(items, key=natural_sort_key) == ["frame_1", "frame_2", "frame_10"]

    def test_zero_padded_and_unpadded_mixed(self):
        """Zero-padded and unpadded numbers must sort by numeric value."""
        items = ["frame_002", "frame_10", "frame_1"]
        result = sorted(items, key=natural_sort_key)
        assert result == ["frame_1", "frame_002", "frame_10"]

    def test_no_digits_falls_back_to_alpha(self):
        """Strings with no digits must fall back to alphabetical order."""
        items = ["charlie", "alpha", "bravo"]
        assert sorted(items, key=natural_sort_key) == ["alpha", "bravo", "charlie"]

    def test_case_insensitive(self):
        """Sort must be case-insensitive so Frame_2 and frame_1 compare correctly."""
        items = ["Frame_2", "frame_1"]
        assert sorted(items, key=natural_sort_key) == ["frame_1", "Frame_2"]

    def test_empty_string(self):
        """An empty string must produce a single-element key list without crashing."""
        assert natural_sort_key("") == [""]

    def test_digits_only(self):
        """A digits-only string must produce an int token flanked by empty strings."""
        assert natural_sort_key("42") == ["", 42, ""]

    def test_mixed_prefix(self):
        """Multiple numeric tokens in a name must all be compared numerically."""
        items = ["shot_10_v2", "shot_2_v10", "shot_2_v2"]
        result = sorted(items, key=natural_sort_key)
        assert result == ["shot_2_v2", "shot_2_v10", "shot_10_v2"]


class TestNatsorted:
    """natsorted - convenience wrapper that returns a new sorted list."""

    def test_returns_sorted_copy(self):
        """natsorted() must return a new list in natural order without mutating the original."""
        original = ["f_10", "f_2", "f_1"]
        result = natsorted(original)
        assert result == ["f_1", "f_2", "f_10"]
        assert original == ["f_10", "f_2", "f_1"]  # original unchanged

    def test_empty_list(self):
        """An empty input must return an empty list."""
        assert natsorted([]) == []

    def test_single_item(self):
        """A single-item list must be returned unchanged."""
        assert natsorted(["only"]) == ["only"]

    def test_already_sorted(self):
        """An already-sorted list must be returned in the same order."""
        items = ["a_1", "a_2", "a_3"]
        assert natsorted(items) == items
