"""Unit tests for corridorkey_new.infra.utils."""

from __future__ import annotations

from corridorkey_new.infra.utils import natural_sort_key


class TestNaturalSortKey:
    def test_numeric_order(self):
        names = ["frame_10.png", "frame_2.png", "frame_1.png"]
        assert sorted(names, key=natural_sort_key) == [
            "frame_1.png",
            "frame_2.png",
            "frame_10.png",
        ]

    def test_multi_number_segments(self):
        names = ["shot_1_v10.exr", "shot_1_v2.exr", "shot_2_v1.exr"]
        assert sorted(names, key=natural_sort_key) == [
            "shot_1_v2.exr",
            "shot_1_v10.exr",
            "shot_2_v1.exr",
        ]

    def test_pure_alpha_strings(self):
        names = ["banana", "apple", "cherry"]
        assert sorted(names, key=natural_sort_key) == ["apple", "banana", "cherry"]

    def test_case_insensitive(self):
        names = ["Frame_2.png", "frame_1.png"]
        assert sorted(names, key=natural_sort_key) == ["frame_1.png", "Frame_2.png"]

    def test_empty_string(self):
        assert natural_sort_key("") == [""]

    def test_only_digits(self):
        names = ["100", "20", "3"]
        assert sorted(names, key=natural_sort_key) == ["3", "20", "100"]

    def test_zero_padded_vs_unpadded(self):
        # Natural sort should treat "001" and "1" as equal numerically
        names = ["frame_001.png", "frame_1.png", "frame_010.png"]
        result = sorted(names, key=natural_sort_key)
        # frame_001 and frame_1 both sort as 1, frame_010 as 10
        assert result.index("frame_010.png") > result.index("frame_001.png")
