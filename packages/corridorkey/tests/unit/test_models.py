"""Unit tests for models.py.

InOutRange is the only domain model in this package. It is persisted to
clip.json and used by the inference loop to restrict which frames are
processed. Tests verify the frame_count arithmetic, contains() boundary
conditions, and the dict serialisation roundtrip.
"""

from __future__ import annotations

import pytest
from corridorkey.models import InOutRange


class TestInOutRange:
    """InOutRange - frame_count arithmetic, contains() boundaries, and dict serialisation."""

    def test_frame_count_inclusive(self):
        """frame_count must be inclusive of both endpoints (out - in + 1)."""
        r = InOutRange(in_point=0, out_point=9)
        assert r.frame_count == 10

    def test_frame_count_single_frame(self):
        """A range where in_point == out_point must report exactly one frame."""
        r = InOutRange(in_point=5, out_point=5)
        assert r.frame_count == 1

    def test_contains_in_range(self):
        """contains() must return True for the in_point, out_point, and any frame between them."""
        r = InOutRange(in_point=10, out_point=20)
        assert r.contains(10)
        assert r.contains(15)
        assert r.contains(20)

    def test_contains_out_of_range(self):
        """contains() must return False for frames strictly outside the range."""
        r = InOutRange(in_point=10, out_point=20)
        assert not r.contains(9)
        assert not r.contains(21)

    def test_to_dict(self):
        """to_dict() must produce the exact keys and values expected by clip.json."""
        r = InOutRange(in_point=3, out_point=7)
        assert r.to_dict() == {"in_point": 3, "out_point": 7}

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) must reconstruct an identical InOutRange."""
        original = InOutRange(in_point=3, out_point=7)
        restored = InOutRange.from_dict(original.to_dict())
        assert restored.in_point == original.in_point
        assert restored.out_point == original.out_point

    def test_from_dict_missing_key_raises(self):
        """from_dict() must raise KeyError when a required key is absent."""
        with pytest.raises(KeyError):
            InOutRange.from_dict({"in_point": 0})
