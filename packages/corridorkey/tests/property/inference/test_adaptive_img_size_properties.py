"""Property-based tests for adaptive_img_size — output is always a valid img_size."""

from __future__ import annotations

from corridorkey.stages.inference.config import VALID_IMG_SIZES, adaptive_img_size
from hypothesis import given
from hypothesis import strategies as st


class TestAdaptiveImgSizeProperties:
    @given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_output_always_valid_img_size(self, vram_gb: float):
        """adaptive_img_size always returns a value from VALID_IMG_SIZES (excluding 0)."""
        result = adaptive_img_size(vram_gb)
        assert result in VALID_IMG_SIZES
        assert result > 0  # 0 means auto-select — adaptive_img_size must return a concrete size

    @given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_output_is_positive_int(self, vram_gb: float):
        result = adaptive_img_size(vram_gb)
        assert isinstance(result, int)
        assert result > 0

    @given(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_monotone_more_vram_not_smaller(self, a: float, b: float):
        """More VRAM should never produce a smaller img_size than less VRAM."""
        low, high = min(a, b), max(a, b)
        assert adaptive_img_size(low) <= adaptive_img_size(high)
