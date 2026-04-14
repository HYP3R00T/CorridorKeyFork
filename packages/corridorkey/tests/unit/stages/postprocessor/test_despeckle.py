"""Unit tests for corridorkey.stages.postprocessor.despeckle."""

from __future__ import annotations

import numpy as np
from corridorkey.stages.postprocessor.despeckle import despeckle_alpha


def _alpha_with_speck(h: int = 128, w: int = 128) -> np.ndarray:
    """Large foreground region with a small isolated speck far from it."""
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    # Large region in the bottom-right — should survive
    alpha[60:110, 60:110, 0] = 1.0
    # Tiny speck (2×2) in the top-left corner, far from the large region
    alpha[2:4, 2:4, 0] = 1.0
    return alpha


class TestDespeckleAlpha:
    def test_zero_min_area_returns_unchanged(self):
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=0)
        assert np.array_equal(result, alpha)

    def test_removes_small_region(self):
        alpha = _alpha_with_speck()
        # min_area larger than the 2×2 speck (4 px) but smaller than the large region
        result = despeckle_alpha(alpha, min_area=10)
        # Speck area should be zeroed out
        assert result[2:4, 2:4, 0].max() == 0.0

    def test_preserves_large_region(self):
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10)
        # Centre of the large region should still be non-zero
        assert result[85, 85, 0] > 0.0

    def test_output_shape_preserved(self):
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10)
        assert result.shape == alpha.shape

    def test_output_dtype_float32(self):
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10)
        assert result.dtype == np.float32

    def test_all_zeros_unchanged(self):
        alpha = np.zeros((32, 32, 1), dtype=np.float32)
        result = despeckle_alpha(alpha, min_area=10)
        assert np.array_equal(result, alpha)
