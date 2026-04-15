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
        """min_area=0 is the early-return guard — input is returned as-is."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=0)
        assert np.array_equal(result, alpha)

    def test_negative_min_area_returns_unchanged(self):
        """Negative min_area also triggers the early-return guard."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=-1)
        assert np.array_equal(result, alpha)

    def test_removes_small_region(self):
        """A speck smaller than min_area is zeroed out."""
        alpha = _alpha_with_speck()
        # min_area larger than the 2×2 speck (4 px) but smaller than the large region
        result = despeckle_alpha(alpha, min_area=10)
        assert result[2:4, 2:4, 0].max() == 0.0

    def test_preserves_large_region(self):
        """A region larger than min_area is kept."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10)
        assert result[85, 85, 0] > 0.0

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10)
        assert result.shape == alpha.shape

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10)
        assert result.dtype == np.float32

    def test_all_zeros_unchanged(self):
        """All-zero alpha produces all-zero output."""
        alpha = np.zeros((32, 32, 1), dtype=np.float32)
        result = despeckle_alpha(alpha, min_area=10)
        assert np.array_equal(result, alpha)

    def test_dilation_zero_skips_dilation(self):
        """dilation=0 skips the dilation step without error."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10, dilation=0)
        assert result.shape == alpha.shape
        assert result.dtype == np.float32

    def test_blur_size_zero_skips_blur(self):
        """blur_size=0 skips the Gaussian blur step without error."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10, blur_size=0)
        assert result.shape == alpha.shape
        assert result.dtype == np.float32

    def test_2d_alpha_input_accepted(self):
        """2D alpha [H, W] (no channel dim) is handled correctly."""
        alpha_2d = np.zeros((64, 64), dtype=np.float32)
        alpha_2d[30:50, 30:50] = 1.0
        alpha_2d[2:4, 2:4] = 1.0
        result = despeckle_alpha(alpha_2d, min_area=10, dilation=0, blur_size=0)
        assert result.shape == (64, 64, 1)

    def test_speck_removed_no_dilation_no_blur(self):
        """Speck is removed even when dilation=0 and blur_size=0."""
        alpha = _alpha_with_speck()
        result = despeckle_alpha(alpha, min_area=10, dilation=0, blur_size=0)
        assert result[2:4, 2:4, 0].max() == 0.0
