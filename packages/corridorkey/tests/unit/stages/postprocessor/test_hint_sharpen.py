"""Unit tests for corridorkey.stages.postprocessor.hint_sharpen."""

from __future__ import annotations

import numpy as np
import pytest
from corridorkey.stages.postprocessor.hint_sharpen import sharpen_with_hint


def _make_arrays(h: int = 32, w: int = 32):
    alpha = np.ones((h, w, 1), dtype=np.float32) * 0.8
    fg = np.ones((h, w, 3), dtype=np.float32) * 0.5
    hint = np.ones((h, w, 1), dtype=np.float32)
    return alpha, fg, hint


class TestSharpenWithHint:
    def test_returns_tuple_of_two_arrays(self):
        alpha, fg, hint = _make_arrays()
        result = sharpen_with_hint(alpha, fg, hint)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes_match_input(self):
        alpha, fg, hint = _make_arrays(48, 64)
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint)
        assert alpha_out.shape == (48, 64, 1)
        assert fg_out.shape == (48, 64, 3)

    def test_full_foreground_hint_preserves_values(self):
        """All-ones hint → mask is all 1 → values unchanged."""
        alpha, fg, hint = _make_arrays()
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint, dilation_px=0)
        np.testing.assert_allclose(alpha_out, alpha)
        np.testing.assert_allclose(fg_out, fg)

    def test_zero_hint_zeros_output(self):
        """All-zeros hint → mask is all 0 → output is zeroed."""
        alpha, fg, _ = _make_arrays()
        hint = np.zeros((32, 32, 1), dtype=np.float32)
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint, dilation_px=0)
        assert alpha_out.max() == pytest.approx(0.0)
        assert fg_out.max() == pytest.approx(0.0)

    def test_2d_hint_accepted(self):
        """hint without channel dim [H, W] should work."""
        alpha, fg, _ = _make_arrays()
        hint_2d = np.ones((32, 32), dtype=np.float32)
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint_2d, dilation_px=0)
        np.testing.assert_allclose(alpha_out, alpha)

    def test_dilation_zero_no_error(self):
        alpha, fg, hint = _make_arrays()
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint, dilation_px=0)
        assert alpha_out.shape == alpha.shape

    def test_dilation_positive_expands_mask(self):
        """A small foreground region with dilation should grow."""
        h, w = 32, 32
        alpha = np.ones((h, w, 1), dtype=np.float32)
        fg = np.ones((h, w, 3), dtype=np.float32)
        # hint: only centre pixel is foreground
        hint = np.zeros((h, w, 1), dtype=np.float32)
        hint[16, 16, 0] = 1.0
        _, fg_no_dil = sharpen_with_hint(alpha, fg, hint, dilation_px=0)
        _, fg_with_dil = sharpen_with_hint(alpha, fg, hint, dilation_px=4)
        # With dilation, more pixels should be non-zero
        assert (fg_with_dil > 0).sum() > (fg_no_dil > 0).sum()

    def test_hint_smaller_than_source_is_upscaled(self):
        """hint at half resolution should be upscaled to match alpha/fg."""
        h, w = 32, 32
        alpha = np.ones((h, w, 1), dtype=np.float32)
        fg = np.ones((h, w, 3), dtype=np.float32)
        hint = np.ones((h // 2, w // 2, 1), dtype=np.float32)
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint, dilation_px=0)
        assert alpha_out.shape == (h, w, 1)
        assert fg_out.shape == (h, w, 3)

    def test_output_dtype_float32(self):
        alpha, fg, hint = _make_arrays()
        alpha_out, fg_out = sharpen_with_hint(alpha, fg, hint)
        assert alpha_out.dtype == np.float32
        assert fg_out.dtype == np.float32
