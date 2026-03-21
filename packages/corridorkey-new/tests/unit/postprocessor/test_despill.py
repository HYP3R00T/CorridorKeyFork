"""Unit tests for corridorkey_new.postprocessor.despill."""

from __future__ import annotations

import numpy as np
from corridorkey_new.postprocessor.despill import remove_spill


def _green_image(h: int = 8, w: int = 8) -> np.ndarray:
    """Return an image with strong green channel."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 1] = 1.0  # pure green
    return img


class TestRemoveSpill:
    def test_zero_strength_returns_unchanged(self):
        img = _green_image()
        result = remove_spill(img, strength=0.0)
        assert np.array_equal(result, img)

    def test_full_strength_reduces_green(self):
        img = _green_image()
        result = remove_spill(img, strength=1.0)
        # Green channel should be reduced
        assert result[:, :, 1].mean() < img[:, :, 1].mean()

    def test_output_dtype_float32(self):
        img = _green_image()
        result = remove_spill(img, strength=1.0)
        assert result.dtype == np.float32

    def test_output_shape_preserved(self):
        img = _green_image(16, 24)
        result = remove_spill(img, strength=1.0)
        assert result.shape == img.shape

    def test_no_spill_image_unchanged(self):
        # A neutral grey image has no green spill — result should be close to input.
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        result = remove_spill(img, strength=1.0)
        assert np.allclose(result, img, atol=1e-5)

    def test_partial_strength_blends(self):
        img = _green_image()
        full = remove_spill(img, strength=1.0)
        half = remove_spill(img, strength=0.5)
        # Half strength should be between original and full despill
        assert half[:, :, 1].mean() > full[:, :, 1].mean()
        assert half[:, :, 1].mean() < img[:, :, 1].mean()
