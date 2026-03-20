"""Unit tests for corridorkey_new.preprocessor.normalise."""

from __future__ import annotations

import numpy as np
from corridorkey_new.preprocessor.normalise import _MEAN, _STD, normalise_image


class TestNormaliseImage:
    def test_returns_float32(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        assert normalise_image(img).dtype == np.float32

    def test_shape_unchanged(self):
        img = np.random.rand(16, 16, 3).astype(np.float32)
        assert normalise_image(img).shape == img.shape

    def test_mean_pixel_normalises_to_zero(self):
        # A pixel equal to the ImageNet mean should normalise to 0
        img = np.array([[_MEAN[0, 0]]], dtype=np.float32)  # shape (1, 1, 3)
        result = normalise_image(img)
        np.testing.assert_allclose(result[0, 0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_known_value(self):
        # pixel = mean + std should normalise to 1.0 per channel
        pixel = _MEAN[0, 0] + _STD[0, 0]
        img = np.array([[pixel]], dtype=np.float32)
        result = normalise_image(img)
        np.testing.assert_allclose(result[0, 0], [1.0, 1.0, 1.0], atol=1e-5)

    def test_output_can_be_negative(self):
        # Values below the mean produce negative outputs — that is correct
        img = np.zeros((4, 4, 3), dtype=np.float32)
        result = normalise_image(img)
        assert result.min() < 0.0

    def test_output_can_exceed_one(self):
        # Values above mean + std produce outputs > 1 — that is correct
        img = np.ones((4, 4, 3), dtype=np.float32)
        result = normalise_image(img)
        assert result.max() > 1.0
