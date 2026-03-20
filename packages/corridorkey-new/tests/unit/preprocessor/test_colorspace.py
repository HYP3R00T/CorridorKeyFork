"""Unit tests for corridorkey_new.preprocessor.colorspace."""

from __future__ import annotations

import numpy as np
import pytest
from corridorkey_new.preprocessor.colorspace import linear_to_srgb


class TestLinearToSrgb:
    def test_returns_float32(self):
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = linear_to_srgb(img)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self):
        img = np.random.rand(16, 16, 3).astype(np.float32)
        assert linear_to_srgb(img).shape == img.shape

    def test_zero_maps_to_zero(self):
        img = np.zeros((1, 1, 3), dtype=np.float32)
        result = linear_to_srgb(img)
        assert result[0, 0, 0] == pytest.approx(0.0)

    def test_one_maps_to_one(self):
        img = np.ones((1, 1, 3), dtype=np.float32)
        result = linear_to_srgb(img)
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_linear_segment_below_threshold(self):
        # Values <= 0.0031308 use the linear segment: out = in * 12.92
        val = 0.001
        img = np.array([[[val, val, val]]], dtype=np.float32)
        result = linear_to_srgb(img)
        assert result[0, 0, 0] == pytest.approx(val * 12.92, rel=1e-4)

    def test_power_segment_above_threshold(self):
        # Values > 0.0031308 use: 1.055 * in^(1/2.4) - 0.055
        val = 0.5
        img = np.array([[[val, val, val]]], dtype=np.float32)
        result = linear_to_srgb(img)
        expected = 1.055 * (val ** (1.0 / 2.4)) - 0.055
        assert result[0, 0, 0] == pytest.approx(expected, rel=1e-4)

    def test_output_clamped_to_0_1(self):
        # Values above 1.0 in linear should be clamped before conversion
        img = np.array([[[2.0, -0.5, 0.5]]], dtype=np.float32)
        result = linear_to_srgb(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_monotonically_increasing(self):
        # sRGB conversion must be monotonic
        vals = np.linspace(0.0, 1.0, 100).astype(np.float32)
        img = vals.reshape(1, 100, 1).repeat(3, axis=2)
        result = linear_to_srgb(img)[0, :, 0]
        assert np.all(np.diff(result) >= 0)
