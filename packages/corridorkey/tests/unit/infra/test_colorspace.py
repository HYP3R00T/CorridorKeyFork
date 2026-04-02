"""Unit tests for corridorkey.infra.colorspace — shared LUT transfer functions."""

from __future__ import annotations

import numpy as np
import pytest
from corridorkey.infra.colorspace import (
    _LUT_SIZE,
    linear_to_srgb_lut,
    lut_apply,
    srgb_to_linear_lut,
)


class TestLutShape:
    def test_linear_to_srgb_lut_size(self):
        assert linear_to_srgb_lut.shape == (_LUT_SIZE,)

    def test_srgb_to_linear_lut_size(self):
        assert srgb_to_linear_lut.shape == (_LUT_SIZE,)

    def test_linear_to_srgb_lut_dtype(self):
        assert linear_to_srgb_lut.dtype == np.float32

    def test_srgb_to_linear_lut_dtype(self):
        assert srgb_to_linear_lut.dtype == np.float32


class TestLutBoundaryValues:
    def test_linear_to_srgb_zero_maps_to_zero(self):
        assert linear_to_srgb_lut[0] == pytest.approx(0.0, abs=1e-4)

    def test_linear_to_srgb_one_maps_to_one(self):
        assert linear_to_srgb_lut[-1] == pytest.approx(1.0, abs=1e-4)

    def test_srgb_to_linear_zero_maps_to_zero(self):
        assert srgb_to_linear_lut[0] == pytest.approx(0.0, abs=1e-4)

    def test_srgb_to_linear_one_maps_to_one(self):
        assert srgb_to_linear_lut[-1] == pytest.approx(1.0, abs=1e-4)

    def test_linear_to_srgb_lut_is_monotonic(self):
        assert np.all(np.diff(linear_to_srgb_lut) >= 0)

    def test_srgb_to_linear_lut_is_monotonic(self):
        assert np.all(np.diff(srgb_to_linear_lut) >= 0)


class TestLutApply:
    def test_output_shape_preserved(self):
        x = np.random.rand(8, 8, 3).astype(np.float32)
        out = lut_apply(x, linear_to_srgb_lut)
        assert out.shape == x.shape

    def test_output_dtype_float32(self):
        x = np.ones((4, 4), dtype=np.float32) * 0.5
        out = lut_apply(x, linear_to_srgb_lut)
        assert out.dtype == np.float32

    def test_zero_input_returns_zero(self):
        x = np.zeros((4, 4, 3), dtype=np.float32)
        out = lut_apply(x, linear_to_srgb_lut)
        assert np.allclose(out, 0.0, atol=1e-4)

    def test_one_input_returns_one(self):
        x = np.ones((4, 4, 3), dtype=np.float32)
        out = lut_apply(x, linear_to_srgb_lut)
        assert np.allclose(out, 1.0, atol=1e-4)

    def test_values_clipped_below_zero(self):
        x = np.full((2, 2), -0.5, dtype=np.float32)
        out = lut_apply(x, linear_to_srgb_lut)
        assert np.allclose(out, 0.0, atol=1e-4)

    def test_values_clipped_above_one(self):
        x = np.full((2, 2), 2.0, dtype=np.float32)
        out = lut_apply(x, linear_to_srgb_lut)
        assert np.allclose(out, 1.0, atol=1e-4)


class TestRoundTrip:
    """linear → sRGB → linear and sRGB → linear → sRGB must be near-lossless."""

    def test_linear_to_srgb_to_linear_roundtrip(self):
        # Sample a range of linear values (skip near-zero where LUT quantisation is coarser)
        x = np.linspace(0.01, 1.0, 256, dtype=np.float32)
        srgb = lut_apply(x, linear_to_srgb_lut)
        recovered = lut_apply(srgb, srgb_to_linear_lut)
        assert np.allclose(x, recovered, atol=5e-4)

    def test_srgb_to_linear_to_srgb_roundtrip(self):
        x = np.linspace(0.01, 1.0, 256, dtype=np.float32)
        linear = lut_apply(x, srgb_to_linear_lut)
        recovered = lut_apply(linear, linear_to_srgb_lut)
        assert np.allclose(x, recovered, atol=5e-4)


class TestKnownValues:
    """Spot-check against analytically computed IEC 61966-2-1 values."""

    def test_mid_grey_linear_to_srgb(self):
        # 0.5 linear → ~0.7354 sRGB  (IEC 61966-2-1)
        x = np.array([0.5], dtype=np.float32)
        out = lut_apply(x, linear_to_srgb_lut)
        assert out[0] == pytest.approx(0.7354, abs=5e-3)

    def test_mid_grey_srgb_to_linear(self):
        # 0.5 sRGB → ~0.2140 linear
        x = np.array([0.5], dtype=np.float32)
        out = lut_apply(x, srgb_to_linear_lut)
        assert out[0] == pytest.approx(0.2140, abs=5e-3)

    def test_linear_region_boundary(self):
        # Values below 0.0031308 use the linear segment (x * 12.92)
        x = np.array([0.001], dtype=np.float32)
        expected = 0.001 * 12.92
        out = lut_apply(x, linear_to_srgb_lut)
        assert out[0] == pytest.approx(expected, abs=1e-3)
