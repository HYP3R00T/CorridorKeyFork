"""Property-based tests for corridorkey.infra.colorspace — LUT transfer functions."""

from __future__ import annotations

import numpy as np
from corridorkey.infra.colorspace import linear_to_srgb_lut, lut_apply, srgb_to_linear_lut
from hypothesis import given, settings
from hypothesis import strategies as st


def _array(min_val: float = 0.0, max_val: float = 1.0):
    return st.builds(
        lambda h, w, v: np.full((h, w, 3), v, dtype=np.float32),
        h=st.integers(1, 16),
        w=st.integers(1, 16),
        v=st.floats(min_val, max_val, allow_nan=False, allow_infinity=False),
    )


class TestLutApplyProperties:
    @given(_array())
    def test_output_range_0_1(self, arr: np.ndarray):
        """lut_apply output is always in [0, 1]."""
        out = lut_apply(arr, linear_to_srgb_lut)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    @given(_array())
    def test_output_shape_preserved(self, arr: np.ndarray):
        out = lut_apply(arr, linear_to_srgb_lut)
        assert out.shape == arr.shape

    @given(_array())
    def test_output_dtype_float32(self, arr: np.ndarray):
        out = lut_apply(arr, linear_to_srgb_lut)
        assert out.dtype == np.float32

    @given(_array(min_val=-2.0, max_val=2.0))
    def test_out_of_range_clamped(self, arr: np.ndarray):
        """Values outside [0, 1] are clamped, not propagated."""
        out = lut_apply(arr, linear_to_srgb_lut)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    @given(_array())
    def test_monotone_linear_to_srgb(self, arr: np.ndarray):
        """Brighter linear input always produces brighter sRGB output."""
        darker = lut_apply(arr * 0.5, linear_to_srgb_lut)
        brighter = lut_apply(arr, linear_to_srgb_lut)
        assert np.all(darker <= brighter + 1e-4)


class TestRoundTripProperties:
    @given(_array(min_val=0.01, max_val=1.0))
    @settings(max_examples=50)
    def test_linear_srgb_linear_roundtrip(self, arr: np.ndarray):
        """linear → sRGB → linear is near-lossless for values away from zero."""
        srgb = lut_apply(arr, linear_to_srgb_lut)
        recovered = lut_apply(srgb, srgb_to_linear_lut)
        assert np.allclose(arr, recovered, atol=5e-4)

    @given(_array(min_val=0.01, max_val=1.0))
    @settings(max_examples=50)
    def test_srgb_linear_srgb_roundtrip(self, arr: np.ndarray):
        """sRGB → linear → sRGB is near-lossless for values away from zero."""
        linear = lut_apply(arr, srgb_to_linear_lut)
        recovered = lut_apply(linear, linear_to_srgb_lut)
        assert np.allclose(arr, recovered, atol=5e-4)
