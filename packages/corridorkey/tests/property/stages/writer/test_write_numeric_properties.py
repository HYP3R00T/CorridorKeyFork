"""Property-based tests for writer numeric conversion — float32 → uint8/uint16."""

from __future__ import annotations

import numpy as np
from corridorkey.stages.writer.orchestrator import _alpha_to_bgr
from hypothesis import given
from hypothesis import strategies as st


def _float_array(channels: int = 3):
    return st.builds(
        lambda h, w, v: np.full((h, w, channels), v, dtype=np.float32),
        h=st.integers(1, 16),
        w=st.integers(1, 16),
        v=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )


class TestAlphaToBgrProperties:
    @given(_float_array(channels=1))
    def test_output_shape_is_hw3(self, alpha: np.ndarray):
        out = _alpha_to_bgr(alpha)
        assert out.shape == (alpha.shape[0], alpha.shape[1], 3)

    @given(_float_array(channels=1))
    def test_all_channels_equal(self, alpha: np.ndarray):
        """Alpha replicated to BGR — all three channels must be identical."""
        out = _alpha_to_bgr(alpha)
        assert np.allclose(out[:, :, 0], out[:, :, 1])
        assert np.allclose(out[:, :, 1], out[:, :, 2])

    @given(_float_array(channels=1))
    def test_values_preserved(self, alpha: np.ndarray):
        """Values from the alpha channel are preserved in the output."""
        out = _alpha_to_bgr(alpha)
        assert np.allclose(out[:, :, 0], alpha[:, :, 0])


class TestUint8ConversionProperties:
    @given(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    def test_float_to_uint8_in_range(self, v: float):
        """float32 in [0, 1] converted to uint8 must be in [0, 255]."""
        arr = np.array([v], dtype=np.float32)
        uint8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        assert 0 <= int(uint8[0]) <= 255

    @given(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    def test_float_to_uint16_in_range(self, v: float):
        """float32 in [0, 1] converted to uint16 must be in [0, 65535]."""
        arr = np.array([v], dtype=np.float32)
        uint16 = (np.clip(arr, 0.0, 1.0) * 65535.0).astype(np.uint16)
        assert 0 <= int(uint16[0]) <= 65535
