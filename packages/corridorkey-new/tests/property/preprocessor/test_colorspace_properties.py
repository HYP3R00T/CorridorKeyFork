"""Property-based tests for corridorkey_new.preprocessor.colorspace."""

from __future__ import annotations

import torch
from corridorkey_new.preprocessor.colorspace import linear_to_srgb
from hypothesis import given
from hypothesis import strategies as st


def _tensor_strategy(min_val: float = 0.0, max_val: float = 1.0):
    return st.builds(
        lambda h, w, v: torch.full((1, 3, h, w), v, dtype=torch.float32),
        h=st.integers(1, 16),
        w=st.integers(1, 16),
        v=st.floats(min_val, max_val, allow_nan=False, allow_infinity=False),
    )


class TestLinearToSrgbProperties:
    @given(_tensor_strategy())
    def test_output_range_0_1(self, image: torch.Tensor):
        """sRGB output is always in [0, 1]."""
        result = linear_to_srgb(image)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    @given(_tensor_strategy())
    def test_output_dtype_float32(self, image: torch.Tensor):
        """Output is always float32."""
        assert linear_to_srgb(image).dtype == torch.float32

    @given(_tensor_strategy())
    def test_output_shape_preserved(self, image: torch.Tensor):
        """Shape is never changed."""
        assert linear_to_srgb(image).shape == image.shape

    @given(_tensor_strategy())
    def test_monotone(self, image: torch.Tensor):
        """Brighter linear input always produces brighter sRGB output (monotone)."""
        darker = linear_to_srgb(image * 0.5)
        brighter = linear_to_srgb(image)
        assert torch.all(darker <= brighter + 1e-6)

    @given(_tensor_strategy(min_val=-2.0, max_val=2.0))
    def test_out_of_range_input_clamped(self, image: torch.Tensor):
        """Out-of-range linear values are clamped, not propagated."""
        result = linear_to_srgb(image)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0
