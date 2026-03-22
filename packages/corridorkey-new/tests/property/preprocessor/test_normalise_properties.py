"""Property-based tests for corridorkey_new.stages.preprocessor.normalise."""

from __future__ import annotations

import torch
from corridorkey_new.stages.preprocessor.normalise import _MEAN, normalise_image
from hypothesis import given
from hypothesis import strategies as st


def _image_strategy():
    return st.builds(
        lambda h, w, v: torch.full((1, 3, h, w), v, dtype=torch.float32),
        h=st.integers(1, 16),
        w=st.integers(1, 16),
        v=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )


class TestNormaliseImageProperties:
    @given(_image_strategy())
    def test_output_dtype_float32(self, image: torch.Tensor):
        """Output is always float32."""
        assert normalise_image(image).dtype == torch.float32

    @given(_image_strategy())
    def test_output_shape_preserved(self, image: torch.Tensor):
        """Shape is never changed."""
        assert normalise_image(image).shape == image.shape

    @given(_image_strategy())
    def test_output_is_finite(self, image: torch.Tensor):
        """Normalised values are not bounded to [0, 1] but must be finite."""
        result = normalise_image(image)
        assert torch.all(torch.isfinite(result))

    @given(
        st.integers(1, 8),
        st.integers(1, 8),
    )
    def test_imagenet_mean_maps_to_zero(self, h: int, w: int):
        """An image filled with ImageNet mean values normalises to ~0."""
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        image = mean.expand(1, 3, h, w).clone()
        result = normalise_image(image)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)
