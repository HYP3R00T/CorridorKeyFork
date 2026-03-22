"""Property-based tests for corridorkey_new.stages.preprocessor.resize."""

from __future__ import annotations

import torch
from corridorkey_new.stages.preprocessor.resize import resize_frame
from hypothesis import given
from hypothesis import strategies as st

_sizes = st.integers(4, 32)
_img_sizes = st.sampled_from([16, 32, 64])


def _make_image(h: int, w: int) -> torch.Tensor:
    return torch.zeros(1, 3, h, w, dtype=torch.float32)


def _make_alpha(h: int, w: int) -> torch.Tensor:
    return torch.zeros(1, 1, h, w, dtype=torch.float32)


class TestResizeFrameProperties:
    @given(_sizes, _sizes, _img_sizes)
    def test_output_image_shape(self, h: int, w: int, img_size: int):
        """Image output is always [1, 3, img_size, img_size]."""
        img, _ = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert img.shape == (1, 3, img_size, img_size)

    @given(_sizes, _sizes, _img_sizes)
    def test_output_alpha_shape(self, h: int, w: int, img_size: int):
        """Alpha output is always [1, 1, img_size, img_size]."""
        _, alpha = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert alpha.shape == (1, 1, img_size, img_size)

    @given(_sizes, _sizes, _img_sizes)
    def test_output_dtype_float32(self, h: int, w: int, img_size: int):
        """Both outputs are always float32."""
        img, alpha = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert img.dtype == torch.float32
        assert alpha.dtype == torch.float32

    @given(_sizes, _sizes, _img_sizes)
    def test_values_stay_in_range(self, h: int, w: int, img_size: int):
        """Downscaling (area mode) never produces values outside [0, 1].
        Upscaling (bicubic) can ring slightly outside [0, 1] — that is a known
        property of the cubic kernel and is not a bug. We only assert the
        output is finite and the dtype is preserved.
        """
        img = torch.rand(1, 3, h, w, dtype=torch.float32)
        alpha = torch.zeros(1, 1, h, w, dtype=torch.float32)
        img_out, alpha_out = resize_frame(img, alpha, img_size)
        assert torch.isfinite(img_out).all()
        assert torch.isfinite(alpha_out).all()
        # Pure downscaling (area mode) — output must stay in [0, 1].
        # Mixed or upscaling paths use bicubic which can ring slightly outside.
        if h > img_size and w > img_size:
            assert img_out.min().item() >= -1e-6
            assert img_out.max().item() <= 1.0 + 1e-6

    @given(_img_sizes)
    def test_already_square_same_size_unchanged_shape(self, img_size: int):
        """Resizing a square image to its own size preserves shape."""
        img_out, alpha_out = resize_frame(
            _make_image(img_size, img_size), _make_alpha(img_size, img_size), img_size
        )
        assert img_out.shape == (1, 3, img_size, img_size)
        assert alpha_out.shape == (1, 1, img_size, img_size)

    @given(_sizes, _sizes, _img_sizes)
    def test_image_and_alpha_same_spatial_dims(self, h: int, w: int, img_size: int):
        """Image and alpha always share the same H and W after resize."""
        img_out, alpha_out = resize_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert img_out.shape[2:] == alpha_out.shape[2:]
