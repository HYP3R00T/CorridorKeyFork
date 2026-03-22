"""Property-based tests for corridorkey_new.stages.preprocessor.resize."""

from __future__ import annotations

import torch
from corridorkey_new.stages.preprocessor.resize import LetterboxPad, letterbox_frame
from hypothesis import given
from hypothesis import strategies as st

_sizes = st.integers(4, 32)
_img_sizes = st.sampled_from([16, 32, 64])


def _make_image(h: int, w: int) -> torch.Tensor:
    return torch.zeros(1, 3, h, w, dtype=torch.float32)


def _make_alpha(h: int, w: int) -> torch.Tensor:
    return torch.zeros(1, 1, h, w, dtype=torch.float32)


class TestLetterboxFrameProperties:
    @given(_sizes, _sizes, _img_sizes)
    def test_output_image_shape(self, h: int, w: int, img_size: int):
        """Image output is always [1, 3, img_size, img_size]."""
        img, _, _ = letterbox_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert img.shape == (1, 3, img_size, img_size)

    @given(_sizes, _sizes, _img_sizes)
    def test_output_alpha_shape(self, h: int, w: int, img_size: int):
        """Alpha output is always [1, 1, img_size, img_size]."""
        _, alpha, _ = letterbox_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert alpha.shape == (1, 1, img_size, img_size)

    @given(_sizes, _sizes, _img_sizes)
    def test_output_dtype_float32(self, h: int, w: int, img_size: int):
        """Both outputs are always float32."""
        img, alpha, _ = letterbox_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert img.dtype == torch.float32
        assert alpha.dtype == torch.float32

    @given(_sizes, _sizes, _img_sizes)
    def test_values_finite(self, h: int, w: int, img_size: int):
        """All output values must be finite."""
        img = torch.rand(1, 3, h, w, dtype=torch.float32)
        alpha = torch.zeros(1, 1, h, w, dtype=torch.float32)
        img_out, alpha_out, _ = letterbox_frame(img, alpha, img_size)
        assert torch.isfinite(img_out).all()
        assert torch.isfinite(alpha_out).all()

    @given(_sizes, _sizes, _img_sizes)
    def test_alpha_always_clamped(self, h: int, w: int, img_size: int):
        """Alpha output is always in [0, 1] — clamped after every resize."""
        alpha = torch.rand(1, 1, h, w, dtype=torch.float32)
        _, alpha_out, _ = letterbox_frame(_make_image(h, w), alpha, img_size)
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0

    @given(_img_sizes)
    def test_already_square_same_size_unchanged_shape(self, img_size: int):
        """Resizing a square image to its own size preserves shape."""
        img_out, alpha_out, pad = letterbox_frame(
            _make_image(img_size, img_size), _make_alpha(img_size, img_size), img_size
        )
        assert img_out.shape == (1, 3, img_size, img_size)
        assert alpha_out.shape == (1, 1, img_size, img_size)
        assert pad.is_noop

    @given(_sizes, _sizes, _img_sizes)
    def test_image_and_alpha_same_spatial_dims(self, h: int, w: int, img_size: int):
        """Image and alpha always share the same H and W after resize."""
        img_out, alpha_out, _ = letterbox_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert img_out.shape[2:] == alpha_out.shape[2:]

    @given(_sizes, _sizes, _img_sizes)
    def test_pad_offsets_sum_to_img_size(self, h: int, w: int, img_size: int):
        """pad.top + pad.bottom + pad.inner_h == img_size (and same for width)."""
        _, _, pad = letterbox_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert pad.top + pad.bottom + pad.inner_h == img_size
        assert pad.left + pad.right + pad.inner_w == img_size

    @given(_sizes, _sizes, _img_sizes)
    def test_pad_returns_letterbox_pad_instance(self, h: int, w: int, img_size: int):
        """Third return value is always a LetterboxPad."""
        _, _, pad = letterbox_frame(_make_image(h, w), _make_alpha(h, w), img_size)
        assert isinstance(pad, LetterboxPad)
