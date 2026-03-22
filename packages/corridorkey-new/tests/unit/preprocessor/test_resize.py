"""Unit tests for corridorkey_new.stages.preprocessor.resize."""

from __future__ import annotations

import torch
import pytest
from corridorkey_new.stages.preprocessor.resize import (
    DEFAULT_ALPHA_UPSAMPLE_MODE,
    DEFAULT_UPSAMPLE_MODE,
    resize_frame,
)


def _make_image(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 3, h, w, dtype=torch.float32)


def _make_alpha(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 1, h, w, dtype=torch.float32)


class TestResizeFrame:
    def test_squish_output_shape_image(self):
        image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert image.shape == (1, 3, 512, 512)

    def test_squish_output_shape_alpha(self):
        image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert alpha.shape == (1, 1, 512, 512)

    def test_already_square_same_size_returns_same_tensors(self):
        img = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        image_out, alpha_out = resize_frame(img, alp, 512)
        # Should be the exact same objects — no-op path
        assert image_out is img
        assert alpha_out is alp

    def test_output_dtype_float32(self):
        image, alpha = resize_frame(_make_image(64, 64), _make_alpha(64, 64), 32)
        assert image.dtype == torch.float32
        assert alpha.dtype == torch.float32

    def test_downscale_values_stay_in_range(self):
        image, alpha = resize_frame(_make_image(128, 128), _make_alpha(128, 128), 32)
        assert image.min().item() >= 0.0 and image.max().item() <= 1.0
        assert alpha.min().item() >= 0.0 and alpha.max().item() <= 1.0

    def test_upscale_image_bicubic_can_ring(self):
        # Bicubic can produce slight overshoot/undershoot — not a bug.
        # Verify shape and dtype only.
        image, alpha = resize_frame(_make_image(16, 16), _make_alpha(16, 16), 64)
        assert image.dtype == torch.float32
        assert image.shape == (1, 3, 64, 64)

    def test_image_and_alpha_same_spatial_size(self):
        image, alpha = resize_frame(_make_image(200, 300), _make_alpha(200, 300), 128)
        assert image.shape[2:] == alpha.shape[2:]

    def test_default_upsample_mode_is_bicubic(self):
        assert DEFAULT_UPSAMPLE_MODE == "bicubic"

    def test_default_alpha_upsample_mode_is_bilinear(self):
        assert DEFAULT_ALPHA_UPSAMPLE_MODE == "bilinear"

    def test_upsample_mode_bilinear_produces_different_result_than_bicubic(self):
        """bilinear and bicubic must produce different tensors for the same input."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_bicubic, _ = resize_frame(src, alp, 128, upsample_mode="bicubic")
        img_bilinear, _ = resize_frame(src, alp, 128, upsample_mode="bilinear")
        assert not torch.allclose(img_bicubic, img_bilinear)

    def test_upsample_mode_has_no_effect_when_downscaling(self):
        """When downscaling, area mode is always used — upsample_mode is ignored."""
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_bicubic, _ = resize_frame(src, alp, 128, upsample_mode="bicubic")
        img_bilinear, _ = resize_frame(src, alp, 128, upsample_mode="bilinear")
        torch.testing.assert_close(img_bicubic, img_bilinear)

    def test_alpha_upsample_mode_independent_of_image_mode(self):
        """alpha_upsample_mode can differ from upsample_mode."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        _, alpha_bilinear = resize_frame(src, alp, 128, upsample_mode="bicubic", alpha_upsample_mode="bilinear")
        _, alpha_bicubic = resize_frame(src, alp, 128, upsample_mode="bicubic", alpha_upsample_mode="bicubic")
        # bilinear and bicubic alpha should differ
        assert not torch.allclose(alpha_bilinear, alpha_bicubic)

    def test_mixed_dimensions_correct_modes(self):
        """Portrait source (tall) into square target: height downscales, width upscales."""
        # src_h=512 > img_size=256 (downscale), src_w=64 < img_size=256 (upscale)
        src = _make_image(512, 64)
        alp = _make_alpha(512, 64)
        image_out, alpha_out = resize_frame(src, alp, 256)
        assert image_out.shape == (1, 3, 256, 256)
        assert alpha_out.shape == (1, 1, 256, 256)

    def test_mixed_dimensions_landscape_into_square(self):
        """Landscape source (wide) into square target: width downscales, height upscales."""
        # src_h=64 < img_size=256 (upscale), src_w=512 > img_size=256 (downscale)
        src = _make_image(64, 512)
        alp = _make_alpha(64, 512)
        image_out, alpha_out = resize_frame(src, alp, 256)
        assert image_out.shape == (1, 3, 256, 256)
        assert alpha_out.shape == (1, 1, 256, 256)


class TestResizeFrameHalfPrecision:
    def test_float16_input_produces_float16_output(self):
        src = _make_image(64, 64).half()
        alp = _make_alpha(64, 64).half()
        image_out, alpha_out = resize_frame(src, alp, 32)
        assert image_out.dtype == torch.float16
        assert alpha_out.dtype == torch.float16


class TestResizeAlphaRange:
    def test_bilinear_alpha_upscale_stays_in_range(self):
        """bilinear alpha upscale must not produce values outside [0, 1]."""
        src = _make_image(16, 16)
        # Sharp 0/1 matte — worst case for ringing
        alp = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        alp[:, :, 8:, :] = 1.0
        _, alpha_out = resize_frame(src, alp, 64, alpha_upsample_mode="bilinear")
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0

    def test_bicubic_alpha_upscale_can_ring(self):
        """bicubic alpha upscale may ring below 0 on a hard edge — this is expected
        and is why alpha_upsample_mode defaults to bilinear, not bicubic."""
        src = _make_image(16, 16)
        alp = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        alp[:, :, 8:, :] = 1.0
        _, alpha_out = resize_frame(src, alp, 64, alpha_upsample_mode="bicubic")
        # Shape and dtype must still be correct even if values ring
        assert alpha_out.shape == (1, 1, 64, 64)
        assert alpha_out.dtype == torch.float32
