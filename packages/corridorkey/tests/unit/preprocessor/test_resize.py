"""Unit tests for corridorkey.stages.preprocessor.resize."""

from __future__ import annotations

import torch
from corridorkey.stages.preprocessor.resize import (
    DEFAULT_IMAGE_UPSAMPLE_MODE,
    _linear_to_srgb,
    _multistep_downscale,
    _srgb_to_linear,
    _unsharp_mask,
    resize_frame,
)


def _make_image(h: int, w: int, fill: float | None = None) -> torch.Tensor:
    if fill is not None:
        return torch.full((1, 3, h, w), fill, dtype=torch.float32)
    return torch.rand(1, 3, h, w, dtype=torch.float32)


def _make_alpha(h: int, w: int, fill: float | None = None) -> torch.Tensor:
    if fill is not None:
        return torch.full((1, 1, h, w), fill, dtype=torch.float32)
    return torch.rand(1, 1, h, w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# resize_frame output shape
# ---------------------------------------------------------------------------


class TestResizeFrameShape:
    def test_landscape_output_square(self):
        img, alp = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert img.shape == (1, 3, 512, 512)
        assert alp.shape == (1, 1, 512, 512)

    def test_portrait_output_square(self):
        img, alp = resize_frame(_make_image(1920, 1080), _make_alpha(1920, 1080), 512)
        assert img.shape == (1, 3, 512, 512)
        assert alp.shape == (1, 1, 512, 512)

    def test_square_source_output_shape(self):
        img, alp = resize_frame(_make_image(256, 256), _make_alpha(256, 256), 512)
        assert img.shape == (1, 3, 512, 512)

    def test_already_target_size_passthrough(self):
        src_img = _make_image(512, 512)
        src_alp = _make_alpha(512, 512)
        img, alp = resize_frame(src_img, src_alp, 512)
        assert img is src_img
        assert alp.shape == src_alp.shape

    def test_output_dtype_float32(self):
        img, alp = resize_frame(_make_image(64, 64), _make_alpha(64, 64), 32)
        assert img.dtype == torch.float32
        assert alp.dtype == torch.float32


# ---------------------------------------------------------------------------
# Alpha clamp after resize
# ---------------------------------------------------------------------------


class TestAlphaClamp:
    def test_alpha_clamped_after_upscale(self):
        src = _make_image(16, 16)
        alp = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        alp[:, :, 8:, :] = 1.0
        _, alpha_out = resize_frame(src, alp, 64)
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0

    def test_alpha_clamped_after_downscale(self):
        _, alpha_out = resize_frame(_make_image(512, 512), _make_alpha(512, 512), 128)
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Colour-aware downscaling
# ---------------------------------------------------------------------------


class TestColourAwareDownscaling:
    def test_srgb_downscale_differs_from_linear_downscale(self):
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_srgb, _ = resize_frame(src, alp, 128, is_srgb=True)
        img_linear, _ = resize_frame(src, alp, 128, is_srgb=False)
        assert not torch.allclose(img_srgb, img_linear)

    def test_srgb_downscale_output_in_range(self):
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img, _ = resize_frame(src, alp, 128, is_srgb=True)
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

    def test_upscale_not_affected_by_is_srgb(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_srgb, _ = resize_frame(src, alp, 128, is_srgb=True)
        img_linear, _ = resize_frame(src, alp, 128, is_srgb=False)
        torch.testing.assert_close(img_srgb, img_linear)


# ---------------------------------------------------------------------------
# Post-upscale sharpening
# ---------------------------------------------------------------------------


class TestPostUpscaleSharpening:
    def test_sharpening_produces_different_result(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_sharp, _ = resize_frame(src, alp, 128, sharpen_strength=0.3)
        img_plain, _ = resize_frame(src, alp, 128, sharpen_strength=0.0)
        assert not torch.allclose(img_sharp, img_plain)

    def test_sharpening_output_in_range(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img, _ = resize_frame(src, alp, 128, sharpen_strength=0.5)
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

    def test_sharpening_disabled_on_downscale(self):
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_sharp, _ = resize_frame(src, alp, 128, sharpen_strength=0.5)
        img_plain, _ = resize_frame(src, alp, 128, sharpen_strength=0.0)
        torch.testing.assert_close(img_sharp, img_plain)

    def test_sharpening_does_not_affect_alpha(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        _, alp_sharp = resize_frame(src, alp, 128, sharpen_strength=0.5)
        _, alp_plain = resize_frame(src, alp, 128, sharpen_strength=0.0)
        torch.testing.assert_close(alp_sharp, alp_plain)


# ---------------------------------------------------------------------------
# Multi-step downscaling
# ---------------------------------------------------------------------------


class TestMultistepDownscaling:
    def test_extreme_ratio_produces_correct_shape(self):
        src = _make_image(4320, 7680)  # 8K
        alp = _make_alpha(4320, 7680)
        img, alp_out = resize_frame(src, alp, 512)
        assert img.shape == (1, 3, 512, 512)
        assert alp_out.shape == (1, 1, 512, 512)

    def test_multistep_helper_produces_correct_shape(self):
        img = _make_image(4096, 4096)
        alp = _make_alpha(4096, 4096)
        img_out, alp_out = _multistep_downscale(img, alp, 256, 256)
        assert img_out.shape == (1, 3, 256, 256)
        assert alp_out.shape == (1, 1, 256, 256)


# ---------------------------------------------------------------------------
# Upsample mode behaviour
# ---------------------------------------------------------------------------


class TestUpsampleModes:
    def test_default_image_upsample_mode_is_bicubic(self):
        assert DEFAULT_IMAGE_UPSAMPLE_MODE == "bicubic"

    def test_bilinear_differs_from_bicubic_on_upscale(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_bicubic, _ = resize_frame(src, alp, 128, image_upsample_mode="bicubic")
        img_bilinear, _ = resize_frame(src, alp, 128, image_upsample_mode="bilinear")
        assert not torch.allclose(img_bicubic, img_bilinear)

    def test_upsample_mode_no_effect_when_downscaling(self):
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_bicubic, _ = resize_frame(src, alp, 128, image_upsample_mode="bicubic")
        img_bilinear, _ = resize_frame(src, alp, 128, image_upsample_mode="bilinear")
        torch.testing.assert_close(img_bicubic, img_bilinear)


# ---------------------------------------------------------------------------
# Half precision
# ---------------------------------------------------------------------------


class TestHalfPrecision:
    def test_float16_input_produces_float16_output(self):
        src = _make_image(64, 64).half()
        alp = _make_alpha(64, 64).half()
        img, alp_out = resize_frame(src, alp, 32)
        assert img.dtype == torch.float16
        assert alp_out.dtype == torch.float16


# ---------------------------------------------------------------------------
# sRGB ↔ linear helpers
# ---------------------------------------------------------------------------


class TestColourspaceHelpers:
    def test_srgb_to_linear_roundtrip(self):
        x = torch.linspace(0.0, 1.0, 256).reshape(1, 1, 16, 16)
        roundtrip = _linear_to_srgb(_srgb_to_linear(x))
        torch.testing.assert_close(roundtrip, x, atol=1e-4, rtol=0)

    def test_linear_to_srgb_roundtrip(self):
        x = torch.linspace(0.0, 1.0, 256).reshape(1, 1, 16, 16)
        roundtrip = _srgb_to_linear(_linear_to_srgb(x))
        torch.testing.assert_close(roundtrip, x, atol=1e-4, rtol=0)

    def test_srgb_to_linear_output_in_range(self):
        x = torch.rand(1, 3, 32, 32)
        assert _srgb_to_linear(x).min().item() >= 0.0
        assert _srgb_to_linear(x).max().item() <= 1.0

    def test_linear_to_srgb_output_in_range(self):
        x = torch.rand(1, 3, 32, 32)
        assert _linear_to_srgb(x).min().item() >= 0.0
        assert _linear_to_srgb(x).max().item() <= 1.0


# ---------------------------------------------------------------------------
# Unsharp mask helper
# ---------------------------------------------------------------------------


class TestUnsharpMask:
    def test_output_shape_unchanged(self):
        x = _make_image(64, 64)
        out = _unsharp_mask(x, strength=0.3)
        assert out.shape == x.shape

    def test_output_in_range(self):
        x = _make_image(64, 64)
        out = _unsharp_mask(x, strength=0.5)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_zero_strength_is_identity(self):
        x = _make_image(64, 64)
        out = _unsharp_mask(x, strength=0.0)
        torch.testing.assert_close(out, x)
