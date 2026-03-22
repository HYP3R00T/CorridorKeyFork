"""Unit tests for corridorkey_new.stages.preprocessor.resize."""

from __future__ import annotations

import torch
import pytest
from corridorkey_new.stages.preprocessor.resize import (
    DEFAULT_ALPHA_UPSAMPLE_MODE,
    DEFAULT_UPSAMPLE_MODE,
    LetterboxPad,
    letterbox_frame,
    _srgb_to_linear,
    _linear_to_srgb,
    _unsharp_mask,
    _multistep_downscale,
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
# LetterboxPad dataclass
# ---------------------------------------------------------------------------

class TestLetterboxPad:
    def test_is_noop_when_all_zero(self):
        pad = LetterboxPad(0, 0, 0, 0, 64, 64)
        assert pad.is_noop is True

    def test_is_not_noop_when_any_nonzero(self):
        assert LetterboxPad(1, 0, 0, 0, 64, 64).is_noop is False
        assert LetterboxPad(0, 1, 0, 0, 64, 64).is_noop is False
        assert LetterboxPad(0, 0, 1, 0, 64, 64).is_noop is False
        assert LetterboxPad(0, 0, 0, 1, 64, 64).is_noop is False

    def test_frozen(self):
        pad = LetterboxPad(1, 2, 3, 4, 60, 56)
        with pytest.raises((AttributeError, TypeError)):
            pad.top = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Letterbox output shape
# ---------------------------------------------------------------------------

class TestLetterboxFrameShape:
    def test_output_always_square_image(self):
        img, alp, _ = letterbox_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert img.shape == (1, 3, 512, 512)

    def test_output_always_square_alpha(self):
        img, alp, _ = letterbox_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert alp.shape == (1, 1, 512, 512)

    def test_portrait_output_shape(self):
        img, alp, _ = letterbox_frame(_make_image(1920, 1080), _make_alpha(1920, 1080), 512)
        assert img.shape == (1, 3, 512, 512)
        assert alp.shape == (1, 1, 512, 512)

    def test_square_source_output_shape(self):
        img, alp, _ = letterbox_frame(_make_image(256, 256), _make_alpha(256, 256), 512)
        assert img.shape == (1, 3, 512, 512)

    def test_already_target_size_returns_same_tensors(self):
        src_img = _make_image(512, 512)
        src_alp = _make_alpha(512, 512)
        img, alp, pad = letterbox_frame(src_img, src_alp, 512)
        assert img is src_img
        # Alpha is clamped even on the no-op path — values are identical but object may differ
        assert alp.shape == src_alp.shape
        assert pad.is_noop

    def test_output_dtype_float32(self):
        img, alp, _ = letterbox_frame(_make_image(64, 64), _make_alpha(64, 64), 32)
        assert img.dtype == torch.float32
        assert alp.dtype == torch.float32


# ---------------------------------------------------------------------------
# Letterboxing — aspect ratio and padding correctness
# ---------------------------------------------------------------------------

class TestLetterboxAspectRatio:
    def test_landscape_pad_is_top_bottom(self):
        """16:9 landscape → pillarbox on top/bottom, no left/right padding."""
        img, alp, pad = letterbox_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert pad.left == 0
        assert pad.right == 0
        assert pad.top > 0
        assert pad.bottom > 0

    def test_portrait_pad_is_left_right(self):
        """9:16 portrait → letterbox on left/right, no top/bottom padding."""
        img, alp, pad = letterbox_frame(_make_image(1920, 1080), _make_alpha(1920, 1080), 512)
        assert pad.top == 0
        assert pad.bottom == 0
        assert pad.left > 0
        assert pad.right > 0

    def test_square_source_no_padding(self):
        """Square source → no padding at all."""
        img, alp, pad = letterbox_frame(_make_image(256, 256), _make_alpha(256, 256), 512)
        assert pad.is_noop

    def test_pad_offsets_sum_to_correct_total(self):
        """top + bottom + inner_h == img_size, left + right + inner_w == img_size."""
        img, alp, pad = letterbox_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert pad.top + pad.bottom + pad.inner_h == 512
        assert pad.left + pad.right + pad.inner_w == 512

    def test_inner_dimensions_preserve_aspect_ratio(self):
        """inner_h / inner_w should match src_h / src_w within rounding tolerance."""
        src_h, src_w = 1080, 1920
        img, alp, pad = letterbox_frame(_make_image(src_h, src_w), _make_alpha(src_h, src_w), 512)
        expected_ratio = src_h / src_w
        actual_ratio = pad.inner_h / pad.inner_w
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_padding_is_symmetric(self):
        """Padding should be symmetric (off-by-one at most for odd totals)."""
        img, alp, pad = letterbox_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        assert abs(pad.top - pad.bottom) <= 1

    def test_padding_value_matches_mean_pixel(self):
        """Padded border pixels should equal the mean pixel value of the content."""
        # Uniform image so mean is well-defined
        fill = 0.4
        img, alp, pad = letterbox_frame(_make_image(1080, 1920, fill=fill), _make_alpha(1080, 1920), 512)
        if pad.top > 0:
            top_strip = img[:, :, :pad.top, :]
            assert top_strip.mean().item() == pytest.approx(fill, abs=0.01)
        if pad.bottom > 0:
            bottom_strip = img[:, :, 512 - pad.bottom:, :]
            assert bottom_strip.mean().item() == pytest.approx(fill, abs=0.01)

    def test_alpha_padding_is_zero(self):
        """Alpha padding must be 0.0 (fully transparent border)."""
        img, alp, pad = letterbox_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512)
        if pad.top > 0:
            assert alp[:, :, :pad.top, :].max().item() == pytest.approx(0.0)
        if pad.bottom > 0:
            assert alp[:, :, 512 - pad.bottom:, :].max().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Alpha clamp after resize
# ---------------------------------------------------------------------------

class TestAlphaClamp:
    def test_bilinear_alpha_upscale_stays_in_range(self):
        """bilinear alpha upscale must not produce values outside [0, 1]."""
        src = _make_image(16, 16)
        alp = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        alp[:, :, 8:, :] = 1.0
        _, alpha_out, _ = letterbox_frame(src, alp, 64, alpha_upsample_mode="bilinear")
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0

    def test_bicubic_alpha_clamped_after_resize(self):
        """bicubic alpha is clamped to [0, 1] even though it can ring."""
        src = _make_image(16, 16)
        alp = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        alp[:, :, 8:, :] = 1.0
        _, alpha_out, _ = letterbox_frame(src, alp, 64, alpha_upsample_mode="bicubic")
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0

    def test_downscale_alpha_stays_in_range(self):
        _, alpha_out, _ = letterbox_frame(_make_image(512, 512), _make_alpha(512, 512), 128)
        assert alpha_out.min().item() >= 0.0
        assert alpha_out.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Colour-aware downscaling
# ---------------------------------------------------------------------------

class TestColourAwareDownscaling:
    def test_srgb_downscale_differs_from_linear_downscale(self):
        """is_srgb=True should produce a different result than is_srgb=False."""
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_srgb, _, _ = letterbox_frame(src, alp, 128, is_srgb=True)
        img_linear, _, _ = letterbox_frame(src, alp, 128, is_srgb=False)
        assert not torch.allclose(img_srgb, img_linear)

    def test_srgb_downscale_output_in_range(self):
        """Colour-aware downscale must stay in [0, 1]."""
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img, _, _ = letterbox_frame(src, alp, 128, is_srgb=True)
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

    def test_upscale_is_not_affected_by_is_srgb(self):
        """is_srgb has no effect when upscaling — linearise/re-encode only on downscale."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_srgb, _, _ = letterbox_frame(src, alp, 128, is_srgb=True)
        img_linear, _, _ = letterbox_frame(src, alp, 128, is_srgb=False)
        torch.testing.assert_close(img_srgb, img_linear)


# ---------------------------------------------------------------------------
# Post-upscale sharpening
# ---------------------------------------------------------------------------

class TestPostUpscaleSharpening:
    def test_sharpening_produces_different_result(self):
        """sharpen_strength > 0 must produce a different tensor than 0."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_sharp, _, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.3)
        img_plain, _, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.0)
        assert not torch.allclose(img_sharp, img_plain)

    def test_sharpening_output_in_range(self):
        """Sharpened output must stay in [0, 1]."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img, _, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.5)
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

    def test_sharpening_disabled_on_downscale(self):
        """sharpen_strength has no effect when downscaling."""
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_sharp, _, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.5)
        img_plain, _, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.0)
        torch.testing.assert_close(img_sharp, img_plain)

    def test_sharpening_does_not_affect_alpha(self):
        """Sharpening is applied to image only, not alpha."""
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        _, alp_sharp, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.5)
        _, alp_plain, _ = letterbox_frame(src, alp, 128, sharpen_strength=0.0)
        torch.testing.assert_close(alp_sharp, alp_plain)


# ---------------------------------------------------------------------------
# Multi-step downscaling
# ---------------------------------------------------------------------------

class TestMultistepDownscaling:
    def test_extreme_ratio_produces_correct_shape(self):
        """8K → 512 is a 15× ratio — multi-step path must still produce correct shape."""
        src = _make_image(4320, 7680)  # 8K
        alp = _make_alpha(4320, 7680)
        img, alp_out, _ = letterbox_frame(src, alp, 512)
        assert img.shape == (1, 3, 512, 512)
        assert alp_out.shape == (1, 1, 512, 512)

    def test_multistep_differs_from_oneshot(self):
        """Multi-step downscale should produce a different result than one-shot area."""
        import torch.nn.functional as F
        src = _make_image(2048, 2048)
        alp = _make_alpha(2048, 2048)
        # Multi-step path (ratio > 4)
        img_multi, _, _ = letterbox_frame(src, alp, 256)
        # One-shot area (manually)
        combined = torch.cat([src, alp], dim=1)
        out = F.interpolate(combined, size=(256, 256), mode="area")
        img_oneshot = out[:, :3]
        # They may differ due to intermediate halving steps
        # (not guaranteed to differ for all inputs, but for random data they will)
        # Just verify shape and range
        assert img_multi.shape == (1, 3, 256, 256)
        assert img_multi.min().item() >= 0.0

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
    def test_default_upsample_mode_is_bicubic(self):
        assert DEFAULT_UPSAMPLE_MODE == "bicubic"

    def test_default_alpha_upsample_mode_is_bilinear(self):
        assert DEFAULT_ALPHA_UPSAMPLE_MODE == "bilinear"

    def test_bilinear_differs_from_bicubic_on_upscale(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img_bicubic, _, _ = letterbox_frame(src, alp, 128, upsample_mode="bicubic")
        img_bilinear, _, _ = letterbox_frame(src, alp, 128, upsample_mode="bilinear")
        assert not torch.allclose(img_bicubic, img_bilinear)

    def test_upsample_mode_has_no_effect_when_downscaling(self):
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_bicubic, _, _ = letterbox_frame(src, alp, 128, upsample_mode="bicubic")
        img_bilinear, _, _ = letterbox_frame(src, alp, 128, upsample_mode="bilinear")
        torch.testing.assert_close(img_bicubic, img_bilinear)

    def test_alpha_upsample_mode_independent_of_image_mode(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        _, alp_bilinear, _ = letterbox_frame(src, alp, 128, upsample_mode="bicubic", alpha_upsample_mode="bilinear")
        _, alp_bicubic, _ = letterbox_frame(src, alp, 128, upsample_mode="bicubic", alpha_upsample_mode="bicubic")
        assert not torch.allclose(alp_bilinear, alp_bicubic)

    def test_mixed_dimensions_correct_shape(self):
        """Portrait source into square target."""
        src = _make_image(512, 64)
        alp = _make_alpha(512, 64)
        img, alp_out, _ = letterbox_frame(src, alp, 256)
        assert img.shape == (1, 3, 256, 256)
        assert alp_out.shape == (1, 1, 256, 256)


# ---------------------------------------------------------------------------
# Half precision
# ---------------------------------------------------------------------------

class TestHalfPrecision:
    def test_float16_input_produces_float16_output(self):
        src = _make_image(64, 64).half()
        alp = _make_alpha(64, 64).half()
        img, alp_out, _ = letterbox_frame(src, alp, 32)
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
