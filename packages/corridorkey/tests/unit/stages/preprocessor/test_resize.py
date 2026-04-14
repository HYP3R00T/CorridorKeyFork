"""Unit tests for corridorkey.stages.preprocessor.resize."""

from __future__ import annotations

import torch
from corridorkey.stages.preprocessor.resize import (
    DEFAULT_IMAGE_UPSAMPLE_MODE,
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

    def test_extreme_ratio_produces_correct_shape(self):
        # Large downscale — bilinear handles any ratio in one pass
        src = _make_image(4320, 7680)
        alp = _make_alpha(4320, 7680)
        img, alp_out = resize_frame(src, alp, 512)
        assert img.shape == (1, 3, 512, 512)
        assert alp_out.shape == (1, 1, 512, 512)


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


class TestBilinearResize:
    def test_default_image_upsample_mode_is_bicubic(self):
        # DEFAULT_IMAGE_UPSAMPLE_MODE is kept for API compatibility
        assert DEFAULT_IMAGE_UPSAMPLE_MODE == "bicubic"

    def test_ignored_params_do_not_change_output(self):
        # image_upsample_mode, sharpen_strength, is_srgb are all ignored —
        # resize always uses plain bilinear to match the reference pipeline.
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img_a, _ = resize_frame(src, alp, 128, image_upsample_mode="bicubic", sharpen_strength=0.5, is_srgb=True)
        img_b, _ = resize_frame(src, alp, 128, image_upsample_mode="bilinear", sharpen_strength=0.0, is_srgb=False)
        torch.testing.assert_close(img_a, img_b)

    def test_output_in_range_downscale(self):
        src = _make_image(512, 512)
        alp = _make_alpha(512, 512)
        img, _ = resize_frame(src, alp, 128)
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

    def test_output_in_range_upscale(self):
        src = _make_image(32, 32)
        alp = _make_alpha(32, 32)
        img, _ = resize_frame(src, alp, 128)
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0


class TestHalfPrecision:
    def test_float16_input_produces_float16_output(self):
        src = _make_image(64, 64).half()
        alp = _make_alpha(64, 64).half()
        img, alp_out = resize_frame(src, alp, 32)
        assert img.dtype == torch.float16
        assert alp_out.dtype == torch.float16
