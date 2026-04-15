"""Unit tests for corridorkey.stages.postprocessor.resize."""

from __future__ import annotations

import numpy as np
import torch
from corridorkey.stages.postprocessor.resize import resize_to_source, resize_to_source_gpu, tensor_to_numpy_hwc


class TestTensorToNumpyHwc:
    def test_shape_conversion(self):
        """[1, C, H, W] tensor is converted to [H, W, C] numpy array."""
        t = torch.zeros(1, 3, 16, 16)
        out = tensor_to_numpy_hwc(t)
        assert out.shape == (16, 16, 3)

    def test_single_channel(self):
        """Single-channel [1, 1, H, W] tensor produces [H, W, 1] output."""
        t = torch.zeros(1, 1, 8, 8)
        out = tensor_to_numpy_hwc(t)
        assert out.shape == (8, 8, 1)

    def test_dtype_float32(self):
        """Output is always float32 regardless of input dtype."""
        t = torch.ones(1, 3, 4, 4, dtype=torch.float16)
        out = tensor_to_numpy_hwc(t)
        assert out.dtype == np.float32

    def test_values_preserved(self):
        """Pixel values are preserved through the conversion."""
        t = torch.full((1, 1, 2, 2), 0.5)
        out = tensor_to_numpy_hwc(t)
        assert np.allclose(out, 0.5)


class TestResizeToSource:
    def test_output_shapes(self):
        """Upscaling produces arrays at the requested target resolution."""
        alpha = torch.zeros(1, 1, 16, 16)
        fg = torch.zeros(1, 3, 16, 16)
        a_np, fg_np = resize_to_source(alpha, fg, 64, 64)
        assert a_np.shape == (64, 64, 1)
        assert fg_np.shape == (64, 64, 3)

    def test_upscale(self):
        """Upscaling from 8×8 to 32×32 produces the correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_np, fg_np = resize_to_source(alpha, fg, 32, 32)
        assert a_np.shape == (32, 32, 1)
        assert fg_np.shape == (32, 32, 3)

    def test_downscale(self):
        """Downscaling from 64×64 to 16×16 produces the correct output shape."""
        alpha = torch.zeros(1, 1, 64, 64)
        fg = torch.zeros(1, 3, 64, 64)
        a_np, fg_np = resize_to_source(alpha, fg, 16, 16)
        assert a_np.shape == (16, 16, 1)
        assert fg_np.shape == (16, 16, 3)

    def test_non_square(self):
        """Non-square target resolution is handled correctly."""
        alpha = torch.zeros(1, 1, 32, 32)
        fg = torch.zeros(1, 3, 32, 32)
        a_np, fg_np = resize_to_source(alpha, fg, 48, 64)
        assert a_np.shape == (48, 64, 1)
        assert fg_np.shape == (48, 64, 3)

    def test_values_clipped_to_0_1(self):
        """Values outside [0, 1] from interpolation are clipped."""
        alpha = torch.full((1, 1, 4, 4), 1.1)
        fg = torch.full((1, 3, 4, 4), -0.1)
        a_np, fg_np = resize_to_source(alpha, fg, 4, 4)
        assert a_np.max() <= 1.0
        assert fg_np.min() >= 0.0

    def test_same_size_passthrough(self):
        """When source size equals model size, values are returned unchanged."""
        alpha = torch.full((1, 1, 16, 16), 0.7)
        fg = torch.full((1, 3, 16, 16), 0.3)
        a_np, fg_np = resize_to_source(alpha, fg, 16, 16)
        assert np.allclose(a_np, 0.7, atol=1e-5)
        assert np.allclose(fg_np, 0.3, atol=1e-5)

    def test_fg_upsample_mode_bilinear(self):
        """fg_upsample_mode='bilinear' produces correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_np, fg_np = resize_to_source(alpha, fg, 32, 32, fg_upsample_mode="bilinear")
        assert fg_np.shape == (32, 32, 3)

    def test_fg_upsample_mode_bicubic(self):
        """fg_upsample_mode='bicubic' produces correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_np, fg_np = resize_to_source(alpha, fg, 32, 32, fg_upsample_mode="bicubic")
        assert fg_np.shape == (32, 32, 3)

    def test_alpha_upsample_mode_bilinear(self):
        """alpha_upsample_mode='bilinear' produces correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_np, fg_np = resize_to_source(alpha, fg, 32, 32, alpha_upsample_mode="bilinear")
        assert a_np.shape == (32, 32, 1)


class TestResizeToSourceGpu:
    def test_output_shapes_upscale(self):
        """Upscaling produces tensors at the requested target resolution."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 32, 32)
        assert a_out.shape == (1, 1, 32, 32)
        assert fg_out.shape == (1, 3, 32, 32)

    def test_output_shapes_downscale(self):
        """Downscaling produces tensors at the requested target resolution."""
        alpha = torch.zeros(1, 1, 64, 64)
        fg = torch.zeros(1, 3, 64, 64)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 16, 16)
        assert a_out.shape == (1, 1, 16, 16)
        assert fg_out.shape == (1, 3, 16, 16)

    def test_same_size_passthrough(self):
        """When source size equals model size, tensors are returned unchanged."""
        alpha = torch.full((1, 1, 16, 16), 0.7)
        fg = torch.full((1, 3, 16, 16), 0.3)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 16, 16)
        assert torch.allclose(a_out, torch.full_like(a_out, 0.7), atol=1e-5)
        assert torch.allclose(fg_out, torch.full_like(fg_out, 0.3), atol=1e-5)

    def test_output_dtype_float32(self):
        """Output tensors are float32."""
        alpha = torch.zeros(1, 1, 8, 8, dtype=torch.float16)
        fg = torch.zeros(1, 3, 8, 8, dtype=torch.float16)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 32, 32)
        assert a_out.dtype == torch.float32
        assert fg_out.dtype == torch.float32

    def test_values_clamped_to_0_1(self):
        """Output values are clamped to [0, 1]."""
        alpha = torch.full((1, 1, 4, 4), 1.5)
        fg = torch.full((1, 3, 4, 4), -0.5)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 8, 8)
        assert a_out.max().item() <= 1.0 + 1e-5
        assert fg_out.min().item() >= 0.0 - 1e-5

    def test_non_square_target(self):
        """Non-square target resolution is handled correctly."""
        alpha = torch.zeros(1, 1, 16, 16)
        fg = torch.zeros(1, 3, 16, 16)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 48, 64)
        assert a_out.shape == (1, 1, 48, 64)
        assert fg_out.shape == (1, 3, 48, 64)

    def test_fg_upsample_mode_bilinear(self):
        """fg_upsample_mode='bilinear' produces correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 32, 32, fg_upsample_mode="bilinear")
        assert fg_out.shape == (1, 3, 32, 32)

    def test_fg_upsample_mode_bicubic(self):
        """fg_upsample_mode='bicubic' produces correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 32, 32, fg_upsample_mode="bicubic")
        assert fg_out.shape == (1, 3, 32, 32)

    def test_alpha_upsample_mode_bilinear(self):
        """alpha_upsample_mode='bilinear' produces correct output shape."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 32, 32, alpha_upsample_mode="bilinear")
        assert a_out.shape == (1, 1, 32, 32)

    def test_lanczos4_maps_to_bicubic_for_gpu(self):
        """'lanczos4' mode (default) is mapped to bicubic for GPU interpolation."""
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        # Should not raise — lanczos4 is silently remapped to bicubic
        a_out, fg_out = resize_to_source_gpu(alpha, fg, 32, 32)
        assert a_out.shape == (1, 1, 32, 32)
