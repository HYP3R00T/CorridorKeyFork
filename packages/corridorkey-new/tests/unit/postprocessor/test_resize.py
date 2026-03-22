"""Unit tests for corridorkey_new.stages.postprocessor.resize."""

from __future__ import annotations

import numpy as np
import torch
from corridorkey_new.stages.postprocessor.resize import resize_to_source, tensor_to_numpy_hwc


class TestTensorToNumpyHwc:
    def test_shape_conversion(self):
        t = torch.zeros(1, 3, 16, 16)
        out = tensor_to_numpy_hwc(t)
        assert out.shape == (16, 16, 3)

    def test_single_channel(self):
        t = torch.zeros(1, 1, 8, 8)
        out = tensor_to_numpy_hwc(t)
        assert out.shape == (8, 8, 1)

    def test_dtype_float32(self):
        t = torch.ones(1, 3, 4, 4, dtype=torch.float16)
        out = tensor_to_numpy_hwc(t)
        assert out.dtype == np.float32

    def test_values_preserved(self):
        t = torch.full((1, 1, 2, 2), 0.5)
        out = tensor_to_numpy_hwc(t)
        assert np.allclose(out, 0.5)


class TestResizeToSource:
    def test_output_shapes(self):
        alpha = torch.zeros(1, 1, 16, 16)
        fg = torch.zeros(1, 3, 16, 16)
        a_np, fg_np = resize_to_source(alpha, fg, 64, 64)
        assert a_np.shape == (64, 64, 1)
        assert fg_np.shape == (64, 64, 3)

    def test_upscale(self):
        alpha = torch.zeros(1, 1, 8, 8)
        fg = torch.zeros(1, 3, 8, 8)
        a_np, fg_np = resize_to_source(alpha, fg, 32, 32)
        assert a_np.shape == (32, 32, 1)
        assert fg_np.shape == (32, 32, 3)

    def test_downscale(self):
        alpha = torch.zeros(1, 1, 64, 64)
        fg = torch.zeros(1, 3, 64, 64)
        a_np, fg_np = resize_to_source(alpha, fg, 16, 16)
        assert a_np.shape == (16, 16, 1)
        assert fg_np.shape == (16, 16, 3)

    def test_non_square(self):
        alpha = torch.zeros(1, 1, 32, 32)
        fg = torch.zeros(1, 3, 32, 32)
        a_np, fg_np = resize_to_source(alpha, fg, 48, 64)
        assert a_np.shape == (48, 64, 1)
        assert fg_np.shape == (48, 64, 3)

    def test_values_clipped_to_0_1(self):
        # Values slightly outside [0,1] from interpolation should be clipped.
        alpha = torch.full((1, 1, 4, 4), 1.1)
        fg = torch.full((1, 3, 4, 4), -0.1)
        a_np, fg_np = resize_to_source(alpha, fg, 4, 4)
        assert a_np.max() <= 1.0
        assert fg_np.min() >= 0.0

    def test_same_size_passthrough(self):
        alpha = torch.full((1, 1, 16, 16), 0.7)
        fg = torch.full((1, 3, 16, 16), 0.3)
        a_np, fg_np = resize_to_source(alpha, fg, 16, 16)
        assert np.allclose(a_np, 0.7, atol=1e-5)
        assert np.allclose(fg_np, 0.3, atol=1e-5)
