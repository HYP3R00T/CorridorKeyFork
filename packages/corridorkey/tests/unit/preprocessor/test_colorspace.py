"""Unit tests for corridorkey.stages.preprocessor.colorspace."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corridorkey.stages.preprocessor.colorspace import linear_to_srgb, linear_to_srgb_numpy


def _img(value: float, shape=(1, 3, 1, 1)) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def _arr(value: float, shape=(4, 4, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


class TestLinearToSrgb:
    def test_returns_float32(self):
        result = linear_to_srgb(_img(0.5))
        assert result.dtype == torch.float32

    def test_output_shape_unchanged(self):
        img = torch.rand(1, 3, 16, 16, dtype=torch.float32)
        assert linear_to_srgb(img).shape == img.shape

    def test_zero_maps_to_zero(self):
        result = linear_to_srgb(_img(0.0))
        assert result[0, 0, 0, 0].item() == pytest.approx(0.0)

    def test_one_maps_to_one(self):
        result = linear_to_srgb(_img(1.0))
        assert result[0, 0, 0, 0].item() == pytest.approx(1.0)

    def test_linear_segment_below_threshold(self):
        # Values <= 0.0031308 use the linear segment: out = in * 12.92
        val = 0.001
        result = linear_to_srgb(_img(val))
        assert result[0, 0, 0, 0].item() == pytest.approx(val * 12.92, abs=1e-4)

    def test_power_segment_above_threshold(self):
        # Values > 0.0031308 use: 1.055 * in^(1/2.4) - 0.055
        val = 0.5
        result = linear_to_srgb(_img(val))
        expected = 1.055 * (val ** (1.0 / 2.4)) - 0.055
        assert result[0, 0, 0, 0].item() == pytest.approx(expected, rel=1e-4)

    def test_output_clamped_to_0_1(self):
        img = torch.tensor([[[[2.0]], [[-0.5]], [[0.5]]]], dtype=torch.float32)
        result = linear_to_srgb(img)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_monotonically_increasing(self):
        vals = torch.linspace(0.0, 1.0, 100, dtype=torch.float32).view(1, 1, 1, 100).expand(1, 3, 1, 100)
        result = linear_to_srgb(vals)[0, 0, 0]
        assert torch.all(result.diff() >= 0)


class TestLinearToSrgbNumpy:
    def test_returns_float32(self):
        result = linear_to_srgb_numpy(_arr(0.5))
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self):
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        assert linear_to_srgb_numpy(arr).shape == arr.shape

    def test_zero_maps_to_zero(self):
        result = linear_to_srgb_numpy(_arr(0.0))
        assert result[0, 0, 0] == pytest.approx(0.0)

    def test_one_maps_to_one(self):
        result = linear_to_srgb_numpy(_arr(1.0))
        assert result[0, 0, 0] == pytest.approx(1.0)

    def test_linear_segment_below_threshold(self):
        val = 0.001
        result = linear_to_srgb_numpy(_arr(val))
        assert result[0, 0, 0] == pytest.approx(val * 12.92, abs=1e-4)

    def test_power_segment_above_threshold(self):
        val = 0.5
        result = linear_to_srgb_numpy(_arr(val))
        expected = 1.055 * (val ** (1.0 / 2.4)) - 0.055
        assert result[0, 0, 0] == pytest.approx(expected, rel=1e-4)

    def test_output_clamped_to_0_1(self):
        arr = np.array([[[2.0, -0.5, 0.5]]], dtype=np.float32)
        result = linear_to_srgb_numpy(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_matches_tensor_version(self):
        """numpy (LUT) and tensor (piecewise) versions must be numerically close.

        The LUT has 65536 entries giving <0.002% error — we allow atol=1e-3
        to account for the quantisation difference between the two paths.
        """
        arr = np.random.rand(8, 8, 3).astype(np.float32)
        np_result = linear_to_srgb_numpy(arr)
        # Convert to [1, 3, H, W] tensor, run tensor version, convert back
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        t_result = linear_to_srgb(t).squeeze(0).permute(1, 2, 0).numpy()
        np.testing.assert_allclose(np_result, t_result, atol=1e-3)
