"""Unit tests for corridorkey_new.preprocessor.normalise."""

from __future__ import annotations

import torch
from corridorkey_new.preprocessor.normalise import _MEAN, _STD, normalise_image


class TestNormaliseImage:
    def test_returns_float32(self):
        img = torch.ones(1, 3, 4, 4, dtype=torch.float32) * 0.5
        assert normalise_image(img).dtype == torch.float32

    def test_shape_unchanged(self):
        img = torch.rand(1, 3, 16, 16, dtype=torch.float32)
        assert normalise_image(img).shape == img.shape

    def test_mean_pixel_normalises_to_zero(self):
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        img = mean.expand(1, 3, 1, 1).clone()
        result = normalise_image(img)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)

    def test_known_value(self):
        # pixel = mean + std should normalise to 1.0 per channel
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1)
        img = (mean + std).expand(1, 3, 1, 1).clone()
        result = normalise_image(img)
        torch.testing.assert_close(result, torch.ones_like(result), atol=1e-5, rtol=0)

    def test_output_can_be_negative(self):
        img = torch.zeros(1, 3, 4, 4, dtype=torch.float32)
        result = normalise_image(img)
        assert result.min().item() < 0.0

    def test_output_can_exceed_one(self):
        img = torch.ones(1, 3, 4, 4, dtype=torch.float32)
        result = normalise_image(img)
        assert result.max().item() > 1.0
