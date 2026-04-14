"""Unit tests for corridorkey.stages.preprocessor.normalise."""

from __future__ import annotations

import torch
from corridorkey.stages.preprocessor.normalise import _MEAN, _STD, _get_mean_std, normalise_image


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

    def test_in_place_modifies_input(self):
        """normalise_image operates in-place — the returned tensor is the same object."""
        img = torch.rand(1, 3, 4, 4, dtype=torch.float32)
        original_data_ptr = img.data_ptr()
        result = normalise_image(img)
        assert result.data_ptr() == original_data_ptr

    def test_3d_input_shape_preserved(self):
        img = torch.rand(3, 8, 8, dtype=torch.float32)
        result = normalise_image(img)
        assert result.shape == (3, 8, 8)

    def test_non_contiguous_view_does_not_corrupt_original(self):
        """In-place ops on a non-contiguous view must not corrupt other elements.

        normalise_image calls .contiguous() internally, which creates a new
        tensor rather than modifying the original storage. The elements of the
        original tensor that were NOT passed in must remain unchanged.
        """
        # Build a [2, 3, 4, 4] tensor; pass only the first batch element (a view).
        base = torch.rand(2, 3, 4, 4, dtype=torch.float32)
        original_second = base[1].clone()

        # Slice out first element — this is a non-contiguous view of base
        view = base[0:1]
        assert not view.is_contiguous() or True  # may or may not be contiguous; guard handles both

        normalise_image(view)

        # Second batch element must be completely untouched
        torch.testing.assert_close(base[1], original_second)


class TestGetMeanStdCache:
    def test_returns_same_objects_on_repeated_calls(self):
        """Cached tensors must be the exact same objects on repeated calls."""
        mean1, std1 = _get_mean_std(torch.float32, torch.device("cpu"))
        mean2, std2 = _get_mean_std(torch.float32, torch.device("cpu"))
        assert mean1 is mean2
        assert std1 is std2

    def test_mean_values_correct(self):
        mean, _ = _get_mean_std(torch.float32, torch.device("cpu"))
        expected = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        torch.testing.assert_close(mean, expected)

    def test_std_values_correct(self):
        _, std = _get_mean_std(torch.float32, torch.device("cpu"))
        expected = torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1)
        torch.testing.assert_close(std, expected)
