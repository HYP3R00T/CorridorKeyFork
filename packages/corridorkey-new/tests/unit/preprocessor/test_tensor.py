"""Unit tests for corridorkey_new.stages.preprocessor.tensor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corridorkey_new.stages.preprocessor.tensor import to_tensors


def _make_image(h: int = 32, w: int = 32) -> np.ndarray:
    return np.random.rand(h, w, 3).astype(np.float32)


def _make_alpha(h: int = 32, w: int = 32) -> np.ndarray:
    return np.random.rand(h, w, 1).astype(np.float32)


class TestToTensors:
    def test_output_shapes(self):
        img_t, alp_t = to_tensors(_make_image(32, 32), _make_alpha(32, 32), "cpu")
        assert img_t.shape == (1, 3, 32, 32)
        assert alp_t.shape == (1, 1, 32, 32)

    def test_output_dtype_float32(self):
        img_t, alp_t = to_tensors(_make_image(), _make_alpha(), "cpu")
        assert img_t.dtype == torch.float32
        assert alp_t.dtype == torch.float32

    def test_output_on_cpu(self):
        img_t, alp_t = to_tensors(_make_image(), _make_alpha(), "cpu")
        assert img_t.device.type == "cpu"
        assert alp_t.device.type == "cpu"

    def test_bgr_false_preserves_channel_order(self):
        """bgr=False — channels should come through as-is."""
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:, :, 0] = 0.1  # channel 0
        image[:, :, 1] = 0.2  # channel 1
        image[:, :, 2] = 0.3  # channel 2
        img_t, _ = to_tensors(image, _make_alpha(4, 4), "cpu", bgr=False)
        assert img_t[0, 0, 0, 0].item() == pytest.approx(0.1)
        assert img_t[0, 1, 0, 0].item() == pytest.approx(0.2)
        assert img_t[0, 2, 0, 0].item() == pytest.approx(0.3)

    def test_bgr_true_reorders_to_rgb_on_device(self):
        """bgr=True — channel 2 (R in BGR) must become channel 0 in the tensor."""
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:, :, 2] = 1.0  # red in BGR is index 2
        img_t, _ = to_tensors(image, _make_alpha(4, 4), "cpu", bgr=True)
        # After BGR→RGB reorder, red should be at tensor channel 0
        assert img_t[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert img_t[0, 1, 0, 0].item() == pytest.approx(0.0)
        assert img_t[0, 2, 0, 0].item() == pytest.approx(0.0)

    def test_bgr_reorder_matches_manual_flip(self):
        """bgr=True result must equal manually flipping channels on CPU."""
        image = np.random.rand(8, 8, 3).astype(np.float32)
        img_bgr, _ = to_tensors(image, _make_alpha(8, 8), "cpu", bgr=True)
        img_rgb, _ = to_tensors(image[:, :, ::-1].copy(), _make_alpha(8, 8), "cpu", bgr=False)
        torch.testing.assert_close(img_bgr, img_rgb)

    def test_alpha_values_preserved_through_combined_transfer(self):
        """Alpha channel must survive the combined [4, H, W] PCIe transfer unchanged."""
        image = _make_image(4, 4)
        alpha = np.random.rand(4, 4, 1).astype(np.float32)
        _, alp_t = to_tensors(image, alpha, "cpu")
        expected = torch.from_numpy(alpha.transpose(2, 0, 1)).unsqueeze(0)
        torch.testing.assert_close(alp_t, expected)
