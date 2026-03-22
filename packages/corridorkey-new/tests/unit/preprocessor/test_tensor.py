"""Unit tests for corridorkey_new.stages.preprocessor.tensor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corridorkey_new.stages.preprocessor.tensor import to_tensor, to_tensors


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


class TestToTensor:
    def test_output_shape(self):
        t = to_tensor(_make_image(32, 32), _make_alpha(32, 32), "cpu")
        assert t.shape == (1, 4, 32, 32)

    def test_output_dtype_float32(self):
        t = to_tensor(_make_image(), _make_alpha(), "cpu")
        assert t.dtype == torch.float32

    def test_output_on_cpu(self):
        t = to_tensor(_make_image(), _make_alpha(), "cpu")
        assert t.device.type == "cpu"

    def test_channel_order_image_then_alpha(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        image[:, :, 0] = 0.1
        image[:, :, 1] = 0.2
        image[:, :, 2] = 0.3
        t = to_tensor(image, alpha, "cpu")
        assert t[0, 0, 0, 0].item() == pytest.approx(0.1)
        assert t[0, 1, 0, 0].item() == pytest.approx(0.2)
        assert t[0, 2, 0, 0].item() == pytest.approx(0.3)
        assert t[0, 3, 0, 0].item() == pytest.approx(1.0)

    def test_non_square_input(self):
        t = to_tensor(_make_image(16, 32), _make_alpha(16, 32), "cpu")
        assert t.shape == (1, 4, 16, 32)

    @pytest.mark.gpu
    def test_output_on_cuda(self):
        t = to_tensor(_make_image(), _make_alpha(), "cuda")
        assert t.device.type == "cuda"
