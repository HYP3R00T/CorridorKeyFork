"""Unit tests for the GPU despill path and auto-dispatch.

Tests run on CPU tensors (no CUDA required) to verify correctness of the
GPU implementation logic. The auto-dispatch tests verify routing behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corridorkey.stages.postprocessor.despill import remove_spill, remove_spill_auto, remove_spill_gpu


def _bchw(h: int = 8, w: int = 8, r: float = 0.2, g: float = 0.8, b: float = 0.2) -> torch.Tensor:
    """Return a [1, 3, H, W] float32 tensor with uniform RGB values."""
    t = torch.zeros(1, 3, h, w, dtype=torch.float32)
    t[:, 0] = r
    t[:, 1] = g
    t[:, 2] = b
    return t


class TestRemoveSpillGpu:
    def test_zero_strength_returns_input_unchanged(self):
        fg = _bchw()
        out = remove_spill_gpu(fg, strength=0.0)
        assert torch.equal(out, fg)

    def test_full_strength_reduces_green(self):
        fg = _bchw(g=1.0, r=0.0, b=0.0)  # pure green
        out = remove_spill_gpu(fg, strength=1.0)
        assert out[:, 1].mean() < fg[:, 1].mean()

    def test_output_shape_preserved(self):
        fg = _bchw(h=16, w=24)
        out = remove_spill_gpu(fg, strength=1.0)
        assert out.shape == fg.shape

    def test_output_dtype_preserved(self):
        fg = _bchw()
        out = remove_spill_gpu(fg, strength=1.0)
        assert out.dtype == torch.float32

    def test_no_spill_image_unchanged(self):
        # Neutral grey — no green spill, result should be identical
        fg = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)
        out = remove_spill_gpu(fg, strength=1.0)
        assert torch.allclose(out, fg, atol=1e-5)

    def test_partial_strength_blends(self):
        fg = _bchw(g=1.0, r=0.0, b=0.0)
        full = remove_spill_gpu(fg, strength=1.0)
        half = remove_spill_gpu(fg, strength=0.5)
        assert half[:, 1].mean() > full[:, 1].mean()
        assert half[:, 1].mean() < fg[:, 1].mean()

    def test_does_not_mutate_input(self):
        fg = _bchw(g=1.0, r=0.0, b=0.0)
        original = fg.clone()
        remove_spill_gpu(fg, strength=1.0)
        assert torch.equal(fg, original)

    def test_gpu_matches_cpu_full_strength(self):
        """GPU result must match CPU numpy result within float32 tolerance."""
        np_rng = np.random.default_rng(42)
        fg_np = np_rng.random((16, 16, 3)).astype(np.float32)

        cpu_out = remove_spill(fg_np, strength=1.0)

        fg_t = torch.from_numpy(fg_np).permute(2, 0, 1).unsqueeze(0)
        gpu_out_t = remove_spill_gpu(fg_t, strength=1.0)
        gpu_out = gpu_out_t.squeeze(0).permute(1, 2, 0).numpy()

        assert np.allclose(cpu_out, gpu_out, atol=1e-5), f"Max diff: {np.abs(cpu_out - gpu_out).max()}"

    def test_gpu_matches_cpu_partial_strength(self):
        """GPU partial-strength result must match CPU within float32 tolerance."""
        np_rng = np.random.default_rng(7)
        fg_np = np_rng.random((16, 16, 3)).astype(np.float32)

        cpu_out = remove_spill(fg_np, strength=0.6)

        fg_t = torch.from_numpy(fg_np).permute(2, 0, 1).unsqueeze(0)
        gpu_out_t = remove_spill_gpu(fg_t, strength=0.6)
        gpu_out = gpu_out_t.squeeze(0).permute(1, 2, 0).numpy()

        assert np.allclose(cpu_out, gpu_out, atol=1e-5)


class TestRemoveSpillAuto:
    def test_cpu_device_uses_numpy_path(self):
        np_rng = np.random.default_rng(1)
        fg = np_rng.random((8, 8, 3)).astype(np.float32)
        out = remove_spill_auto(fg, strength=1.0, device="cpu")
        expected = remove_spill(fg, strength=1.0)
        assert np.allclose(out, expected, atol=1e-6)

    def test_none_device_uses_cpu_path(self):
        np_rng = np.random.default_rng(2)
        fg = np_rng.random((8, 8, 3)).astype(np.float32)
        out = remove_spill_auto(fg, strength=1.0, device=None)
        expected = remove_spill(fg, strength=1.0)
        assert np.allclose(out, expected, atol=1e-6)

    def test_output_is_numpy_float32(self):
        fg = np.random.default_rng(3).random((8, 8, 3)).astype(np.float32)
        out = remove_spill_auto(fg, strength=1.0, device="cpu")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32

    def test_output_shape_preserved(self):
        fg = np.random.default_rng(4).random((16, 24, 3)).astype(np.float32)
        out = remove_spill_auto(fg, strength=1.0, device="cpu")
        assert out.shape == fg.shape

    def test_zero_strength_returns_unchanged(self):
        fg = np.random.default_rng(5).random((8, 8, 3)).astype(np.float32)
        out = remove_spill_auto(fg, strength=0.0, device="cpu")
        assert np.array_equal(out, fg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_returns_numpy(self):
        fg = np.random.default_rng(6).random((8, 8, 3)).astype(np.float32)
        out = remove_spill_auto(fg, strength=1.0, device="cuda")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == fg.shape
