"""Unit tests for the GPU despeckle path and auto-dispatch.

Tests run on CPU tensors (no CUDA required) to verify correctness of the
GPU implementation logic. The auto-dispatch tests verify routing behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corridorkey.stages.postprocessor.despeckle import (
    _build_cc_scramble_init,
    _connected_components_gpu,
    despeckle_alpha,
    despeckle_alpha_auto,
    despeckle_alpha_gpu,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alpha_with_speck(h: int = 64, w: int = 64) -> torch.Tensor:
    """[1, 1, H, W] float32 tensor: large region + tiny isolated speck."""
    t = torch.zeros(1, 1, h, w, dtype=torch.float32)
    t[0, 0, 30:55, 30:55] = 1.0  # large region (~625 px)
    t[0, 0, 2:4, 2:4] = 1.0  # tiny speck (4 px)
    return t


def _alpha_np_with_speck(h: int = 64, w: int = 64) -> np.ndarray:
    """[H, W, 1] float32 numpy: large region + tiny isolated speck."""
    a = np.zeros((h, w, 1), dtype=np.float32)
    a[30:55, 30:55, 0] = 1.0
    a[2:4, 2:4, 0] = 1.0
    return a


# ---------------------------------------------------------------------------
# _build_cc_scramble_init
# ---------------------------------------------------------------------------


class TestBuildCcScrambleInit:
    def test_output_shape(self):
        init = _build_cc_scramble_init(8, 8, "cpu")
        assert init.shape == (1, 1, 8, 8)

    def test_output_dtype_float32(self):
        init = _build_cc_scramble_init(4, 4, "cpu")
        assert init.dtype == torch.float32

    def test_all_values_positive(self):
        init = _build_cc_scramble_init(8, 8, "cpu")
        assert (init > 0).all()

    def test_all_values_unique(self):
        """Each pixel must get a unique label (no collisions for small grids)."""
        init = _build_cc_scramble_init(8, 8, "cpu")
        flat = init.view(-1)
        assert flat.unique().numel() == flat.numel()

    def test_cached(self):
        """Same (h, w, device) must return the same tensor object."""
        a = _build_cc_scramble_init(4, 4, "cpu")
        b = _build_cc_scramble_init(4, 4, "cpu")
        assert a is b


# ---------------------------------------------------------------------------
# _connected_components_gpu
# ---------------------------------------------------------------------------


class TestConnectedComponentsGpu:
    def test_single_component_gets_one_label(self):
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        mask[0, 0, 2:6, 2:6] = True
        comp = _connected_components_gpu(mask, max_iterations=20)
        labels_in_region = comp[mask].unique()
        assert labels_in_region.numel() == 1

    def test_two_separated_components_get_different_labels(self):
        mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        mask[0, 0, 1:4, 1:4] = True  # component A
        mask[0, 0, 12:15, 12:15] = True  # component B
        comp = _connected_components_gpu(mask, max_iterations=40)
        label_a = comp[0, 0, 2, 2].item()
        label_b = comp[0, 0, 13, 13].item()
        assert label_a != label_b

    def test_background_pixels_are_zero(self):
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        mask[0, 0, 3:5, 3:5] = True
        comp = _connected_components_gpu(mask, max_iterations=20)
        bg = comp[~mask]
        assert (bg == 0).all()

    def test_output_shape_matches_input(self):
        mask = torch.zeros(1, 1, 12, 16, dtype=torch.bool)
        comp = _connected_components_gpu(mask, max_iterations=10)
        assert comp.shape == mask.shape


# ---------------------------------------------------------------------------
# despeckle_alpha_gpu
# ---------------------------------------------------------------------------


class TestDespeckleAlphaGpu:
    def test_zero_min_area_returns_unchanged(self):
        alpha = _alpha_with_speck()
        out = despeckle_alpha_gpu(alpha, min_area=0)
        assert torch.equal(out, alpha)

    def test_removes_small_region(self):
        alpha = _alpha_with_speck()
        out = despeckle_alpha_gpu(alpha, min_area=10, dilation=0, blur_size=0)
        # Speck (4 px) should be zeroed
        assert out[0, 0, 2:4, 2:4].max().item() == pytest.approx(0.0, abs=1e-5)

    def test_preserves_large_region(self):
        alpha = _alpha_with_speck()
        out = despeckle_alpha_gpu(alpha, min_area=10, dilation=0, blur_size=0)
        # Centre of large region must survive
        assert out[0, 0, 42, 42].item() > 0.5

    def test_output_shape_preserved(self):
        alpha = _alpha_with_speck()
        out = despeckle_alpha_gpu(alpha, min_area=10)
        assert out.shape == alpha.shape

    def test_output_dtype_float32(self):
        alpha = _alpha_with_speck()
        out = despeckle_alpha_gpu(alpha, min_area=10)
        assert out.dtype == torch.float32

    def test_all_zeros_unchanged(self):
        alpha = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        out = despeckle_alpha_gpu(alpha, min_area=10)
        assert torch.equal(out, alpha)

    def test_gpu_matches_cpu_no_dilation_no_blur(self):
        """GPU result must match CPU result when dilation=0 and blur_size=0."""
        alpha_np = _alpha_np_with_speck()
        cpu_out = despeckle_alpha(alpha_np, min_area=10, dilation=0, blur_size=0)

        alpha_t = torch.from_numpy(alpha_np[:, :, 0]).unsqueeze(0).unsqueeze(0)
        gpu_out_t = despeckle_alpha_gpu(alpha_t, min_area=10, dilation=0, blur_size=0)
        gpu_out = gpu_out_t.squeeze(0).permute(1, 2, 0).numpy()

        # Large region must be preserved in both
        assert cpu_out[42, 42, 0] > 0.5
        assert gpu_out[42, 42, 0] > 0.5
        # Speck must be removed in both
        assert cpu_out[2:4, 2:4, 0].max() == pytest.approx(0.0, abs=1e-5)
        assert gpu_out[2:4, 2:4, 0].max() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# despeckle_alpha_auto
# ---------------------------------------------------------------------------


class TestDespeckleAlphaAuto:
    def test_cpu_device_uses_numpy_path(self):
        alpha = _alpha_np_with_speck()
        out = despeckle_alpha_auto(alpha, min_area=10, dilation=0, blur_size=0, device="cpu")
        expected = despeckle_alpha(alpha, min_area=10, dilation=0, blur_size=0)
        assert np.allclose(out, expected, atol=1e-6)

    def test_none_device_uses_cpu_path(self):
        alpha = _alpha_np_with_speck()
        out = despeckle_alpha_auto(alpha, min_area=10, dilation=0, blur_size=0, device=None)
        expected = despeckle_alpha(alpha, min_area=10, dilation=0, blur_size=0)
        assert np.allclose(out, expected, atol=1e-6)

    def test_output_is_numpy_float32(self):
        alpha = _alpha_np_with_speck()
        out = despeckle_alpha_auto(alpha, min_area=10, device="cpu")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32

    def test_output_shape_preserved(self):
        alpha = _alpha_np_with_speck(32, 48)
        out = despeckle_alpha_auto(alpha, min_area=10, device="cpu")
        assert out.shape == alpha.shape

    def test_zero_min_area_returns_unchanged(self):
        alpha = _alpha_np_with_speck()
        out = despeckle_alpha_auto(alpha, min_area=0, device="cpu")
        assert np.array_equal(out, alpha)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_returns_numpy(self):
        alpha = _alpha_np_with_speck()
        out = despeckle_alpha_auto(alpha, min_area=10, device="cuda")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == alpha.shape
