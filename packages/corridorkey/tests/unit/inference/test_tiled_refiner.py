"""Unit tests for _run_refiner_tiled in corridorkey.stages.inference.orchestrator.

Tests cover:
  - Output shape matches input shape for various resolutions
  - Single-tile images (image smaller than tile_size) produce correct output
  - Exact-multiple images (no edge tiles) produce correct output
  - Non-multiple images (edge tiles with padding) produce correct output
  - Bypass flag is always reset to False after each tile, even on exception
  - dtype is preserved (float32 in → float32 out)
  - A constant refiner (all-zeros delta) produces all-zeros output
  - A constant refiner (all-ones delta) produces all-ones output (blend normalisation)
  - Overlap larger than half tile_size is clamped safely
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from corridorkey.stages.inference.orchestrator import _run_refiner_tiled, _TiledRefinerState

# ---------------------------------------------------------------------------
# Minimal refiner stubs
# ---------------------------------------------------------------------------


class _ZeroRefiner(nn.Module):
    """Always returns all-zeros delta — output should be all zeros."""

    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return torch.zeros(img.shape[0], 4, img.shape[2], img.shape[3], device=img.device)


class _OnesRefiner(nn.Module):
    """Always returns all-ones delta — after blend normalisation output should be all ones."""

    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return torch.ones(img.shape[0], 4, img.shape[2], img.shape[3], device=img.device)


class _EchoRefiner(nn.Module):
    """Returns the coarse input as delta — useful for checking values pass through."""

    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return coarse.clone()


class _RaisingRefiner(nn.Module):
    """Raises on the second call — tests that bypass is reset after an exception."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        if self._calls >= 2:
            raise RuntimeError("deliberate test error")
        return torch.zeros(img.shape[0], 4, img.shape[2], img.shape[3])


def _state() -> _TiledRefinerState:
    return _TiledRefinerState()


def _rgb(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 3, h, w)


def _coarse(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 4, h, w)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestTiledRefinerOutputShape:
    @pytest.mark.parametrize(
        "h,w",
        [
            (512, 512),  # exact single tile
            (256, 256),  # smaller than tile
            (1024, 1024),  # exact 2×2 tiles
            (600, 800),  # non-multiple, non-square
            (513, 513),  # one pixel over tile boundary
        ],
    )
    def test_output_shape_matches_input(self, h: int, w: int):
        refiner = _ZeroRefiner()
        result = _run_refiner_tiled(refiner, _rgb(h, w), _coarse(h, w), _state(), tile_size=512, overlap=128)
        assert result.shape == (1, 4, h, w)

    def test_batch_size_preserved(self):
        refiner = _ZeroRefiner()
        rgb = torch.rand(2, 3, 512, 512)
        coarse = torch.rand(2, 4, 512, 512)
        result = _run_refiner_tiled(refiner, rgb, coarse, _state(), tile_size=512, overlap=128)
        assert result.shape == (2, 4, 512, 512)


# ---------------------------------------------------------------------------
# Zero refiner — output must be all zeros
# ---------------------------------------------------------------------------


class TestZeroRefiner:
    def test_single_tile_all_zeros(self):
        refiner = _ZeroRefiner()
        result = _run_refiner_tiled(refiner, _rgb(256, 256), _coarse(256, 256), _state(), tile_size=512, overlap=128)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_exact_tile_all_zeros(self):
        refiner = _ZeroRefiner()
        result = _run_refiner_tiled(refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_multi_tile_all_zeros(self):
        refiner = _ZeroRefiner()
        result = _run_refiner_tiled(
            refiner, _rgb(1024, 1024), _coarse(1024, 1024), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_non_multiple_all_zeros(self):
        refiner = _ZeroRefiner()
        result = _run_refiner_tiled(refiner, _rgb(600, 800), _coarse(600, 800), _state(), tile_size=512, overlap=128)
        assert torch.allclose(result, torch.zeros_like(result))


# ---------------------------------------------------------------------------
# Ones refiner — blend normalisation must produce all ones
# ---------------------------------------------------------------------------


class TestOnesRefiner:
    def test_single_tile_all_ones(self):
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(refiner, _rgb(256, 256), _coarse(256, 256), _state(), tile_size=512, overlap=128)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_exact_tile_all_ones(self):
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_multi_tile_all_ones(self):
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(
            refiner, _rgb(1024, 1024), _coarse(1024, 1024), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_non_multiple_all_ones(self):
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(refiner, _rgb(600, 800), _coarse(600, 800), _state(), tile_size=512, overlap=128)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


class TestDtypePreservation:
    def test_float32_in_float32_out(self):
        refiner = _ZeroRefiner()
        rgb = _rgb(256, 256).float()
        coarse = _coarse(256, 256).float()
        result = _run_refiner_tiled(refiner, rgb, coarse, _state(), tile_size=512, overlap=128)
        assert result.dtype == torch.float32

    def test_float16_in_float32_out(self):
        # Tiled pass upcasts to float32 internally; result is cast back to input dtype.
        refiner = _ZeroRefiner()
        rgb = _rgb(256, 256).half()
        coarse = _coarse(256, 256).half()
        result = _run_refiner_tiled(refiner, rgb, coarse, _state(), tile_size=512, overlap=128)
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Bypass flag safety
# ---------------------------------------------------------------------------


class TestBypassFlag:
    def test_bypass_false_after_normal_run(self):
        state = _state()
        refiner = _ZeroRefiner()
        _run_refiner_tiled(refiner, _rgb(256, 256), _coarse(256, 256), state, tile_size=512, overlap=128)
        assert state.bypass is False

    def test_bypass_false_after_refiner_exception(self):
        """bypass must be reset to False even when the refiner raises."""
        state = _state()
        refiner = _RaisingRefiner()
        with pytest.raises(RuntimeError, match="deliberate test error"):
            _run_refiner_tiled(refiner, _rgb(1024, 512), _coarse(1024, 512), state, tile_size=512, overlap=128)
        assert state.bypass is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_overlap_larger_than_half_tile_clamped(self):
        """Overlap > tile_size//2 - 1 must not crash — safe_overlap clamps it."""
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=300)
        assert result.shape == (1, 4, 512, 512)

    def test_zero_overlap(self):
        """overlap=0 means no blending — tiles are stitched directly."""
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=256, overlap=0)
        assert result.shape == (1, 4, 512, 512)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_image_smaller_than_tile(self):
        """Image smaller than tile_size — single padded tile, cropped back."""
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(refiner, _rgb(100, 150), _coarse(100, 150), _state(), tile_size=512, overlap=128)
        assert result.shape == (1, 4, 100, 150)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


# ---------------------------------------------------------------------------
# refiner_scale
# ---------------------------------------------------------------------------


class TestRefinerScale:
    def test_scale_zero_produces_all_zeros(self):
        """refiner_scale=0.0 must zero out all delta output."""
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(
            refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128, refiner_scale=0.0
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_scale_one_unchanged(self):
        """refiner_scale=1.0 must produce the same result as the default."""
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(
            refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128, refiner_scale=1.0
        )
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_scale_half_produces_half(self):
        """refiner_scale=0.5 must halve the delta output."""
        refiner = _OnesRefiner()
        result = _run_refiner_tiled(
            refiner, _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128, refiner_scale=0.5
        )
        expected = torch.full_like(result, 0.5)
        assert torch.allclose(result, expected, atol=1e-5)
