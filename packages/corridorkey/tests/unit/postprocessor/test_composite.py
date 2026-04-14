"""Unit tests for corridorkey.stages.postprocessor.composite."""

from __future__ import annotations

import numpy as np
from corridorkey.stages.postprocessor.composite import apply_source_passthrough, make_preview


class TestMakePreview:
    def test_output_shape(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.zeros((32, 32, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=8)
        assert out.shape == (32, 32, 3)

    def test_output_dtype_float32(self):
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.zeros((16, 16, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=4)
        assert out.dtype == np.float32

    def test_fully_opaque_shows_fg(self):
        # alpha=1 everywhere → composite should equal fg
        fg = np.full((16, 16, 3), 0.8, dtype=np.float32)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=4)
        assert np.allclose(out, fg, atol=1e-5)

    def test_fully_transparent_shows_background(self):
        # alpha=0 everywhere → composite should equal the checkerboard bg
        fg = np.ones((16, 16, 3), dtype=np.float32)
        alpha = np.zeros((16, 16, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=4)
        # fg is all-ones but alpha is 0, so output must be < 1 (checkerboard is grey)
        assert out.max() < 1.0

    def test_values_in_range(self):
        fg = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        alpha = np.random.default_rng(1).random((32, 32, 1)).astype(np.float32)
        out = make_preview(fg, alpha, checker_size=8)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5


class TestApplySourcePassthrough:
    def test_output_shape(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.8, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source)
        assert out.shape == (32, 32, 3)

    def test_output_dtype_float32(self):
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        source = np.ones((16, 16, 3), dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source)
        assert out.dtype == np.float32

    def test_fully_opaque_uses_source(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.9, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert np.allclose(out, source, atol=1e-5)

    def test_zero_erode_zero_blur(self):
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        source = np.full((16, 16, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert out.shape == (16, 16, 3)


class TestApplySourcePassthroughEvenKernelBranch:
    def test_blur_px_zero_skips_blur(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert out.shape == (32, 32, 3)
        assert out.dtype == np.float32

    def test_blur_px_one(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=1)
        assert out.shape == (32, 32, 3)

    def test_blur_px_two(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=2)
        assert out.shape == (32, 32, 3)

    def test_erode_zero_blur_zero_output_equals_source(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert np.allclose(out, source, atol=1e-4)

    def test_alpha_2d_input_accepted(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha_2d = np.ones((32, 32), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha_2d, source, edge_erode_px=0, edge_blur_px=0)
        assert out.shape == (32, 32, 3)

    def test_output_dtype_float32(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=1, edge_blur_px=3)
        assert out.dtype == np.float32

    def test_output_values_clipped_to_unit_range(self):
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5
