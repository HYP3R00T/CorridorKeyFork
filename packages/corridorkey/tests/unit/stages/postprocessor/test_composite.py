"""Unit tests for corridorkey.stages.postprocessor.composite."""

from __future__ import annotations

import numpy as np
from corridorkey.stages.postprocessor.composite import (
    apply_source_passthrough,
    linear_to_srgb,
    make_preview,
    make_processed,
    srgb_to_linear,
)


class TestSrgbToLinear:
    def test_zero_maps_to_zero(self):
        """sRGB 0.0 maps to linear 0.0."""
        x = np.zeros((4, 4, 3), dtype=np.float32)
        out = srgb_to_linear(x)
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_one_maps_to_one(self):
        """sRGB 1.0 maps to linear 1.0."""
        x = np.ones((4, 4, 3), dtype=np.float32)
        out = srgb_to_linear(x)
        assert np.allclose(out, 1.0, atol=1e-4)

    def test_output_dtype_float32(self):
        """Output is always float32 regardless of input dtype."""
        x = np.full((2, 2, 1), 0.5, dtype=np.float32)
        out = srgb_to_linear(x)
        assert out.dtype == np.float32

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        x = np.zeros((8, 16, 3), dtype=np.float32)
        out = srgb_to_linear(x)
        assert out.shape == (8, 16, 3)

    def test_midpoint_is_darker_than_input(self):
        """sRGB 0.5 converts to a linear value less than 0.5 (gamma expansion)."""
        x = np.full((2, 2, 1), 0.5, dtype=np.float32)
        out = srgb_to_linear(x)
        assert out.mean() < 0.5

    def test_values_above_one_are_clipped(self):
        """Input values > 1.0 are clipped before LUT lookup."""
        x = np.full((2, 2, 1), 2.0, dtype=np.float32)
        out = srgb_to_linear(x)
        assert out.max() <= 1.0 + 1e-5

    def test_values_below_zero_are_clipped(self):
        """Input values < 0.0 are clipped before LUT lookup."""
        x = np.full((2, 2, 1), -1.0, dtype=np.float32)
        out = srgb_to_linear(x)
        assert out.min() >= 0.0


class TestLinearToSrgb:
    def test_zero_maps_to_zero(self):
        """Linear 0.0 maps to sRGB 0.0."""
        x = np.zeros((4, 4, 3), dtype=np.float32)
        out = linear_to_srgb(x)
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_one_maps_to_one(self):
        """Linear 1.0 maps to sRGB 1.0."""
        x = np.ones((4, 4, 3), dtype=np.float32)
        out = linear_to_srgb(x)
        assert np.allclose(out, 1.0, atol=1e-4)

    def test_midpoint_is_brighter_than_input(self):
        """Linear 0.5 converts to an sRGB value greater than 0.5 (gamma compression)."""
        x = np.full((2, 2, 1), 0.5, dtype=np.float32)
        out = linear_to_srgb(x)
        assert out.mean() > 0.5

    def test_roundtrip_is_identity(self):
        """srgb_to_linear followed by linear_to_srgb should recover the original value."""
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, 256, 1)
        roundtrip = linear_to_srgb(srgb_to_linear(x))
        assert np.allclose(roundtrip, x, atol=1e-3)


class TestMakeProcessed:
    def test_output_shape(self):
        """Output is [H, W, 4] — three FG channels plus alpha."""
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.zeros((16, 16, 1), dtype=np.float32)
        out = make_processed(fg, alpha)
        assert out.shape == (16, 16, 4)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        fg = np.zeros((8, 8, 3), dtype=np.float32)
        alpha = np.zeros((8, 8, 1), dtype=np.float32)
        out = make_processed(fg, alpha)
        assert out.dtype == np.float32

    def test_alpha_channel_is_fourth(self):
        """The fourth channel of the output equals the input alpha."""
        fg = np.zeros((8, 8, 3), dtype=np.float32)
        alpha = np.full((8, 8, 1), 0.7, dtype=np.float32)
        out = make_processed(fg, alpha)
        assert np.allclose(out[:, :, 3:4], alpha, atol=1e-5)

    def test_transparent_region_rgb_is_zero(self):
        """Where alpha=0, the premultiplied RGB channels must be zero."""
        fg = np.ones((8, 8, 3), dtype=np.float32)
        alpha = np.zeros((8, 8, 1), dtype=np.float32)
        out = make_processed(fg, alpha)
        assert np.allclose(out[:, :, :3], 0.0, atol=1e-6)

    def test_opaque_region_rgb_is_fg_linear(self):
        """Where alpha=1, the premultiplied RGB equals sRGB-to-linear(fg)."""
        fg = np.full((4, 4, 3), 0.5, dtype=np.float32)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        out = make_processed(fg, alpha)
        expected_linear = srgb_to_linear(fg)
        assert np.allclose(out[:, :, :3], expected_linear, atol=1e-4)

    def test_values_in_range(self):
        """All output values are in [0, 1]."""
        rng = np.random.default_rng(0)
        fg = rng.random((16, 16, 3)).astype(np.float32)
        alpha = rng.random((16, 16, 1)).astype(np.float32)
        out = make_processed(fg, alpha)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5


class TestMakePreview:
    def test_output_shape(self):
        """Output shape is [H, W, 3]."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.zeros((32, 32, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=8)
        assert out.shape == (32, 32, 3)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.zeros((16, 16, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=4)
        assert out.dtype == np.float32

    def test_fully_opaque_shows_fg(self):
        """When alpha=1 everywhere, the composite equals fg."""
        fg = np.full((16, 16, 3), 0.8, dtype=np.float32)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=4)
        assert np.allclose(out, fg, atol=1e-5)

    def test_fully_transparent_shows_background(self):
        """When alpha=0 everywhere, the composite shows the checkerboard (not fg)."""
        fg = np.ones((16, 16, 3), dtype=np.float32)
        alpha = np.zeros((16, 16, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=4)
        assert out.max() < 1.0

    def test_checkerboard_has_two_distinct_values(self):
        """The transparent background should contain two distinct grey values."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.zeros((32, 32, 1), dtype=np.float32)
        out = make_preview(fg, alpha, checker_size=8)
        unique_vals = np.unique(np.round(out[:, :, 0], 3))
        assert len(unique_vals) == 2

    def test_values_in_range(self):
        """All output values are in [0, 1]."""
        fg = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
        alpha = np.random.default_rng(1).random((32, 32, 1)).astype(np.float32)
        out = make_preview(fg, alpha, checker_size=8)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5


class TestApplySourcePassthrough:
    def test_output_shape(self):
        """Output shape matches fg input shape."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.8, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source)
        assert out.shape == (32, 32, 3)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        source = np.ones((16, 16, 3), dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source)
        assert out.dtype == np.float32

    def test_fully_opaque_uses_source(self):
        """With alpha=1 and no erosion/blur, output equals source."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.9, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert np.allclose(out, source, atol=1e-5)

    def test_transparent_region_uses_model_fg(self):
        """Where alpha < 0.95, the interior mask is 0 so model FG is used."""
        fg = np.full((32, 32, 3), 0.3, dtype=np.float32)
        alpha = np.zeros((32, 32, 1), dtype=np.float32)  # fully transparent
        source = np.full((32, 32, 3), 0.9, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert np.allclose(out, fg, atol=1e-5)

    def test_alpha_2d_input_accepted(self):
        """2D alpha [H, W] (no channel dim) is handled correctly."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha_2d = np.ones((32, 32), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha_2d, source, edge_erode_px=0, edge_blur_px=0)
        assert out.shape == (32, 32, 3)

    def test_erode_zero_skips_erosion(self):
        """edge_erode_px=0 skips the erosion step without error."""
        fg = np.zeros((16, 16, 3), dtype=np.float32)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        source = np.full((16, 16, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert out.shape == (16, 16, 3)

    def test_blur_zero_skips_blur(self):
        """edge_blur_px=0 skips the Gaussian blur step without error."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=0)
        assert out.dtype == np.float32

    def test_output_values_in_range(self):
        """Output values are in [0, 1]."""
        fg = np.zeros((32, 32, 3), dtype=np.float32)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        source = np.full((32, 32, 3), 0.5, dtype=np.float32)
        out = apply_source_passthrough(fg, alpha, source, edge_erode_px=0, edge_blur_px=1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5
