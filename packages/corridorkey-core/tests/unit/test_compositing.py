"""Tests for corridorkey_core.compositing.

Compositing is pure math - no GPU, no model files, no filesystem. Every
function here must produce correct results for any valid float32 input, so
tests use small synthetic arrays and verify exact numerical properties
(roundtrips, identity cases, range constraints, shape preservation).

All tests run in the fast suite.
"""

import numpy as np
import pytest
import torch
from corridorkey_core.compositing import (
    clean_matte,
    composite_premul,
    composite_straight,
    create_checkerboard,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)


def _solid(h: int, w: int, r: float, g: float, b: float) -> np.ndarray:
    """Create a solid-color float32 [H, W, 3] array."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


class TestColorSpaceConversion:
    """sRGB <-> linear transfer function correctness.

    The pipeline converts between sRGB and linear light at several points
    (input normalisation, compositing, output encoding). If the transfer
    function is not its own inverse, colour will drift on every round-trip.
    """

    def test_roundtrip_numpy(self):
        """sRGB -> linear -> sRGB must recover the original value within float32 tolerance."""
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)

    def test_roundtrip_tensor(self):
        """Tensor inputs must produce a Tensor output and satisfy the same roundtrip."""
        x = torch.linspace(0.0, 1.0, 256)
        result = srgb_to_linear(linear_to_srgb(x))
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, x, atol=1e-5)

    def test_linear_to_srgb_black_and_white(self):
        """Absolute black and white must map to exactly 0.0 and ~1.0 in sRGB."""
        x = np.array([0.0, 1.0], dtype=np.float32)
        result = linear_to_srgb(x)
        assert np.isclose(result[0], 0.0, atol=1e-6)
        assert np.isclose(result[1], 1.0, atol=1e-4)

    def test_srgb_to_linear_clamps_negative(self):
        """Negative sRGB values must not produce negative linear values."""
        x = np.array([-0.5], dtype=np.float32)
        result = srgb_to_linear(x)
        assert result[0] >= 0.0

    def test_linear_to_srgb_clamps_negative(self):
        """Negative linear values must not produce negative sRGB values."""
        x = np.array([-0.5], dtype=np.float32)
        result = linear_to_srgb(x)
        assert result[0] >= 0.0

    def test_linear_to_srgb_is_brighter(self):
        """sRGB-encoded values must always be >= their linear equivalents in (0, 1).

        This is the defining property of gamma encoding - sRGB brightens
        mid-tones relative to linear light.
        """
        x = np.linspace(0.01, 0.99, 100, dtype=np.float32)
        assert np.all(linear_to_srgb(x) >= x)


class TestPremultiply:
    """Alpha premultiplication correctness.

    Premultiplied alpha is the internal representation used throughout the
    pipeline. Incorrect premultiplication causes fringing and incorrect
    compositing results downstream.
    """

    def test_full_alpha_unchanged(self):
        """With alpha=1 every pixel must pass through unmodified."""
        fg = _solid(4, 4, 0.8, 0.2, 0.5)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.allclose(result, fg)

    def test_zero_alpha_gives_black(self):
        """With alpha=0 the result must be fully transparent (all zeros)."""
        fg = _solid(4, 4, 1.0, 1.0, 1.0)
        alpha = np.zeros((4, 4, 1), dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.allclose(result, 0.0)

    def test_half_alpha_halves_values(self):
        """With alpha=0.5 all channel values must be halved."""
        fg = _solid(4, 4, 1.0, 1.0, 1.0)
        alpha = np.full((4, 4, 1), 0.5, dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.allclose(result, 0.5)


class TestCompositing:
    """Over-operator correctness for straight and premultiplied alpha.

    The compositing formulas are the foundation of the preview and export
    pipeline. Identity cases (alpha=0, alpha=1) must be exact.
    """

    def test_straight_full_alpha_shows_fg(self):
        """Full alpha must show the foreground and completely hide the background."""
        fg = _solid(4, 4, 1.0, 0.0, 0.0)
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        assert np.allclose(result, fg)

    def test_straight_zero_alpha_shows_bg(self):
        """Zero alpha must show the background and completely hide the foreground."""
        fg = _solid(4, 4, 1.0, 0.0, 0.0)
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.zeros((4, 4, 1), dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        assert np.allclose(result, bg)

    def test_premul_full_alpha_shows_fg(self):
        """Premultiplied composite with alpha=1 must equal the foreground."""
        fg = _solid(4, 4, 1.0, 0.0, 0.0)
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        result = composite_premul(fg, bg, alpha)
        assert np.allclose(result, fg)

    def test_premul_zero_alpha_shows_bg(self):
        """Premultiplied composite with alpha=0 must equal the background."""
        fg = _solid(4, 4, 0.0, 0.0, 0.0)  # premul: fg already multiplied by alpha=0
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.zeros((4, 4, 1), dtype=np.float32)
        result = composite_premul(fg, bg, alpha)
        assert np.allclose(result, bg)


class TestDespill:
    """Green-spill suppression correctness.

    Despill removes the green colour cast that reflects off a green screen
    onto the subject. It must never increase the green channel, must be a
    no-op at strength=0, and must accept both numpy and torch inputs.
    """

    def test_no_spill_unchanged(self):
        """Pure red has no green to remove - the image must be returned unchanged."""
        img = _solid(4, 4, 1.0, 0.0, 0.0)
        result = despill(img, strength=1.0)
        assert np.allclose(result, img, atol=1e-6)

    def test_green_channel_reduced(self):
        """A green-heavy image must have its G channel reduced, never increased."""
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        result = despill(img, strength=1.0)
        assert np.all(result[..., 1] <= img[..., 1] + 1e-6)

    def test_zero_strength_unchanged(self):
        """strength=0 must be a complete no-op regardless of image content."""
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        result = despill(img, strength=0.0)
        assert np.allclose(result, img)

    def test_output_shape_preserved(self):
        """Output shape must always match input shape."""
        img = _solid(8, 8, 0.3, 0.8, 0.3)
        result = despill(img)
        assert result.shape == img.shape

    def test_max_mode(self):
        """max green_limit_mode must produce a valid output shape."""
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        result = despill(img, green_limit_mode="max", strength=1.0)
        assert result.shape == img.shape

    def test_invalid_mode_raises(self):
        """An unrecognised green_limit_mode must raise ValueError immediately."""
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        with pytest.raises(ValueError, match="green_limit_mode"):
            despill(img, green_limit_mode="median")

    def test_tensor_input(self):
        """Torch tensor input must produce a Torch tensor output of the same shape."""
        img = torch.tensor(_solid(4, 4, 0.2, 0.9, 0.2))
        result = despill(img, strength=1.0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == img.shape


class TestCleanMatte:
    """Matte despeckling correctness.

    clean_matte removes small isolated alpha islands (noise) while
    preserving the main subject region. Incorrect thresholds or morphology
    would either leave noise in the matte or eat into the subject edges.
    """

    def _make_alpha_with_island(self) -> np.ndarray:
        """100x100 alpha with a large foreground blob and a small island."""
        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[10:80, 10:80] = 1.0  # large region (4900 px)
        alpha[90:93, 90:93] = 1.0  # small island (9 px)
        return alpha

    def test_removes_small_island(self):
        """Regions below area_threshold must be zeroed out."""
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100, dilation=0, blur_size=0)
        assert np.allclose(result[90:93, 90:93], 0.0)

    def test_preserves_large_region(self):
        """The main subject region (above threshold) must remain non-zero."""
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100, dilation=0, blur_size=0)
        assert result[40, 40] > 0.0

    def test_3d_input_returns_3d(self):
        """A [H, W, 1] input must produce a [H, W, 1] output."""
        alpha = self._make_alpha_with_island()[:, :, np.newaxis]
        result = clean_matte(alpha, area_threshold=100)
        assert result.ndim == 3
        assert result.shape[2] == 1

    def test_2d_input_returns_2d(self):
        """A [H, W] input must produce a [H, W] output."""
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100)
        assert result.ndim == 2

    def test_output_range(self):
        """Output values must stay in [0, 1] after despeckling."""
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestCreateCheckerboard:
    """Checkerboard background generation for transparency preview.

    The checkerboard is rendered behind transparent areas in the GUI preview.
    Shape, dtype, and colour accuracy must be exact so the preview looks correct.
    """

    def test_output_shape(self):
        """Output must be [H, W, 3] with the requested dimensions."""
        result = create_checkerboard(320, 240)
        assert result.shape == (240, 320, 3)

    def test_output_dtype(self):
        """Output must be float32 to match the rest of the compositing pipeline."""
        result = create_checkerboard(64, 64)
        assert result.dtype == np.float32

    def test_only_two_colors(self):
        """The checkerboard must contain exactly two distinct colour values."""
        result = create_checkerboard(64, 64, checker_size=32, color1=0.2, color2=0.8)
        unique = np.unique(result[..., 0].round(4))
        assert len(unique) == 2

    def test_color_values(self):
        """The two colours must match the requested color1 and color2 values."""
        result = create_checkerboard(64, 64, checker_size=32, color1=0.1, color2=0.9)
        flat = result[..., 0].flatten()
        assert np.any(np.isclose(flat, 0.1, atol=1e-4))
        assert np.any(np.isclose(flat, 0.9, atol=1e-4))

    def test_rgb_channels_equal(self):
        """All three RGB channels must be identical - the pattern is greyscale."""
        result = create_checkerboard(64, 64)
        assert np.allclose(result[..., 0], result[..., 1])
        assert np.allclose(result[..., 1], result[..., 2])
