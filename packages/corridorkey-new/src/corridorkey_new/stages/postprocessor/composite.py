"""Postprocessor stage — compositing helpers.

make_processed          — premultiplied linear RGBA (primary compositor output)
apply_source_passthrough — replace model FG in opaque interior with source pixels
make_preview            — checkerboard composite for visual QC

All compositing is done in linear light (physically correct). sRGB inputs are
converted to linear before blending and back to sRGB for display/writing.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

# sRGB transfer function constants (IEC 61966-2-1)
_LINEAR_THRESHOLD = 0.0031308
_ENCODED_THRESHOLD = 0.04045
_LINEAR_SCALE = 12.92
_GAMMA = 1.0 / 2.4
_SCALE = 1.055
_OFFSET = 0.055

# LUT-accelerated sRGB transfer functions (65536 entries, <0.002% error).
_LUT_SIZE = 65536
_lut_x = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float64)

_linear_to_srgb_lut: np.ndarray = np.where(
    _lut_x <= _LINEAR_THRESHOLD,
    _lut_x * _LINEAR_SCALE,
    _SCALE * np.power(np.maximum(_lut_x, 1e-12), _GAMMA) - _OFFSET,
).astype(np.float32)

_srgb_to_linear_lut: np.ndarray = np.where(
    _lut_x <= _ENCODED_THRESHOLD,
    _lut_x / _LINEAR_SCALE,
    np.power(np.maximum((_lut_x + _OFFSET) / _SCALE, 1e-12), 2.4),
).astype(np.float32)


def _lut(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    indices = np.clip(np.rint(x * (_LUT_SIZE - 1)).astype(np.uint16), 0, _LUT_SIZE - 1)
    return lut[indices]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return _lut(np.clip(x, 0.0, 1.0), _srgb_to_linear_lut)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return _lut(np.clip(x, 0.0, None), _linear_to_srgb_lut)


def make_processed(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build a premultiplied linear RGBA array — the primary compositor output.

    Transparent regions are correctly zeroed out (fg * alpha = 0 where alpha = 0),
    so no black-blob artefacts appear when the file is opened in a compositor.

    Args:
        fg: [H, W, 3] float32 sRGB straight, range 0-1.
        alpha: [H, W, 1] float32 linear, range 0-1.

    Returns:
        [H, W, 4] float32 premultiplied linear RGBA, range 0-1.
    """
    fg_linear = srgb_to_linear(fg)
    fg_premul = fg_linear * alpha  # premultiply in linear light
    return np.concatenate([fg_premul, alpha], axis=-1).astype(np.float32)


def apply_source_passthrough(
    fg: np.ndarray,
    alpha: np.ndarray,
    source: np.ndarray,
    edge_erode_px: int = 3,
    edge_blur_px: int = 7,
) -> np.ndarray:
    """Replace model FG in opaque interior regions with original source pixels.

    The model's FG prediction in semi-transparent edge regions is contaminated
    by the background colour (green spill, dark fringing). In fully opaque
    interior regions the source pixel is a better FG estimate than the model.

    Strategy:
      1. Build an "interior mask" = alpha eroded by edge_erode_px.
      2. Blur the mask edge to create a soft blend seam.
      3. Blend: output = source * interior + model_fg * (1 - interior).

    Args:
        fg: [H, W, 3] float32 sRGB model FG, range 0-1.
        alpha: [H, W, 1] float32 alpha, range 0-1.
        source: [H, W, 3] float32 sRGB original source image, range 0-1.
        edge_erode_px: Erosion kernel radius in pixels.
        edge_blur_px: Gaussian blur sigma for the blend seam.

    Returns:
        [H, W, 3] float32 sRGB FG with interior replaced by source pixels.
    """
    alpha_2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha

    # Build interior mask via erosion of the alpha matte.
    if edge_erode_px > 0:
        k = edge_erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        interior = cv2.erode(alpha_2d, kernel, iterations=1)
    else:
        interior = alpha_2d.copy()

    # Soft blend seam.
    if edge_blur_px > 0:
        blur_k = max(1, edge_blur_px * 2 + 1)
        # Ensure odd kernel size
        if blur_k % 2 == 0:
            blur_k += 1
        interior = cv2.GaussianBlur(interior, (blur_k, blur_k), 0)

    interior = np.clip(interior, 0.0, 1.0)[:, :, np.newaxis]  # [H, W, 1]

    return (source * interior + fg * (1.0 - interior)).astype(np.float32)


def make_preview(fg: np.ndarray, alpha: np.ndarray, checker_size: int) -> np.ndarray:
    """Composite fg over a checkerboard background in linear light.

    Args:
        fg: [H, W, 3] float32 sRGB array, range 0-1.
        alpha: [H, W, 1] float32 array, range 0-1.
        checker_size: Tile size in pixels for the checkerboard.

    Returns:
        [H, W, 3] float32 sRGB composite, range 0-1.
    """
    h, w = fg.shape[:2]

    bg_srgb = _get_checkerboard(w, h, checker_size)
    bg_linear = srgb_to_linear(bg_srgb)
    fg_linear = srgb_to_linear(fg)

    comp_linear = fg_linear * alpha + bg_linear * (1.0 - alpha)

    return linear_to_srgb(comp_linear).astype(np.float32)


@functools.lru_cache(maxsize=8)
def _get_checkerboard(width: int, height: int, checker_size: int) -> np.ndarray:
    """Return a cached [H, W, 3] float32 sRGB checkerboard array."""
    return _make_checkerboard(width, height, checker_size)


def _make_checkerboard(
    width: int,
    height: int,
    checker_size: int,
    color1: float = 0.15,
    color2: float = 0.55,
) -> np.ndarray:
    """Return a [H, W, 3] float32 sRGB checkerboard array."""
    x_tiles = np.arange(width) // checker_size
    y_tiles = np.arange(height) // checker_size
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)
    pattern = np.where((x_grid + y_grid) % 2 == 0, color1, color2).astype(np.float32)
    return np.stack([pattern, pattern, pattern], axis=-1)
