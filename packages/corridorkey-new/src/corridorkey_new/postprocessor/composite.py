"""Postprocessor stage — checkerboard preview composite.

Composites the foreground over a checkerboard background to produce a
preview image that makes transparency visible.

Compositing is done in linear light (physically correct). The checkerboard
is defined in sRGB and converted to linear before blending. The result is
converted back to sRGB for display/writing.
"""

from __future__ import annotations

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

    # Checkerboard is defined in sRGB — convert to linear for correct blending.
    bg_srgb = _make_checkerboard(w, h, checker_size)
    bg_linear = srgb_to_linear(bg_srgb)
    fg_linear = srgb_to_linear(fg)

    # Straight alpha composite in linear light.
    comp_linear = fg_linear * alpha + bg_linear * (1.0 - alpha)

    return linear_to_srgb(comp_linear).astype(np.float32)


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
