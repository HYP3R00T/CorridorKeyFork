"""Postprocessor stage — checkerboard preview composite.

Composites the foreground over a checkerboard background to produce a
preview image that makes transparency visible.
"""

from __future__ import annotations

import numpy as np


def make_preview(fg: np.ndarray, alpha: np.ndarray, checker_size: int) -> np.ndarray:
    """Composite fg over a checkerboard background.

    Args:
        fg: [H, W, 3] float32 sRGB array, range 0-1.
        alpha: [H, W, 1] float32 array, range 0-1.
        checker_size: Tile size in pixels for the checkerboard.

    Returns:
        [H, W, 3] float32 sRGB composite, range 0-1.
    """
    h, w = fg.shape[:2]
    bg = _make_checkerboard(w, h, checker_size)
    comp = fg * alpha + bg * (1.0 - alpha)
    return comp.astype(np.float32)


def _make_checkerboard(
    width: int,
    height: int,
    checker_size: int,
    color1: float = 0.2,
    color2: float = 0.4,
) -> np.ndarray:
    """Return a [H, W, 3] float32 checkerboard array."""
    x_tiles = np.arange(width) // checker_size
    y_tiles = np.arange(height) // checker_size
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)
    pattern = np.where((x_grid + y_grid) % 2 == 0, color1, color2).astype(np.float32)
    return np.stack([pattern, pattern, pattern], axis=-1)
