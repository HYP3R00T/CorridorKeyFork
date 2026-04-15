from __future__ import annotations

import functools

import cv2
import numpy as np

from corridorkey.infra.colorspace import linear_to_srgb_lut, lut_apply, srgb_to_linear_lut


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert a float32 sRGB array to linear light via the IEC 61966-2-1 LUT."""
    return lut_apply(np.clip(x, 0.0, 1.0), srgb_to_linear_lut)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert a float32 linear-light array to sRGB via the IEC 61966-2-1 LUT."""
    return lut_apply(np.clip(x, 0.0, 1.0), linear_to_srgb_lut)


def make_processed(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build a premultiplied linear RGBA array — the primary compositor output.

    Converts fg from sRGB to linear light before premultiplying, so transparent
    regions are correctly zeroed out (fg_linear * alpha = 0 where alpha = 0).
    No black-blob artefacts appear when the file is opened in a compositor.

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
        alpha: [H, W, 1] or [H, W] float32 alpha, range 0-1.
        source: [H, W, 3] float32 sRGB original source image, range 0-1.
        edge_erode_px: Erosion kernel radius in pixels.
        edge_blur_px: Gaussian blur sigma for the blend seam.

    Returns:
        [H, W, 3] float32 sRGB FG with interior replaced by source pixels.
    """
    alpha_2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha

    # Build interior mask: only pixels that are confidently fully opaque.
    interior = (alpha_2d > 0.95).astype(np.float32)

    if edge_erode_px > 0:
        k = edge_erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        interior = cv2.erode(interior, kernel)

    if edge_blur_px > 0:
        blur_k = edge_blur_px * 2 + 1
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

    bg_srgb = _checkerboard(w, h, checker_size)
    bg_linear = srgb_to_linear(bg_srgb)
    fg_linear = srgb_to_linear(fg)

    comp_linear = fg_linear * alpha + bg_linear * (1.0 - alpha)

    return linear_to_srgb(comp_linear).astype(np.float32)


@functools.lru_cache(maxsize=8)
def _checkerboard(width: int, height: int, checker_size: int) -> np.ndarray:
    x_tiles = np.arange(width) // checker_size
    y_tiles = np.arange(height) // checker_size
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)
    pattern = np.where((x_grid + y_grid) % 2 == 0, 0.15, 0.55).astype(np.float32)
    return np.stack([pattern, pattern, pattern], axis=-1)
