"""Postprocessor stage — alpha matte cleanup (despeckle).

Removes small disconnected foreground regions from a predicted alpha matte
using connected-component analysis.

The safe_zone mask is used only to zero out removed components. Kept regions
retain their original alpha values unchanged — the dilation+blur only softens
the boundary of the removal mask, not the kept alpha itself.
"""

from __future__ import annotations

import cv2
import numpy as np


def despeckle_alpha(
    alpha: np.ndarray,
    min_area: int,
    dilation: int = 25,
    blur_size: int = 5,
) -> np.ndarray:
    """Remove small disconnected regions from a predicted alpha matte.

    Builds a binary keep-mask from connected components, dilates it to recover
    semi-transparent edge pixels excluded by the binary threshold, then blurs
    the mask edge. The original alpha is multiplied by this mask so that:
      - Removed components → zeroed out
      - Kept regions → original alpha values preserved (no softening)
      - Transition band → smoothly faded out

    Args:
        alpha: [H, W, 1] float32 array, range 0-1.
        min_area: Minimum connected component area in pixels to keep.
            Regions smaller than this are zeroed out.
        dilation: Dilation radius in pixels applied after component removal
            to recover semi-transparent edge pixels lost by the 0.5 threshold.
            Default 25.
        blur_size: Gaussian blur radius applied after dilation to soften the
            hard mask edge at removed-component boundaries. Default 5.

    Returns:
        [H, W, 1] float32 array with small islands removed.
    """
    if min_area <= 0:
        return alpha

    a2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    binary = (a2d > 0.5).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keep_mask = np.zeros_like(binary)
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep_mask[labels == i] = 255

    # Dilate to recover semi-transparent edge pixels excluded by the 0.5 threshold
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        keep_mask = cv2.dilate(keep_mask, kernel)

    # Blur to soften the hard removal boundary
    if blur_size > 0:
        b_size = int(blur_size * 2 + 1)
        keep_mask = cv2.GaussianBlur(keep_mask, (b_size, b_size), 0)

    keep_f = keep_mask.astype(np.float32) / 255.0

    # Clamp keep_f to [0, 1] and multiply — this zeroes removed regions while
    # leaving kept regions at their original alpha values (keep_f ≈ 1.0 there).
    keep_f = np.clip(keep_f, 0.0, 1.0)
    result = (a2d * keep_f)[:, :, np.newaxis].astype(np.float32)
    return result
