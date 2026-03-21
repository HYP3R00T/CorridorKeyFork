"""Postprocessor stage — alpha matte cleanup (despeckle).

Removes small disconnected foreground regions from a predicted alpha matte
using connected-component analysis.
"""

from __future__ import annotations

import cv2
import numpy as np


def despeckle_alpha(alpha: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small disconnected regions from a predicted alpha matte.

    Args:
        alpha: [H, W, 1] float32 array, range 0-1.
        min_area: Minimum connected component area in pixels to keep.
            Regions smaller than this are zeroed out.

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

    # Dilate to recover edges lost during binarisation (dilation=25 → kernel 51×51)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    keep_mask = cv2.dilate(keep_mask, kernel)

    # Blur to soften the hard mask edge (blur_size=5 → kernel 11×11)
    keep_mask = cv2.GaussianBlur(keep_mask, (11, 11), 0)

    keep_f = keep_mask.astype(np.float32) / 255.0
    result = (a2d * keep_f)[:, :, np.newaxis].astype(np.float32)
    return result
