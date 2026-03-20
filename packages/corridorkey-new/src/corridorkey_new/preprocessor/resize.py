"""Preprocessing stage — resize (step 5).

Fits image and alpha into img_size × img_size before inference.
Both arrays are always resized with the same strategy so they stay
spatially aligned.

Swap or extend this file to add new resize strategies (e.g. letterbox,
tiling) without touching any other part of the preprocessor.
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ResizeStrategy = Literal["squish", "letterbox"]


def resize_frame(
    image: np.ndarray,
    alpha: np.ndarray,
    img_size: int,
    strategy: ResizeStrategy,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and alpha to img_size × img_size.

    Args:
        image: float32 [H, W, 3] sRGB, range 0.0–1.0.
        alpha: float32 [H, W, 1] linear, range 0.0–1.0.
        img_size: Target square resolution (e.g. 2048).
        strategy: "squish" or "letterbox". Letterbox is not yet implemented
            and falls back to squish with a warning.

    Returns:
        Tuple of (image [img_size, img_size, 3], alpha [img_size, img_size, 1]),
        both float32.
    """
    if strategy == "letterbox":
        logger.warning("resize_strategy='letterbox' is not yet implemented — falling back to 'squish'.")

    return _squish(image, alpha, img_size)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _squish(
    image: np.ndarray,
    alpha: np.ndarray,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stretch both arrays to img_size × img_size regardless of aspect ratio."""
    image_out = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # cv2.resize drops the channel dim on single-channel arrays — restore it.
    alpha_out = cv2.resize(alpha[:, :, 0], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return image_out, alpha_out[:, :, np.newaxis]
