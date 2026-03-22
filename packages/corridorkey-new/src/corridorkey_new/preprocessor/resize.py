"""Preprocessing stage — resize (step 5).

Fits image and alpha into img_size × img_size before inference.
Both tensors are always resized with the same strategy so they stay
spatially aligned.

Uses torch.nn.functional.interpolate so the operation runs on whatever
device the tensors live on (CUDA, MPS, or CPU).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.nn import functional

logger = logging.getLogger(__name__)

ResizeStrategy = Literal["squish", "letterbox"]


def resize_frame(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
    strategy: ResizeStrategy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha tensors to img_size × img_size.

    Args:
        image: float32 [1, 3, H, W], sRGB, range 0.0–1.0.
        alpha: float32 [1, 1, H, W], linear, range 0.0–1.0.
        img_size: Target square resolution (e.g. 2048).
        strategy: "squish" or "letterbox". Letterbox is not yet implemented
            and falls back to squish with a warning.

    Returns:
        Tuple of (image [1, 3, img_size, img_size], alpha [1, 1, img_size, img_size]),
        both float32.
    """
    if strategy == "letterbox":
        logger.warning("resize_strategy='letterbox' is not yet implemented — falling back to 'squish'.")

    return _squish(image, alpha, img_size)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _squish(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stretch both tensors to img_size × img_size regardless of aspect ratio.

    Uses area interpolation when downscaling (anti-aliased, preserves detail)
    and bilinear when upscaling, matching the previous cv2.resize behaviour.
    """
    src_h, src_w = image.shape[2], image.shape[3]
    downscaling = (src_h > img_size) or (src_w > img_size)
    mode = "area" if downscaling else "bilinear"
    kwargs: dict = {} if downscaling else {"align_corners": False}

    image_out = functional.interpolate(image, size=(img_size, img_size), mode=mode, **kwargs)
    alpha_out = functional.interpolate(alpha, size=(img_size, img_size), mode=mode, **kwargs)
    return image_out, alpha_out
