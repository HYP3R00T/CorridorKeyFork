"""Preprocessing stage — resize.

Resizes image and alpha to img_size × img_size for model inference.
The frame is squished to a square — no aspect ratio preservation, no padding.

Interpolation
-------------
Both image and alpha are resized with INTER_LINEAR (bilinear) in sRGB space,
exactly matching the reference pipeline:

    cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    cv2.resize(mask,  (img_size, img_size), interpolation=cv2.INTER_LINEAR)

The model was trained on INTER_LINEAR-resized sRGB inputs. Any deviation from
this (area filter, color-space conversion, sharpening) changes the input
distribution and degrades model output quality.

Alpha clamp
-----------
Alpha is clamped to [0, 1] after resize to eliminate floating-point rounding
artefacts.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as functional

logger = logging.getLogger(__name__)

UpsampleMode = Literal["bicubic", "bilinear"]  # kept for any external references
ImageUpsampleMode = Literal["bicubic", "bilinear"]

DEFAULT_IMAGE_UPSAMPLE_MODE: ImageUpsampleMode = "bicubic"


def resize_frame(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
    image_upsample_mode: ImageUpsampleMode = DEFAULT_IMAGE_UPSAMPLE_MODE,
    sharpen_strength: float = 0.3,
    is_srgb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha to img_size × img_size using bilinear interpolation.

    Matches the reference pipeline exactly:
        cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        cv2.resize(mask,  (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    The image_upsample_mode, sharpen_strength, and is_srgb parameters are
    accepted for API compatibility but ignored — the model requires plain
    bilinear resize in sRGB space regardless of direction.

    Args:
        image: float32 [1, 3, H, W], sRGB, range 0.0–1.0.
        alpha: float32 [1, 1, H, W], linear, range 0.0–1.0.
        img_size: Target square resolution (e.g. 2048).
        image_upsample_mode: Ignored. Kept for API compatibility.
        sharpen_strength: Ignored. Kept for API compatibility.
        is_srgb: Ignored. Kept for API compatibility.

    Returns:
        Tuple of:
            image  [1, 3, img_size, img_size] float32
            alpha  [1, 1, img_size, img_size] float32, clamped to [0, 1]
    """
    src_h, src_w = image.shape[2], image.shape[3]

    if src_h == img_size and src_w == img_size:
        return image, alpha.clamp(0.0, 1.0)

    size = (img_size, img_size)
    img_out = functional.interpolate(image, size=size, mode="bilinear", align_corners=False)
    alp_out = functional.interpolate(alpha, size=size, mode="bilinear", align_corners=False)

    return img_out, alp_out.clamp(0.0, 1.0)
