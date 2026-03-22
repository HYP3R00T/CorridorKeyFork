"""Preprocessing stage — resize.

Fits image and alpha into img_size x img_size before inference.
Both tensors are concatenated along the channel dimension and resized in a
single interpolate call — one kernel launch instead of two.

Uses torch.nn.functional.interpolate so the operation runs on whatever
device the tensors live on (CUDA, MPS, or CPU).

Interpolation modes
-------------------
Downscaling  -- "area" (anti-aliased box filter, no ringing, best quality).
                Always used for downscaling regardless of upsample_mode.
Upscaling    -- "bicubic" with antialias=True (default, sharpest quality)
             -- "bilinear" with antialias=True (faster, slightly softer)

Per-dimension mode selection
-----------------------------
The downscale/upscale decision is made per-dimension. If one dimension is
being downscaled and the other upscaled (e.g. portrait source into a square
target), two passes are used so each dimension gets the correct mode.
This avoids blurry results from applying area mode to an upscaled dimension.

Alpha resize mode
-----------------
Alpha mattes are often better served by bilinear interpolation than bicubic,
since bicubic can ring and produce negative alpha values at sharp matte edges.
alpha_upsample_mode is independently configurable from upsample_mode.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

UpsampleMode = Literal["bicubic", "bilinear"]

DEFAULT_UPSAMPLE_MODE: UpsampleMode = "bicubic"
DEFAULT_ALPHA_UPSAMPLE_MODE: UpsampleMode = "bilinear"


def resize_frame(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
    upsample_mode: UpsampleMode = DEFAULT_UPSAMPLE_MODE,
    alpha_upsample_mode: UpsampleMode = DEFAULT_ALPHA_UPSAMPLE_MODE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha tensors to img_size x img_size.

    Image and alpha are concatenated and resized in a single interpolate call
    when both dimensions go the same direction (both down or both up). When
    one dimension is downscaled and the other upscaled, two passes are used
    so each dimension gets the correct mode.

    Downscaling uses area interpolation (anti-aliased, no ringing).
    Upscaling uses upsample_mode / alpha_upsample_mode with antialias=True.

    Args:
        image: float32 [1, 3, H, W], sRGB, range 0.0-1.0.
        alpha: float32 [1, 1, H, W], linear, range 0.0-1.0.
        img_size: Target square resolution (e.g. 2048).
        upsample_mode: Interpolation mode for upscaling the image.
            "bicubic" (default) gives the sharpest result.
            "bilinear" is faster but softer.
            Has no effect when downscaling -- area mode is always used then.
        alpha_upsample_mode: Interpolation mode for upscaling the alpha matte.
            Defaults to "bilinear" to avoid bicubic ringing on matte edges.

    Returns:
        Tuple of (image [1, 3, img_size, img_size], alpha [1, 1, img_size, img_size]),
        both float32.
    """
    src_h, src_w = image.shape[2], image.shape[3]

    if src_h == img_size and src_w == img_size:
        return image, alpha

    h_down = src_h > img_size
    w_down = src_w > img_size

    if h_down == w_down:
        return _resize_single_pass(
            image, alpha, img_size, img_size,
            downscaling=h_down,
            img_mode=upsample_mode,
            alpha_mode=alpha_upsample_mode,
        )

    # Mixed: one dimension down, one up.
    # Pass 1: handle the downscaling dimension with area mode.
    # Pass 2: handle the upscaling dimension with the configured upsample mode.
    mid_h = img_size if h_down else src_h
    mid_w = img_size if w_down else src_w

    image_mid, alpha_mid = _resize_single_pass(
        image, alpha, mid_h, mid_w,
        downscaling=True,
        img_mode=upsample_mode,
        alpha_mode=alpha_upsample_mode,
    )
    return _resize_single_pass(
        image_mid, alpha_mid, img_size, img_size,
        downscaling=False,
        img_mode=upsample_mode,
        alpha_mode=alpha_upsample_mode,
    )


def _resize_single_pass(
    image: torch.Tensor,
    alpha: torch.Tensor,
    target_h: int,
    target_w: int,
    downscaling: bool,
    img_mode: UpsampleMode,
    alpha_mode: UpsampleMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha in a single concatenated interpolate call.

    Concatenating along the channel dimension means one kernel dispatch
    instead of two. The channels are split back out after the resize.

    For downscaling, area mode is always used (ignores img_mode/alpha_mode).
    For upscaling, img_mode and alpha_mode may differ, so we check whether
    they match and either do one combined call or two separate calls.
    """
    size = (target_h, target_w)

    if downscaling:
        combined = torch.cat([image, alpha], dim=1)
        out = F.interpolate(combined, size=size, mode="area")
        return out[:, :3], out[:, 3:]

    if img_mode == alpha_mode:
        combined = torch.cat([image, alpha], dim=1)
        out = F.interpolate(combined, size=size, mode=img_mode, align_corners=False, antialias=True)
        return out[:, :3], out[:, 3:]

    image_out = F.interpolate(image, size=size, mode=img_mode, align_corners=False, antialias=True)
    alpha_out = F.interpolate(alpha, size=size, mode=alpha_mode, align_corners=False, antialias=True)
    return image_out, alpha_out
