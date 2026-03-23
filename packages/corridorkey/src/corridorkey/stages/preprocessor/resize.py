"""Preprocessing stage — resize.

Resizes image and alpha to img_size × img_size for model inference.
The frame is squished to a square — no aspect ratio preservation, no padding.
This matches the reference pipeline and gives the model 100% of its spatial
capacity on the actual content.

Interpolation modes
-------------------
Downscaling  -- "area" (anti-aliased box filter, no ringing, best quality).
                Always used for downscaling regardless of upsample_mode.
                For extreme ratios (>4×), halved iteratively before the final
                area pass to avoid aliasing artefacts on 8K+ sources.
Upscaling    -- "bicubic" with antialias=True (default, sharpest quality)
             -- "bilinear" with antialias=True (faster, slightly softer)

Colour-aware downscaling
------------------------
When downscaling sRGB images, the pipeline linearises the image before the
area downsample and re-encodes to sRGB afterwards. This produces a more
accurate average luminance at edges compared to downsampling in gamma space.

Post-upscale sharpening
-----------------------
After bicubic upscaling, a mild unsharp mask is applied to recover softness
introduced by the antialias filter. Strength is configurable (0.0 = off).

Alpha clamp
-----------
Alpha is clamped to [0, 1] after every resize operation to eliminate
out-of-range values from floating-point rounding.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as functional

from corridorkey.stages.preprocessor.colorspace import linear_to_srgb

logger = logging.getLogger(__name__)

UpsampleMode = Literal["bicubic", "bilinear"]  # kept for any external references
ImageUpsampleMode = Literal["bicubic", "bilinear"]

DEFAULT_IMAGE_UPSAMPLE_MODE: ImageUpsampleMode = "bicubic"
# Alpha upscale is fixed to bilinear in PyTorch — bicubic rings below zero on
# hard matte edges, producing negative alpha values that corrupt compositing.
_ALPHA_UPSAMPLE_MODE = "bilinear"

# Ratio threshold above which multi-step downscaling is used.
_MULTISTEP_THRESHOLD = 4.0

# sRGB linearisation threshold (IEC 61966-2-1)
_SRGB_LINEARISE_THRESHOLD = 0.04045
_SRGB_LINEARISE_SCALE = 12.92
_SRGB_LINEARISE_OFFSET = 0.055
_SRGB_LINEARISE_GAMMA = 2.4


def resize_frame(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
    image_upsample_mode: ImageUpsampleMode = DEFAULT_IMAGE_UPSAMPLE_MODE,
    sharpen_strength: float = 0.3,
    is_srgb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha to img_size × img_size.

    Simple square resize — no aspect ratio preservation, no padding.
    Identical to the reference pipeline's cv2.resize(image, (img_size, img_size)).

    Downscaling uses colour-aware area interpolation (sRGB → linear → area → sRGB).
    Upscaling uses image_upsample_mode for the image and bilinear for alpha.

    Args:
        image: float32 [1, 3, H, W], sRGB, range 0.0–1.0.
        alpha: float32 [1, 1, H, W], linear, range 0.0–1.0.
        img_size: Target square resolution (e.g. 2048).
        image_upsample_mode: Interpolation mode for upscaling the image.
            "bicubic" (default) is sharpest. "bilinear" is faster.
        sharpen_strength: Unsharp mask strength applied after upscaling.
            0.3 (default) recovers antialias softness. 0.0 disables.
        is_srgb: If True, linearise before downscaling and re-encode to sRGB
            afterwards (colour-accurate downscaling). Set False for linear inputs.

    Returns:
        Tuple of:
            image  [1, 3, img_size, img_size] float32
            alpha  [1, 1, img_size, img_size] float32, clamped to [0, 1]
    """
    src_h, src_w = image.shape[2], image.shape[3]

    if src_h == img_size and src_w == img_size:
        return image, alpha.clamp(0.0, 1.0)

    img_out, alp_out = _resize_content(
        image,
        alpha,
        img_size,
        img_size,
        src_h,
        src_w,
        image_upsample_mode=image_upsample_mode,
        sharpen_strength=sharpen_strength,
        is_srgb=is_srgb,
    )
    return img_out, alp_out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_content(
    image: torch.Tensor,
    alpha: torch.Tensor,
    target_h: int,
    target_w: int,
    src_h: int,
    src_w: int,
    image_upsample_mode: ImageUpsampleMode,
    sharpen_strength: float,
    is_srgb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha to (target_h, target_w) with quality improvements."""
    h_down = src_h > target_h
    w_down = src_w > target_w

    if h_down == w_down:
        img_out, alp_out = _single_pass(
            image,
            alpha,
            target_h,
            target_w,
            downscaling=h_down,
            img_mode=image_upsample_mode,
            is_srgb=is_srgb,
        )
    else:
        # Mixed: one dimension down, one up — two passes
        mid_h = target_h if h_down else src_h
        mid_w = target_w if w_down else src_w
        img_mid, alp_mid = _single_pass(
            image,
            alpha,
            mid_h,
            mid_w,
            downscaling=True,
            img_mode=image_upsample_mode,
            is_srgb=is_srgb,
        )
        img_out, alp_out = _single_pass(
            img_mid,
            alp_mid,
            target_h,
            target_w,
            downscaling=False,
            img_mode=image_upsample_mode,
            is_srgb=False,
        )

    net_upscale = (target_h * target_w) > (src_h * src_w)
    if net_upscale and sharpen_strength > 0.0:
        img_out = _unsharp_mask(img_out, strength=sharpen_strength)

    return img_out, alp_out


def _single_pass(
    image: torch.Tensor,
    alpha: torch.Tensor,
    target_h: int,
    target_w: int,
    downscaling: bool,
    img_mode: ImageUpsampleMode,
    is_srgb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = (target_h, target_w)
    src_h, src_w = image.shape[2], image.shape[3]

    if downscaling:
        if is_srgb:
            image = _srgb_to_linear(image)
        ratio = max(src_h / target_h, src_w / target_w)
        if ratio > _MULTISTEP_THRESHOLD:
            image, alpha = _multistep_downscale(image, alpha, target_h, target_w)
        else:
            combined = torch.cat([image, alpha], dim=1)
            out = functional.interpolate(combined, size=size, mode="area")
            image, alpha = out[:, :3], out[:, 3:]
        if is_srgb:
            image = _linear_to_srgb(image)
        return image, alpha

    if img_mode == _ALPHA_UPSAMPLE_MODE:
        combined = torch.cat([image, alpha], dim=1)
        out = functional.interpolate(combined, size=size, mode=img_mode, align_corners=False, antialias=True)
        return out[:, :3], out[:, 3:]

    img_out = functional.interpolate(image, size=size, mode=img_mode, align_corners=False, antialias=True)
    alp_out = functional.interpolate(alpha, size=size, mode=_ALPHA_UPSAMPLE_MODE, align_corners=False, antialias=True)
    return img_out, alp_out


def _multistep_downscale(
    image: torch.Tensor,
    alpha: torch.Tensor,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_h, cur_w = image.shape[2], image.shape[3]
    while cur_h / target_h > 2.0 or cur_w / target_w > 2.0:
        next_h = max(cur_h // 2, target_h)
        next_w = max(cur_w // 2, target_w)
        combined = torch.cat([image, alpha], dim=1)
        out = functional.interpolate(combined, size=(next_h, next_w), mode="area")
        image, alpha = out[:, :3], out[:, 3:]
        cur_h, cur_w = next_h, next_w
    if cur_h != target_h or cur_w != target_w:
        combined = torch.cat([image, alpha], dim=1)
        out = functional.interpolate(combined, size=(target_h, target_w), mode="area")
        image, alpha = out[:, :3], out[:, 3:]
    return image, alpha


def _unsharp_mask(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=image.dtype, device=image.device)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    c = image.shape[1]
    kernel = kernel_2d.expand(c, 1, 5, 5).contiguous()
    blurred = functional.conv2d(image, kernel, padding=2, groups=c)
    return (image + strength * (image - blurred)).clamp(0.0, 1.0)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    linear = x / _SRGB_LINEARISE_SCALE
    gamma = ((x + _SRGB_LINEARISE_OFFSET) / (1.0 + _SRGB_LINEARISE_OFFSET)).clamp(min=1e-12).pow(_SRGB_LINEARISE_GAMMA)
    return torch.where(x <= _SRGB_LINEARISE_THRESHOLD, linear, gamma)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return linear_to_srgb(x)
