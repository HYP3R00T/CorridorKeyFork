"""Preprocessing stage — resize.

Fits image and alpha into img_size × img_size before inference using
letterboxing: the frame is scaled to fit within the square while preserving
aspect ratio, and the remaining border is padded with the mean pixel value
of the frame. This gives the model a neutral border that matches the image
statistics rather than an arbitrary grey that may be far from the image
distribution.

The pad offsets are returned in a ``LetterboxPad`` so the postprocessor can
crop the model output back to the original aspect ratio before scaling to
source resolution.

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

Per-dimension mode selection
-----------------------------
The downscale/upscale decision is made per-dimension. If one dimension is
being downscaled and the other upscaled (e.g. portrait source into a square
target), two passes are used so each dimension gets the correct mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional

from corridorkey_new.stages.preprocessor.colorspace import linear_to_srgb

logger = logging.getLogger(__name__)

UpsampleMode = Literal["bicubic", "bilinear"]

DEFAULT_UPSAMPLE_MODE: UpsampleMode = "bicubic"
DEFAULT_ALPHA_UPSAMPLE_MODE: UpsampleMode = "bilinear"

# Ratio threshold above which multi-step downscaling is used.
_MULTISTEP_THRESHOLD = 4.0

# sRGB linearisation threshold (IEC 61966-2-1)
_SRGB_LINEARISE_THRESHOLD = 0.04045
_SRGB_LINEARISE_SCALE = 12.92
_SRGB_LINEARISE_OFFSET = 0.055
_SRGB_LINEARISE_GAMMA = 2.4


@dataclass(frozen=True)
class LetterboxPad:
    """Padding offsets added during letterboxing.

    All values are in pixels at model resolution (img_size × img_size).
    The postprocessor uses these to crop the model output back to the
    original aspect ratio before scaling to source resolution.

    Attributes:
        top: Rows of padding at the top.
        bottom: Rows of padding at the bottom.
        left: Columns of padding at the left.
        right: Columns of padding at the right.
        inner_h: Height of the content region (excluding padding).
        inner_w: Width of the content region (excluding padding).
    """

    top: int
    bottom: int
    left: int
    right: int
    inner_h: int
    inner_w: int

    @property
    def is_noop(self) -> bool:
        """True when no padding was added (source was already square)."""
        return self.top == 0 and self.bottom == 0 and self.left == 0 and self.right == 0


def letterbox_frame(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
    upsample_mode: UpsampleMode = DEFAULT_UPSAMPLE_MODE,
    alpha_upsample_mode: UpsampleMode = DEFAULT_ALPHA_UPSAMPLE_MODE,
    sharpen_strength: float = 0.0,
    is_srgb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, LetterboxPad]:
    """Fit image and alpha into img_size × img_size preserving aspect ratio.

    The frame is scaled so its longest dimension equals img_size, then padded
    symmetrically with the mean pixel value of the frame. The pad offsets are
    returned so the postprocessor can crop back.

    Args:
        image: float32 [1, 3, H, W], sRGB, range 0.0–1.0.
        alpha: float32 [1, 1, H, W], linear, range 0.0–1.0.
        img_size: Target square resolution (e.g. 2048).
        upsample_mode: Interpolation mode for upscaling the image.
        alpha_upsample_mode: Interpolation mode for upscaling the alpha matte.
        sharpen_strength: Unsharp mask strength applied after upscaling.
            0.0 disables sharpening. Typical range 0.1–0.5.
        is_srgb: If True, linearise before downscaling and re-encode to sRGB
            afterwards (colour-accurate downscaling). Set False for linear
            inputs (they are already linear).

    Returns:
        Tuple of:
            image  [1, 3, img_size, img_size] float32
            alpha  [1, 1, img_size, img_size] float32, clamped to [0, 1]
            pad    LetterboxPad with crop offsets for the postprocessor
    """
    src_h, src_w = image.shape[2], image.shape[3]

    # --- Compute inner (content) dimensions preserving aspect ratio ----------
    scale = img_size / max(src_h, src_w)
    inner_h = round(src_h * scale)
    inner_w = round(src_w * scale)
    # Clamp to img_size in case of rounding overshoot
    inner_h = min(inner_h, img_size)
    inner_w = min(inner_w, img_size)

    # --- Compute symmetric padding offsets ------------------------------------
    pad_total_h = img_size - inner_h
    pad_total_w = img_size - inner_w
    pad_top = pad_total_h // 2
    pad_bottom = pad_total_h - pad_top
    pad_left = pad_total_w // 2
    pad_right = pad_total_w - pad_left

    pad = LetterboxPad(
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        inner_h=inner_h,
        inner_w=inner_w,
    )

    # --- Resize content to inner dimensions -----------------------------------
    if src_h == inner_h and src_w == inner_w:
        img_inner = image
        alp_inner = alpha
    else:
        img_inner, alp_inner = _resize_content(
            image,
            alpha,
            inner_h,
            inner_w,
            src_h,
            src_w,
            upsample_mode=upsample_mode,
            alpha_upsample_mode=alpha_upsample_mode,
            sharpen_strength=sharpen_strength,
            is_srgb=is_srgb,
        )

    # Clamp alpha after resize
    alp_inner = alp_inner.clamp(0.0, 1.0)

    # --- Early exit if no padding needed (already square) --------------------
    if pad.is_noop:
        return img_inner, alp_inner, pad

    # --- Pad with mean pixel value -------------------------------------------
    # Mean computed over the content region (spatial dims), per channel.
    # Shape: [1, 3, 1, 1] — broadcast-ready.
    mean_val = img_inner.mean(dim=(2, 3), keepdim=True)

    img_out = functional.pad(img_inner, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    # Fill padded regions with mean value per channel
    if pad_top > 0:
        img_out[:, :, :pad_top, :] = mean_val
    if pad_bottom > 0:
        img_out[:, :, img_size - pad_bottom :, :] = mean_val
    if pad_left > 0:
        img_out[:, :, pad_top : img_size - pad_bottom, :pad_left] = mean_val
    if pad_right > 0:
        img_out[:, :, pad_top : img_size - pad_bottom, img_size - pad_right :] = mean_val

    # Alpha padding is always 0.0 (fully transparent border)
    alp_out = functional.pad(alp_inner, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    return img_out, alp_out, pad


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
    upsample_mode: UpsampleMode,
    alpha_upsample_mode: UpsampleMode,
    sharpen_strength: float,
    is_srgb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha to (target_h, target_w) with quality improvements."""
    h_down = src_h > target_h
    w_down = src_w > target_w

    if h_down == w_down:
        # Both dimensions go the same direction — single pass
        img_out, alp_out = _single_pass(
            image,
            alpha,
            target_h,
            target_w,
            downscaling=h_down,
            img_mode=upsample_mode,
            alpha_mode=alpha_upsample_mode,
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
            img_mode=upsample_mode,
            alpha_mode=alpha_upsample_mode,
            is_srgb=is_srgb,
        )
        img_out, alp_out = _single_pass(
            img_mid,
            alp_mid,
            target_h,
            target_w,
            downscaling=False,
            img_mode=upsample_mode,
            alpha_mode=alpha_upsample_mode,
            is_srgb=False,  # already linearised/re-encoded in first pass if needed
        )

    # Post-upscale sharpening (image only, upscale path)
    upscaling = not (h_down and w_down)
    if upscaling and sharpen_strength > 0.0:
        img_out = _unsharp_mask(img_out, strength=sharpen_strength)

    return img_out, alp_out


def _single_pass(
    image: torch.Tensor,
    alpha: torch.Tensor,
    target_h: int,
    target_w: int,
    downscaling: bool,
    img_mode: UpsampleMode,
    alpha_mode: UpsampleMode,
    is_srgb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha in one or two interpolate calls."""
    size = (target_h, target_w)
    src_h, src_w = image.shape[2], image.shape[3]

    if downscaling:
        # Colour-aware downscaling: linearise → area → re-encode
        if is_srgb:
            image = _srgb_to_linear(image)

        # Multi-step: halve iteratively for extreme ratios
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

    # Upscaling
    if img_mode == alpha_mode:
        combined = torch.cat([image, alpha], dim=1)
        out = functional.interpolate(combined, size=size, mode=img_mode, align_corners=False, antialias=True)
        return out[:, :3], out[:, 3:]

    img_out = functional.interpolate(image, size=size, mode=img_mode, align_corners=False, antialias=True)
    alp_out = functional.interpolate(alpha, size=size, mode=alpha_mode, align_corners=False, antialias=True)
    return img_out, alp_out


def _multistep_downscale(
    image: torch.Tensor,
    alpha: torch.Tensor,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Halve iteratively until within 2× of target, then do a final area pass.

    Avoids aliasing artefacts when downscaling by more than 4× in one shot.
    Image is expected to be in linear light when this is called.
    """
    cur_h, cur_w = image.shape[2], image.shape[3]

    while cur_h / target_h > 2.0 or cur_w / target_w > 2.0:
        next_h = max(cur_h // 2, target_h)
        next_w = max(cur_w // 2, target_w)
        combined = torch.cat([image, alpha], dim=1)
        out = functional.interpolate(combined, size=(next_h, next_w), mode="area")
        image, alpha = out[:, :3], out[:, 3:]
        cur_h, cur_w = next_h, next_w

    # Final pass to exact target
    if cur_h != target_h or cur_w != target_w:
        combined = torch.cat([image, alpha], dim=1)
        out = functional.interpolate(combined, size=(target_h, target_w), mode="area")
        image, alpha = out[:, :3], out[:, 3:]

    return image, alpha


def _unsharp_mask(image: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply a mild unsharp mask to recover softness from antialias filtering.

    Uses a 5×5 Gaussian blur as the low-frequency reference.
    output = image + strength * (image - blur(image))

    Args:
        image: float32 [1, 3, H, W].
        strength: Blend weight for the detail layer. Typical range 0.1–0.5.

    Returns:
        Sharpened float32 [1, 3, H, W], clamped to [0, 1].
    """
    # 5×5 Gaussian kernel, sigma≈1.0
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=image.dtype, device=image.device)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  # [5, 5]
    # Expand to [C, 1, 5, 5] for depthwise conv; contiguous() required for MPS.
    c = image.shape[1]
    kernel = kernel_2d.expand(c, 1, 5, 5).contiguous()
    blurred = functional.conv2d(image, kernel, padding=2, groups=c)
    sharpened = image + strength * (image - blurred)
    return sharpened.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# sRGB ↔ linear transfer functions (on-device)
# Linearisation (_srgb_to_linear) is only needed here — no shared version
# exists in colorspace.py. Re-encoding (_linear_to_srgb) delegates to the
# shared colorspace implementation to avoid duplicating the constants.
# ---------------------------------------------------------------------------


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB → linear light (IEC 61966-2-1), on-device."""
    x = x.clamp(0.0, 1.0)
    linear = x / _SRGB_LINEARISE_SCALE
    gamma = ((x + _SRGB_LINEARISE_OFFSET) / (1.0 + _SRGB_LINEARISE_OFFSET)).clamp(min=1e-12).pow(_SRGB_LINEARISE_GAMMA)
    return torch.where(x <= _SRGB_LINEARISE_THRESHOLD, linear, gamma)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear light → sRGB (IEC 61966-2-1), on-device.

    Delegates to the shared colorspace implementation.
    """
    return linear_to_srgb(x)
