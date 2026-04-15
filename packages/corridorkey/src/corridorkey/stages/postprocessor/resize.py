from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from corridorkey.stages.postprocessor.config import AlphaUpsampleMode, FgUpsampleMode

_FG_INTERP_MAP: dict[str, int] = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
_ALPHA_INTERP_MAP: dict[str, int] = {
    "bilinear": cv2.INTER_LINEAR,
    "lanczos4": cv2.INTER_LANCZOS4,
}

# functional.interpolate mode mapping — Lanczos is not supported, bicubic is the
# closest equivalent at inference resolutions.
_FG_TORCH_MODE: dict[str, str] = {
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "lanczos4": "bicubic",
}
_ALPHA_TORCH_MODE: dict[str, str] = {
    "bilinear": "bilinear",
    "lanczos4": "bicubic",
}


def tensor_to_numpy_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert a [1, C, H, W] tensor to a [H, W, C] float32 numpy array."""
    return t.detach().cpu().float().squeeze(0).permute(1, 2, 0).numpy()


def resize_to_source(
    alpha: torch.Tensor,
    fg: torch.Tensor,
    source_h: int,
    source_w: int,
    fg_upsample_mode: FgUpsampleMode = "lanczos4",
    alpha_upsample_mode: AlphaUpsampleMode = "lanczos4",
) -> tuple[np.ndarray, np.ndarray]:
    """Resize alpha and fg tensors back to source resolution and convert to numpy.

    CPU path — always available. Downscaling uses INTER_AREA. Upscaling uses
    the configured modes.

    Args:
        alpha: [1, 1, H, W] tensor, range 0-1.
        fg: [1, 3, H, W] tensor, range 0-1.
        source_h: Target height in pixels.
        source_w: Target width in pixels.
        fg_upsample_mode: Interpolation for upscaling FG. Default "lanczos4".
        alpha_upsample_mode: Interpolation for upscaling alpha. Default "lanczos4".

    Returns:
        Tuple of (alpha_np [H, W, 1], fg_np [H, W, 3]), both float32 numpy.
    """
    alpha_np = tensor_to_numpy_hwc(alpha.float()).clip(0.0, 1.0)  # [H_model, W_model, 1]
    fg_np = tensor_to_numpy_hwc(fg.float()).clip(0.0, 1.0)  # [H_model, W_model, 3]

    model_h, model_w = alpha_np.shape[:2]
    target = (source_w, source_h)  # cv2 uses (width, height)

    if source_h == model_h and source_w == model_w:
        return alpha_np, fg_np

    upscaling = source_h * source_w > model_h * model_w

    alpha_interp = _ALPHA_INTERP_MAP[alpha_upsample_mode] if upscaling else cv2.INTER_AREA
    alpha_2d = cv2.resize(alpha_np[:, :, 0], target, interpolation=alpha_interp)
    alpha_out = np.clip(alpha_2d, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)

    fg_interp = _FG_INTERP_MAP[fg_upsample_mode] if upscaling else cv2.INTER_AREA
    fg_out = cv2.resize(fg_np, target, interpolation=fg_interp).astype(np.float32)

    return alpha_out, fg_out


def resize_to_source_gpu(
    alpha: torch.Tensor,
    fg: torch.Tensor,
    source_h: int,
    source_w: int,
    fg_upsample_mode: FgUpsampleMode = "lanczos4",
    alpha_upsample_mode: AlphaUpsampleMode = "lanczos4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize alpha and fg tensors back to source resolution, keeping data on GPU.

    GPU path — uses ``functional.interpolate`` so tensors stay on device throughout.
    Returns BCHW float32 tensors; the caller converts to numpy after any
    subsequent GPU ops (despeckle, despill).

    Lanczos is not supported by ``functional.interpolate`` — "lanczos4" is mapped to
    "bicubic", which is visually equivalent at inference resolutions.

    Downscaling uses ``functional.interpolate`` with ``mode="area"`` (equivalent to
    INTER_AREA). Upscaling uses the configured mode.

    Args:
        alpha: [1, 1, H, W] tensor on CUDA, range 0-1.
        fg: [1, 3, H, W] tensor on CUDA, range 0-1.
        source_h: Target height in pixels.
        source_w: Target width in pixels.
        fg_upsample_mode: Interpolation for upscaling FG. Default "lanczos4".
        alpha_upsample_mode: Interpolation for upscaling alpha. Default "lanczos4".

    Returns:
        Tuple of (alpha [1, 1, H, W], fg [1, 3, H, W]), both float32 on device.
    """
    alpha = alpha.float().clamp(0.0, 1.0)
    fg = fg.float().clamp(0.0, 1.0)

    model_h, model_w = alpha.shape[2], alpha.shape[3]

    if source_h == model_h and source_w == model_w:
        return alpha, fg

    upscaling = source_h * source_w > model_h * model_w
    size = (source_h, source_w)

    if upscaling:
        alpha_mode = _ALPHA_TORCH_MODE[alpha_upsample_mode]
        fg_mode = _FG_TORCH_MODE[fg_upsample_mode]
        alpha_out = functional.interpolate(alpha, size=size, mode=alpha_mode, align_corners=False)
        fg_out = functional.interpolate(fg, size=size, mode=fg_mode, align_corners=False)
    else:
        alpha_out = functional.interpolate(alpha, size=size, mode="area")
        fg_out = functional.interpolate(fg, size=size, mode="area")

    return alpha_out.clamp(0.0, 1.0), fg_out.clamp(0.0, 1.0)
