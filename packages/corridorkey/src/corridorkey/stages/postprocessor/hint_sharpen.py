"""Postprocessor — hint-guided alpha and FG sharpening.

Uses the alpha hint (at source resolution) to build a hard binary mask that
eliminates the soft edge tails introduced when upscaling model output from
inference resolution to source resolution.

The model output at inference resolution is actually sharp, but a 1.875×
Lanczos upscale spreads each edge transition over ~23px at 4K vs ~8px in the
hint. This module restores that sharpness by:

    1. Binarising the hint at native resolution (threshold 0.5).
    2. Dilating by ``dilation_px`` to give a small breathing margin so the
       model's fine edge detail is not clipped.
    3. Upscaling with INTER_NEAREST to source resolution — preserves the hard
       boundary exactly.
    4. Multiplying both alpha and FG by the mask — kills the soft tail and
       zeros white bleed in the background zone.

Public entry point: sharpen_with_hint(alpha, fg, hint, dilation_px)
"""

from __future__ import annotations

import cv2
import numpy as np


def sharpen_with_hint(
    alpha: np.ndarray,
    fg: np.ndarray,
    hint: np.ndarray,
    dilation_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a hard binary mask derived from the alpha hint to alpha and FG.

    Args:
        alpha: Alpha matte [H, W, 1] float32 0-1 at source resolution.
        fg: Foreground image [H, W, 3] float32 0-1 at source resolution.
        hint: Alpha hint [H, W, 1] float32 0-1 at source (hint native) resolution.
            Must already be at source resolution (same H, W as alpha/fg).
        dilation_px: Dilation radius in pixels applied to the binarised hint
            before upscaling. Gives breathing room so fine model edge detail
            is not clipped. Default 3.

    Returns:
        Tuple of (alpha, fg) with the hard mask applied, same shapes as input.
    """
    src_h, src_w = alpha.shape[:2]

    # Step 1 — binarise hint at its native resolution
    hint_2d = hint[:, :, 0] if hint.ndim == 3 else hint
    binary = (hint_2d >= 0.5).astype(np.uint8)

    # Step 2 — dilate to recover breathing room around edges
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_px + 1, 2 * dilation_px + 1),
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    # Step 3 — upscale to source resolution with INTER_NEAREST (hard boundary)
    if binary.shape[0] != src_h or binary.shape[1] != src_w:
        binary = cv2.resize(binary, (src_w, src_h), interpolation=cv2.INTER_NEAREST)

    # Step 4 — apply mask: zero out everything outside the hint region
    mask = binary[:, :, np.newaxis].astype(np.float32)
    alpha_out = alpha * mask
    fg_out = fg * mask

    return alpha_out, fg_out
