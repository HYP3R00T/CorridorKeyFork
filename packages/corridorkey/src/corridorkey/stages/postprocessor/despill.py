"""Postprocessor stage — green spill removal.

Luminance-preserving green spill suppression. Excess green is redistributed
equally to red and blue channels to neutralise the spill without darkening
the subject.
"""

from __future__ import annotations

import numpy as np


def remove_spill(fg: np.ndarray, strength: float) -> np.ndarray:
    """Remove green spill from a foreground image.

    Args:
        fg: [H, W, 3] float32 sRGB array, range 0-1.
        strength: Blend factor 0.0 (no change) to 1.0 (full despill).

    Returns:
        [H, W, 3] float32 array with spill suppressed.
    """
    if strength <= 0.0:
        return fg

    r = fg[:, :, 0]
    g = fg[:, :, 1]
    b = fg[:, :, 2]

    green_limit = (r + b) / 2.0
    spill_amount = np.maximum(g - green_limit, 0.0)

    g_new = g - spill_amount
    r_new = r + spill_amount * 0.5
    b_new = b + spill_amount * 0.5

    despilled = np.stack([r_new, g_new, b_new], axis=-1).astype(np.float32)

    if strength < 1.0:
        return (fg * (1.0 - strength) + despilled * strength).astype(np.float32)

    return despilled
