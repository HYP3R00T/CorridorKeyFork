"""Preprocessing stage — ImageNet normalisation (step 6).

Normalises the image with ImageNet mean and std before inference.
This is a model input contract — the weights were trained exclusively
on inputs in this distribution.

The alpha hint is never normalised — it is passed through as-is.
"""

from __future__ import annotations

import numpy as np

# ImageNet mean and std — model input contract, do not change.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalise_image(image: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalisation to an sRGB image.

    Args:
        image: float32 [H, W, 3] sRGB, range 0.0–1.0.

    Returns:
        float32 [H, W, 3], normalised. Values will be outside [0, 1] —
        that is expected and correct.
    """
    return ((image - _MEAN) / _STD).astype(np.float32)
