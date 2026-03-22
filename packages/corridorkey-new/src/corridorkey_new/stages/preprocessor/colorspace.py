"""Preprocessing stage — color space conversion (step 4).

Converts linear light images to sRGB before inference.
The model requires sRGB input — this is an input contract, not an optimisation.

Operates on a PyTorch tensor so the computation runs on whatever device the
tensor lives on (CUDA, MPS, or CPU).
"""

from __future__ import annotations

import torch

# sRGB transfer function constants (IEC 61966-2-1)
_LINEAR_THRESHOLD = 0.0031308
_LINEAR_SCALE = 12.92
_GAMMA = 1.0 / 2.4
_SCALE = 1.055
_OFFSET = 0.055


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """Convert a linear light float32 tensor to sRGB.

    Uses the IEC 61966-2-1 piecewise transfer function. Values are clamped
    to [0, 1] before conversion. Runs on whatever device the tensor lives on.

    Only call this when ``ClipManifest.is_linear`` is True.

    Args:
        image: float32 tensor [..., 3], linear light, range 0.0–1.0.

    Returns:
        float32 tensor [..., 3], sRGB gamma-encoded, range 0.0–1.0.
    """
    x = image.clamp(0.0, 1.0)
    linear_part = x * _LINEAR_SCALE
    gamma_part = _SCALE * x.clamp(min=1e-12).pow(_GAMMA) - _OFFSET
    return torch.where(x <= _LINEAR_THRESHOLD, linear_part, gamma_part)
