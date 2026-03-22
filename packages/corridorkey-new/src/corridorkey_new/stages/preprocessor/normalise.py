"""Preprocessing stage — ImageNet normalisation (step 6).

Normalises the image with ImageNet mean and std before inference.
This is a model input contract — the weights were trained exclusively
on inputs in this distribution.

Operates on a PyTorch tensor so the computation runs on whatever device the
tensor lives on (CUDA, MPS, or CPU).

The alpha hint is never normalised — it is passed through as-is.
"""

from __future__ import annotations

import torch

# ImageNet mean and std — model input contract, do not change.
# Shape [1, 3, 1, 1] for broadcasting against [B, 3, H, W].
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def normalise_image(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalisation to an sRGB image tensor.

    Args:
        image: float32 tensor [B, 3, H, W] or [3, H, W], sRGB, range 0.0–1.0.

    Returns:
        float32 tensor, same shape, normalised. Values will be outside [0, 1] —
        that is expected and correct.
    """
    mean = torch.tensor(_MEAN, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    # Ensure batch dim for broadcasting, then restore original rank if needed
    if image.ndim == 3:
        return ((image.unsqueeze(0) - mean) / std).squeeze(0)
    return (image - mean) / std
