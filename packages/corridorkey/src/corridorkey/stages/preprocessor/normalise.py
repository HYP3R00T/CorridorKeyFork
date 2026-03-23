"""Preprocessing stage — ImageNet normalisation.

Normalises the image with ImageNet mean and std before inference.
This is a model input contract — the weights were trained exclusively
on inputs in this distribution.

Operates on a PyTorch tensor so the computation runs on whatever device the
tensor lives on (CUDA, MPS, or CPU).

The alpha hint is never normalised — it is passed through as-is.

In-place ops
------------
``sub_`` and ``div_`` modify the tensor in-place, eliminating two intermediate
allocations compared to ``(image - mean) / std``. This is safe here because
the tensor is freshly created by ``to_tensors`` and is not referenced anywhere
else at the point normalisation runs.

Caching
-------
The mean and std tensors are cached per (dtype, device) pair via
``functools.lru_cache``. For a 1000-frame clip this avoids 2000 small tensor
allocations — the cached tensors are reused on every frame.
"""

from __future__ import annotations

import functools

import torch

# ImageNet mean and std — model input contract, do not change.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


@functools.lru_cache(maxsize=8)
def _get_mean_std(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached mean and std tensors for the given dtype and device.

    Cached per (dtype, device) — at most 8 entries (one per device/dtype combo
    seen in practice). Avoids re-allocating these small tensors on every frame.
    """
    mean = torch.tensor(_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
    return mean, std


def normalise_image(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalisation to an sRGB image tensor in-place.

    Uses ``sub_`` / ``div_`` to avoid allocating intermediate tensors.
    Mean and std tensors are cached per (dtype, device) — no allocation on
    repeated calls with the same configuration.

    The input tensor is modified and returned — do not use the original
    reference after calling this function.

    Note on views: in-place ops on a non-contiguous view will modify the
    underlying storage, which may corrupt other tensors sharing that storage.
    ``contiguous()`` is called automatically to guard against this — it is a
    no-op when the tensor is already contiguous (the common case).

    Args:
        image: float32 tensor [B, 3, H, W] or [3, H, W], sRGB, range 0.0–1.0.

    Returns:
        The same tensor (or a contiguous copy if input was a non-contiguous
        view), normalised in-place. Values will be outside [0, 1] — expected.
    """
    squeezed = image.ndim == 3
    if squeezed:
        image = image.unsqueeze(0)

    # Guard: in-place ops on a non-contiguous view corrupt shared storage.
    # contiguous() is a no-op when already contiguous (the common case).
    image = image.contiguous()

    mean, std = _get_mean_std(image.dtype, image.device)
    image.sub_(mean).div_(std)

    return image.squeeze(0) if squeezed else image
