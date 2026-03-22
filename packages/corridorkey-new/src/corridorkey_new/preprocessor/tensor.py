"""Preprocessing stage — tensor construction (steps 7–9).

Converts NumPy arrays from the reader into PyTorch tensors in channel-first
layout and moves them to the target device. All subsequent transforms operate
on these tensors directly on the device.

This is the boundary between NumPy (CPU disk I/O) and PyTorch (device compute).
"""

from __future__ import annotations

import numpy as np
import torch


def to_tensors(
    image: np.ndarray,
    alpha: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert image and alpha NumPy arrays to device tensors.

    Steps:
        7. Transpose [H, W, C] → [C, H, W], add batch dim → [1, C, H, W]
        8. Cast to float32
        9. Move to device

    Args:
        image: float32 [H, W, 3], sRGB or linear.
        alpha: float32 [H, W, 1], linear, range 0.0–1.0.
        device: PyTorch device string ("cuda", "mps", "cpu").

    Returns:
        Tuple of (image [1, 3, H, W], alpha [1, 1, H, W]) on the specified device.
    """
    img_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    alp_t = torch.from_numpy(alpha.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return img_t, alp_t


def to_tensor(
    image: np.ndarray,
    alpha: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Build the final model input tensor by concatenating image and alpha.

    Kept for backward compatibility with any callers that use the old single-
    tensor API. Prefer ``to_tensors`` when transforms need to be applied
    between construction and concatenation.

    Args:
        image: float32 [H, W, 3], ImageNet-normalised.
        alpha: float32 [H, W, 1], linear, range 0.0–1.0.
        device: PyTorch device string ("cuda", "mps", "cpu").

    Returns:
        float32 tensor [1, 4, H, W] on the specified device.
    """
    img_t, alp_t = to_tensors(image, alpha, device)
    return torch.cat([img_t, alp_t], dim=1)  # [1, 4, H, W]
