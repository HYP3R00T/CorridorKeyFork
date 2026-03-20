"""Preprocessing stage — tensor conversion (steps 7–9).

Concatenates image and alpha, transposes to PyTorch channel-first layout,
adds the batch dimension, and moves the result to the target device.

This is the boundary between NumPy (CPU) and PyTorch (GPU).
"""

from __future__ import annotations

import numpy as np
import torch


def to_tensor(
    image: np.ndarray,
    alpha: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Build the model input tensor from normalised image and alpha arrays.

    Steps:
        7. Concatenate image [H, W, 3] + alpha [H, W, 1] → [H, W, 4]
        8. Transpose to [4, H, W], add batch dim → [1, 4, H, W], cast float32
        9. Move to device

    Args:
        image: float32 [H, W, 3], ImageNet-normalised.
        alpha: float32 [H, W, 1], linear, range 0.0–1.0.
        device: PyTorch device string ("cuda", "mps", "cpu").

    Returns:
        float32 tensor [1, 4, H, W] on the specified device.
    """
    combined = np.concatenate([image, alpha], axis=-1)  # [H, W, 4]
    chw = combined.transpose((2, 0, 1))  # [4, H, W]
    tensor = torch.from_numpy(chw).unsqueeze(0).float()  # [1, 4, H, W]
    return tensor.to(device)
