from __future__ import annotations

import numpy as np
import torch


def to_tensors(
    image: np.ndarray,
    alpha: np.ndarray,
    device: str,
    bgr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert image and alpha NumPy arrays to device tensors.

    Concatenates image and alpha into a single array before the device
    transfer so only one DMA operation is needed. Splits back on-device.
    Then reorders BGR→RGB on-device if needed.

    Args:
        image: float32 [H, W, 3], sRGB or linear, BGR or RGB channel order.
        alpha: float32 [H, W, 1], linear, range 0.0–1.0.
        device: PyTorch device string ("cuda", "mps", "cpu").
        bgr: If True, reorder image channels BGR→RGB on-device after transfer.
            Pass the value returned by ``read_frame_pair``.

    Returns:
        Tuple of (image [1, 3, H, W] RGB, alpha [1, 1, H, W]) on the device.
    """
    # HWC → CHW for both, then stack into [4, H, W] for a single transfer.
    img_chw = np.ascontiguousarray(image.transpose(2, 0, 1))  # [3, H, W]
    alp_chw = np.ascontiguousarray(alpha.transpose(2, 0, 1))  # [1, H, W]
    combined = np.concatenate([img_chw, alp_chw], axis=0)  # [4, H, W]

    t = (
        torch
        .from_numpy(combined)
        .unsqueeze(0)  # [1, 4, H, W]
        .float()
        .to(device)
    )

    img_t = t[:, :3]  # [1, 3, H, W]
    alp_t = t[:, 3:]  # [1, 1, H, W]

    if bgr:
        # Reorder BGR→RGB on-device. On CUDA this is a strided view — no copy.
        img_t = img_t[:, [2, 1, 0], :, :]

    return img_t, alp_t
