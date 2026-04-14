from __future__ import annotations

import torch
import torch.nn.functional as functional


def resize_frame(
    image: torch.Tensor,
    alpha: torch.Tensor,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and alpha to img_size × img_size using bilinear interpolation.

    Matches the reference pipeline exactly:
        cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        cv2.resize(mask,  (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    Args:
        image: float32 [1, 3, H, W], sRGB, range 0.0–1.0.
        alpha: float32 [1, 1, H, W], linear, range 0.0–1.0.
        img_size: Target square resolution (e.g. 2048).

    Returns:
        Tuple of:
            image  [1, 3, img_size, img_size] float32
            alpha  [1, 1, img_size, img_size] float32, clamped to [0, 1]
    """
    src_h, src_w = image.shape[2], image.shape[3]

    if src_h == img_size and src_w == img_size:
        return image, alpha.clamp(0.0, 1.0)

    size = (img_size, img_size)
    img_out = functional.interpolate(image, size=size, mode="bilinear", align_corners=False)
    alp_out = functional.interpolate(alpha, size=size, mode="bilinear", align_corners=False)

    return img_out, alp_out.clamp(0.0, 1.0)
