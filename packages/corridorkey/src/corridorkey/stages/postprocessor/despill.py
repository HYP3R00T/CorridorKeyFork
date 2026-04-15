from __future__ import annotations

import numpy as np
import torch


def remove_spill(fg: np.ndarray, strength: float) -> np.ndarray:
    """Remove green spill from a foreground image (CPU numpy path).

    Luminance-preserving suppression: excess green is redistributed equally
    to red and blue channels to neutralise the spill without darkening the subject.

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


def remove_spill_gpu(fg: torch.Tensor, strength: float) -> torch.Tensor:
    """Remove green spill on GPU using in-place ops (no CPU sync).

    Keeps data on device throughout. All ops are in-place after the initial
    clone to avoid mutating the input tensor.

    Args:
        fg: [B, 3, H, W] float32 BCHW tensor on CUDA, range 0-1.
        strength: Blend factor 0.0 (no change) to 1.0 (full despill).

    Returns:
        [B, 3, H, W] float32 tensor with spill suppressed, same device as input.
    """
    if strength <= 0.0:
        return fg

    out = fg.clone()
    r, g, b = out[:, 0], out[:, 1], out[:, 2]  # mutable channel views

    limit = r.add(b).div_(2.0)  # (R + B) / 2, in-place on limit
    spill = (g - limit).clamp_(min=0.0)  # excess green, in-place clamp
    g.sub_(spill)  # G -= spill
    spill_half = spill.mul_(0.5)  # reuse spill buffer for spill/2
    r.add_(spill_half)  # R += spill/2
    b.add_(spill_half)  # B += spill/2

    if strength < 1.0:
        out.lerp_(fg, 1.0 - strength)  # blend toward original: out = out + (fg - out) * (1 - s)

    return out


def remove_spill_auto(
    fg: np.ndarray,
    strength: float,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Remove green spill, using GPU when available.

    Routes to ``remove_spill_gpu`` when ``device`` is a CUDA device,
    otherwise falls back to the CPU numpy path. The output is always a
    numpy float32 array regardless of which path ran.

    Args:
        fg: [H, W, 3] float32 sRGB array, range 0-1.
        strength: Blend factor 0.0 (no change) to 1.0 (full despill).
        device: Torch device string or object. When None or non-CUDA, uses CPU.

    Returns:
        [H, W, 3] float32 array with spill suppressed.
    """
    if strength <= 0.0:
        return fg

    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type != "cuda":
        return remove_spill(fg, strength)

    # HWC numpy → BCHW tensor on device
    t = torch.from_numpy(fg).permute(2, 0, 1).unsqueeze(0).to(device=dev, dtype=torch.float32)
    t_out = remove_spill_gpu(t, strength)
    # BCHW tensor → HWC numpy
    return t_out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
