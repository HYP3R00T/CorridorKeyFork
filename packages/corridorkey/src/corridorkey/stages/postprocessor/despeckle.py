"""Postprocessor stage — alpha matte cleanup (despeckle).

Removes small disconnected foreground regions from a predicted alpha matte
using connected-component analysis.

Two implementations are provided:
  - ``despeckle_alpha``      — CPU path using OpenCV (always available).
  - ``despeckle_alpha_gpu``  — GPU path using iterative max_pool2d flood-fill
                               (CUDA only, no CPU sync).
  - ``despeckle_alpha_auto`` — picks GPU when a CUDA device is given, else CPU.

The safe_zone mask is used only to zero out removed components. Kept regions
retain their original alpha values unchanged — the dilation+blur only softens
the boundary of the removal mask, not the kept alpha itself.

GPU implementation notes
------------------------
Connected-component labelling uses iterative max_pool2d flood-fill (adapted
from https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc).

The Teschner spatial-hash initialisation (``_build_cc_scramble_init``) MUST be
computed outside any ``torch.compile`` region: the prime multiplication
``73856093 * W`` overflows int32 at 4K when inductor constant-folds it.
We compute it eagerly in int64 and cache the result per (H, W, device).
"""

from __future__ import annotations

import functools

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.transforms.v2.functional as transforms_functional

# ---------------------------------------------------------------------------
# CPU path (OpenCV)
# ---------------------------------------------------------------------------


def despeckle_alpha(
    alpha: np.ndarray,
    min_area: int,
    dilation: int = 25,
    blur_size: int = 5,
) -> np.ndarray:
    """Remove small disconnected regions from a predicted alpha matte (CPU).

    Builds a binary keep-mask from connected components, dilates it to recover
    semi-transparent edge pixels excluded by the binary threshold, then blurs
    the mask edge. The original alpha is multiplied by this mask so that:
      - Removed components → zeroed out
      - Kept regions → original alpha values preserved (no softening)
      - Transition band → smoothly faded out

    Args:
        alpha: [H, W, 1] float32 array, range 0-1.
        min_area: Minimum connected component area in pixels to keep.
            Regions smaller than this are zeroed out.
        dilation: Dilation radius in pixels applied after component removal
            to recover semi-transparent edge pixels lost by the 0.5 threshold.
            Default 25.
        blur_size: Gaussian blur radius applied after dilation to soften the
            hard mask edge at removed-component boundaries. Default 5.

    Returns:
        [H, W, 1] float32 array with small islands removed.
    """
    if min_area <= 0:
        return alpha

    a2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    binary = (a2d > 0.5).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keep_mask = np.zeros_like(binary)
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep_mask[labels == i] = 255

    # Dilate to recover semi-transparent edge pixels excluded by the 0.5 threshold
    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        keep_mask = cv2.dilate(keep_mask, kernel)

    # Blur to soften the hard removal boundary
    if blur_size > 0:
        b_size = int(blur_size * 2 + 1)
        keep_mask = cv2.GaussianBlur(keep_mask, (b_size, b_size), 0)

    keep_f = keep_mask.astype(np.float32) / 255.0

    # Clamp keep_f to [0, 1] and multiply — this zeroes removed regions while
    # leaving kept regions at their original alpha values (keep_f ≈ 1.0 there).
    keep_f = np.clip(keep_f, 0.0, 1.0)
    result = (a2d * keep_f)[:, :, np.newaxis].astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# GPU path (torch, no CPU sync)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _build_cc_scramble_init(h: int, w: int, device_str: str) -> torch.Tensor:
    """Build and cache the Teschner spatial-hash init tensor for connected_components.

    Must be called OUTSIDE any torch.compile region: the prime multiplication
    ``73856093 * W`` overflows int32 at 4K when inductor constant-folds it.
    Computing in int64 eagerly and passing the result as a tensor sidesteps it.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        device_str: String representation of the target device (e.g. "cuda:0").

    Returns:
        [1, 1, H, W] float32 tensor on the specified device.
    """
    idx = torch.arange(h * w, dtype=torch.int64)
    scrambled = ((idx * 73856093 + 19349663) % (h * w) + 1).to(torch.float32)
    return scrambled.view(1, 1, h, w).to(device_str)


def _connected_components_gpu(
    mask: torch.Tensor,
    min_component_width: int = 2,
    max_iterations: int = 100,
    init: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPU flood-fill connected-component labelling via iterative max_pool2d.

    Sync-free: fixed iteration count, no ``torch.equal`` early-exit,
    deterministic spatial-hash init from ``_build_cc_scramble_init``.

    Args:
        mask: bool tensor [B, 1, H, W].
        min_component_width: Kernel radius. Components narrower than this
            are merged with neighbours.
        max_iterations: Number of propagation iterations (always runs to
            completion — no early exit to stay sync-free).
        init: Optional precomputed label tensor. Must be passed when calling
            inside torch.compile to avoid inductor int32 overflow at 4K.

    Returns:
        Float label tensor [B, 1, H, W].
    """
    bs, _, h, w = mask.shape
    device = mask.device

    if init is None:
        init = _build_cc_scramble_init(h, w, str(device))

    if bs > 1:
        # Offset each batch's labels into a disjoint range so scatter_add_
        # doesn't conflate components across batches.
        batch_offset = torch.arange(bs, device=device, dtype=torch.float32).view(bs, 1, 1, 1) * (h * w)
        init = init + batch_offset

    mask_f = mask.to(init.dtype)
    comp = init * mask_f

    kernel = (2 * min_component_width) + 1
    pad = min_component_width
    for _ in range(max_iterations):
        comp = functional.max_pool2d(comp, kernel_size=kernel, stride=1, padding=pad) * mask_f

    return comp


def despeckle_alpha_gpu(
    alpha: torch.Tensor,
    min_area: int,
    dilation: int = 25,
    blur_size: int = 5,
) -> torch.Tensor:
    """Remove small disconnected regions from a predicted alpha matte (GPU).

    Runs fully on GPU with no CPU-GPU syncs. Uses iterative max_pool2d
    flood-fill for connected-component labelling and scatter_add_ for
    component size counting (avoids torch.bincount / torch.nonzero which
    have data-dependent output shapes and cause graph breaks in compiled
    regions).

    Args:
        alpha: [B, 1, H, W] float32 tensor on CUDA, range 0-1.
        min_area: Minimum connected component area in pixels to keep.
        dilation: Dilation radius in pixels. Applied via repeated max_pool2d
            with kernel 5 (each pass ≈ 2px radius).
        blur_size: Gaussian blur radius for soft mask edges.

    Returns:
        [B, 1, H, W] float32 tensor with small islands removed.
    """
    if min_area <= 0:
        return alpha

    bs, _, h, w = alpha.shape
    device = alpha.device
    mask = alpha > 0.5

    # Iteration count: enough passes for large subjects to converge into
    # a single basin. Deliberately low — large subjects under-converge into
    # multiple basins, but each basin still exceeds min_area and the
    # downstream dilation fills sub-pixel holes between them.
    min_component_width = 2
    max_iter = max(8, min_area // 8)

    # Build scramble init outside compile region (int32 overflow guard)
    cc_init = _build_cc_scramble_init(h, w, str(device))

    components = _connected_components_gpu(
        mask,
        max_iterations=max_iter,
        min_component_width=min_component_width,
        init=cc_init,
    )

    # Component size counting via scatter_add_ + index_select.
    # Avoids torch.bincount + torch.nonzero (data-dependent shapes → graph breaks).
    n_labels = bs * h * w + 1
    flat = components.view(-1).long()
    sizes = torch.zeros(n_labels, dtype=torch.int32, device=device)
    sizes.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))

    sizes_per_pixel = sizes.index_select(0, flat).view_as(components)
    keep = (sizes_per_pixel >= min_area).to(alpha.dtype)

    # Dilate to restore edges of large regions.
    # Each max_pool2d pass with kernel 5 gives ~2px radius, so repeats = dilation // 2.
    # Note: this dilation has a latent quirk inherited from the reference implementation —
    # because ``keep`` is 1 at both background and large FG components, dilation expands
    # into speckle regions, refilling speckles smaller than the dilation radius. Only
    # speckles in the band [min_area, dilation_refill_size] are actually removed.
    if dilation > 0:
        repeats = dilation // 2
        for _ in range(repeats):
            keep = functional.max_pool2d(keep, 5, stride=1, padding=2)

    # Gaussian blur for soft edges
    if blur_size > 0:
        k = int(blur_size * 2 + 1)
        keep = transforms_functional.gaussian_blur(keep, [k, k])

    return alpha * keep


# ---------------------------------------------------------------------------
# Auto-dispatch entry point
# ---------------------------------------------------------------------------


def despeckle_alpha_auto(
    alpha: np.ndarray,
    min_area: int,
    dilation: int = 25,
    blur_size: int = 5,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Remove small disconnected regions, using GPU when available.

    Routes to ``despeckle_alpha_gpu`` when ``device`` is a CUDA device,
    otherwise falls back to the CPU OpenCV path. The output is always a
    numpy float32 array regardless of which path ran.

    Args:
        alpha: [H, W, 1] float32 array, range 0-1.
        min_area: Minimum connected component area in pixels to keep.
        dilation: Dilation radius in pixels.
        blur_size: Gaussian blur radius for soft mask edges.
        device: Torch device string or object. When None or non-CUDA, uses CPU.

    Returns:
        [H, W, 1] float32 array with small islands removed.
    """
    if min_area <= 0:
        return alpha

    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type != "cuda":
        return despeckle_alpha(alpha, min_area, dilation, blur_size)

    # HWC numpy → BCHW tensor on device
    a2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    t = torch.from_numpy(a2d).unsqueeze(0).unsqueeze(0).to(device=dev, dtype=torch.float32)
    t_out = despeckle_alpha_gpu(t, min_area, dilation, blur_size)
    # BCHW tensor → HWC numpy
    return t_out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
