"""Postprocessor stage — orchestrator.

Runs steps 1–6 in order. Owns no transformation logic itself — each step
is delegated to its own module.

    Step 1   — resize tensors back to source resolution  → resize.py
    Step 1.5 — hint-guided sharpening (alpha + FG mask)  → hint_sharpen.py
    Step 2   — source passthrough (interior FG replace)  → composite.py
    Step 3   — alpha matte cleanup (despeckle)           → despeckle.py
    Step 4   — green spill removal                       → despill.py
    Step 5   — build processed RGBA + checkerboard comp  → composite.py
    Step 6   — return ProcessedFrame                 (here)

Public entry point: postprocess_frame(result, config, stem="")

GPU pipeline (CUDA devices)
---------------------------
When inference tensors are on a CUDA device, steps 1–4 run entirely on GPU
with a single GPU→CPU transfer at the end:

    GPU tensor
      → resize_to_source_gpu  (F.interpolate, stays on device)
      → despeckle_alpha_gpu   (max_pool2d flood-fill, no CPU sync)
      → remove_spill_gpu      (in-place ops, no CPU sync)
      → .cpu().numpy()        (one transfer for both alpha and fg)

Steps 1.5 (hint sharpen) and 2 (source passthrough) use OpenCV and run on
CPU after the single transfer. Steps 5–6 (composite, premultiply) are CPU.

CPU pipeline (CPU / MPS devices)
---------------------------------
All steps use the existing numpy/OpenCV paths unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.postprocessor.composite import apply_source_passthrough, make_preview, make_processed
from corridorkey.stages.postprocessor.config import PostprocessConfig
from corridorkey.stages.postprocessor.contracts import ProcessedFrame
from corridorkey.stages.postprocessor.despeckle import despeckle_alpha, despeckle_alpha_gpu
from corridorkey.stages.postprocessor.despill import remove_spill, remove_spill_gpu
from corridorkey.stages.postprocessor.hint_sharpen import sharpen_with_hint
from corridorkey.stages.postprocessor.resize import resize_to_source, resize_to_source_gpu

logger = logging.getLogger(__name__)


def postprocess_frame(
    result: InferenceResult,
    config: PostprocessConfig,
    stem: str = "",
    output_dir: Path | None = None,
) -> ProcessedFrame:
    """Postprocess a single inference result into output-ready numpy arrays.

    On CUDA devices, steps 1–4 (resize, despeckle, despill) run entirely on
    GPU with a single GPU→CPU transfer, matching the reference engine's
    fused postprocessing pipeline. On CPU/MPS the numpy/OpenCV paths are used.

    Args:
        result: InferenceResult from the inference stage.
        config: Postprocessing options (despill, despeckle, source_passthrough).
        stem: Filename stem for output naming (e.g. "frame_000001").
            Defaults to "frame_{frame_index:06d}" when empty.
        output_dir: Root output directory. Required when config.debug_dump=True.

    Returns:
        ProcessedFrame with all arrays at source resolution, float32.
    """
    meta = result.meta
    output_stem = stem or f"frame_{meta.frame_index:06d}"
    use_gpu = result.alpha.device.type == "cuda"

    if use_gpu:
        alpha_np, fg_np = _postprocess_gpu_pipeline(result, config, output_stem, output_dir)
    else:
        alpha_np, fg_np = _postprocess_cpu_pipeline(result, config, output_stem, output_dir)

    # Steps 1.5 and 2 always run on CPU (OpenCV-based)
    if config.hint_sharpen and meta.alpha_hint is not None:
        alpha_np, fg_np = sharpen_with_hint(
            alpha_np,
            fg_np,
            meta.alpha_hint,
            dilation_px=config.hint_sharpen_dilation,
        )
        if config.debug_dump and output_dir is not None:
            _debug_write(output_dir, output_stem, "01_hint_alpha", alpha_np, is_alpha=True)
            _debug_write(output_dir, output_stem, "01_hint_fg", fg_np)

    if config.source_passthrough and meta.source_image is not None:
        fg_np = apply_source_passthrough(
            fg_np,
            alpha_np,
            meta.source_image,
            config.edge_erode_px,
            config.edge_blur_px,
        )
        if config.debug_dump and output_dir is not None:
            _debug_write(output_dir, output_stem, "02_passthrough_fg", fg_np)

    # Step 5 — build outputs (always CPU)
    processed_np = make_processed(fg_np, alpha_np)
    comp_np = make_preview(fg_np, alpha_np, config.checkerboard_size)

    logger.debug(
        "postprocess_frame: frame=%d source=(%d,%d) stem=%s",
        meta.frame_index,
        meta.original_h,
        meta.original_w,
        output_stem,
    )

    return ProcessedFrame(
        alpha=alpha_np,
        fg=fg_np,
        processed=processed_np,
        comp=comp_np,
        frame_index=meta.frame_index,
        source_h=meta.original_h,
        source_w=meta.original_w,
        stem=output_stem,
    )


def _postprocess_gpu_pipeline(
    result: InferenceResult,
    config: PostprocessConfig,
    output_stem: str,
    output_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1–4 on GPU: resize → despeckle → despill → single CPU transfer.

    Returns (alpha_np [H, W, 1], fg_np [H, W, 3]) float32 numpy arrays.
    """
    meta = result.meta

    # Step 1 — resize on GPU (F.interpolate, no CPU sync)
    alpha_t, fg_t = resize_to_source_gpu(
        result.alpha,
        result.fg,
        meta.original_h,
        meta.original_w,
        fg_upsample_mode=config.fg_upsample_mode,
        alpha_upsample_mode=config.alpha_upsample_mode,
    )

    # Debug dump — raw inference output before postprocessing (requires CPU)
    if config.debug_dump and output_dir is not None:
        alpha_raw = _bchw_to_hwc_numpy(alpha_t)
        fg_raw = _bchw_to_hwc_numpy(fg_t)
        _debug_write(output_dir, output_stem, "00_raw_alpha", alpha_raw, is_alpha=True)
        _debug_write(output_dir, output_stem, "00_raw_fg", fg_raw)

    # Step 3 — despeckle on GPU (no CPU sync)
    if config.auto_despeckle:
        alpha_t = despeckle_alpha_gpu(
            alpha_t,
            config.despeckle_size,
            config.despeckle_dilation,
            config.despeckle_blur,
        )
        if config.debug_dump and output_dir is not None:
            _debug_write(output_dir, output_stem, "03_despeckle_alpha", _bchw_to_hwc_numpy(alpha_t), is_alpha=True)

    # Step 4 — despill on GPU (no CPU sync)
    fg_t = remove_spill_gpu(fg_t, config.despill_strength)
    if config.debug_dump and output_dir is not None:
        _debug_write(output_dir, output_stem, "04_despill_fg", _bchw_to_hwc_numpy(fg_t))

    # Single GPU→CPU transfer for both tensors
    alpha_np = _bchw_to_hwc_numpy(alpha_t)
    fg_np = _bchw_to_hwc_numpy(fg_t)

    return alpha_np, fg_np


def _postprocess_cpu_pipeline(
    result: InferenceResult,
    config: PostprocessConfig,
    output_stem: str,
    output_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1–4 on CPU: resize → despeckle → despill using numpy/OpenCV.

    Returns (alpha_np [H, W, 1], fg_np [H, W, 3]) float32 numpy arrays.
    """
    meta = result.meta

    # Step 1 — resize and convert to numpy
    alpha_np, fg_np = resize_to_source(
        result.alpha,
        result.fg,
        meta.original_h,
        meta.original_w,
        fg_upsample_mode=config.fg_upsample_mode,
        alpha_upsample_mode=config.alpha_upsample_mode,
    )

    if config.debug_dump and output_dir is not None:
        _debug_write(output_dir, output_stem, "00_raw_alpha", alpha_np, is_alpha=True)
        _debug_write(output_dir, output_stem, "00_raw_fg", fg_np)

    # Step 3 — despeckle on CPU
    if config.auto_despeckle:
        alpha_np = despeckle_alpha(
            alpha_np,
            config.despeckle_size,
            config.despeckle_dilation,
            config.despeckle_blur,
        )
        if config.debug_dump and output_dir is not None:
            _debug_write(output_dir, output_stem, "03_despeckle_alpha", alpha_np, is_alpha=True)

    # Step 4 — despill on CPU
    fg_np = remove_spill(fg_np, config.despill_strength)
    if config.debug_dump and output_dir is not None:
        _debug_write(output_dir, output_stem, "04_despill_fg", fg_np)

    return alpha_np, fg_np


def _bchw_to_hwc_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a [1, C, H, W] GPU tensor to a [H, W, C] float32 numpy array."""
    return t.squeeze(0).permute(1, 2, 0).cpu().float().numpy().astype(np.float32)


def _debug_write(output_dir: Path, stem: str, tag: str, arr: np.ndarray, *, is_alpha: bool = False) -> None:
    """Write a single debug frame as PNG. Silently skips on any error."""
    try:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{stem}__{tag}.png"
        if is_alpha:
            a2d = arr[:, :, 0] if arr.ndim == 3 else arr
            bgr = np.stack([a2d, a2d, a2d], axis=-1)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        uint8 = (np.clip(bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(path), uint8)
    except Exception as e:
        logger.debug("debug_write failed for %s__%s: %s", stem, tag, e)
