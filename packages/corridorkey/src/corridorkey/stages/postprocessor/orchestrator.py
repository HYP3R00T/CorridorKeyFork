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
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.postprocessor.composite import apply_source_passthrough, make_preview, make_processed
from corridorkey.stages.postprocessor.config import PostprocessConfig
from corridorkey.stages.postprocessor.contracts import ProcessedFrame
from corridorkey.stages.postprocessor.despeckle import despeckle_alpha
from corridorkey.stages.postprocessor.despill import remove_spill
from corridorkey.stages.postprocessor.hint_sharpen import sharpen_with_hint
from corridorkey.stages.postprocessor.resize import resize_to_source

logger = logging.getLogger(__name__)


def postprocess_frame(
    result: InferenceResult,
    config: PostprocessConfig,
    stem: str = "",
    output_dir: Path | None = None,
) -> ProcessedFrame:
    """Postprocess a single inference result into output-ready numpy arrays.

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

    # Step 1 — resize tensors back to source resolution
    alpha_np, fg_np = resize_to_source(
        result.alpha,
        result.fg,
        meta.original_h,
        meta.original_w,
        fg_upsample_mode=config.fg_upsample_mode,
        alpha_upsample_mode=config.alpha_upsample_mode,
    )

    # Debug dump — raw inference output before any postprocessing
    if config.debug_dump and output_dir is not None:
        _debug_write(output_dir, output_stem, "00_raw_alpha", alpha_np, is_alpha=True)
        _debug_write(output_dir, output_stem, "00_raw_fg", fg_np)

    # Step 1.5 — hint-guided sharpening: apply a hard binary mask derived from
    # the alpha hint to eliminate soft edge tails introduced by upscaling, and
    # zero FG white bleed in the background zone.
    # Runs before source_passthrough so the passthrough only fills the interior
    # region that the mask has already confirmed as foreground.
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

    # Step 2 — source passthrough: replace model FG in opaque interior regions
    # with original source pixels to eliminate dark fringing from background
    # contamination in the model FG prediction.
    # Must run before despill so that despill is applied to the already-blended
    # FG (including the passed-through source pixels).
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

    # Step 3 — alpha matte cleanup
    if config.auto_despeckle:
        alpha_np = despeckle_alpha(alpha_np, config.despeckle_size, config.despeckle_dilation, config.despeckle_blur)
        if config.debug_dump and output_dir is not None:
            _debug_write(output_dir, output_stem, "03_despeckle_alpha", alpha_np, is_alpha=True)

    # Step 4 — green spill removal (on straight sRGB FG, after passthrough is blended in)
    fg_np = remove_spill(fg_np, config.despill_strength)
    if config.debug_dump and output_dir is not None:
        _debug_write(output_dir, output_stem, "04_despill_fg", fg_np)

    # Step 5 — build outputs
    processed_np = make_processed(fg_np, alpha_np)
    comp_np = make_preview(fg_np, alpha_np, config.checkerboard_size)

    logger.debug(
        "postprocess_frame: frame=%d source=(%d,%d) stem=%s",
        meta.frame_index,
        meta.original_h,
        meta.original_w,
        output_stem,
    )

    # Step 6 — return
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


def _debug_write(output_dir: Path, stem: str, tag: str, arr: np.ndarray, *, is_alpha: bool = False) -> None:
    """Write a single debug frame as PNG. Silently skips on any error."""
    try:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{stem}__{tag}.png"
        if is_alpha:
            # Alpha [H, W, 1] → grayscale BGR
            a2d = arr[:, :, 0] if arr.ndim == 3 else arr
            bgr = np.stack([a2d, a2d, a2d], axis=-1)
        else:
            # FG [H, W, 3] RGB → BGR
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        uint8 = (np.clip(bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(path), uint8)
    except Exception as e:
        logger.debug("debug_write failed for %s__%s: %s", stem, tag, e)
