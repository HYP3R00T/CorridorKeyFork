"""Postprocessor stage — orchestrator.

Runs steps 1–5 in order. Owns no transformation logic itself — each step
is delegated to its own module.

    Step 1 — resize tensors back to source resolution  → resize.py
    Step 2 — green spill removal                       → despill.py
    Step 3 — alpha matte cleanup (despeckle)           → despeckle.py
    Step 4 — checkerboard preview composite            → composite.py
    Step 5 — return PostprocessedFrame                 (here)

Public entry point: postprocess_frame(result, config, stem="")
"""

from __future__ import annotations

import logging

from corridorkey_new.inference.contracts import InferenceResult
from corridorkey_new.postprocessor.composite import make_preview
from corridorkey_new.postprocessor.config import PostprocessConfig
from corridorkey_new.postprocessor.contracts import PostprocessedFrame
from corridorkey_new.postprocessor.despeckle import despeckle_alpha
from corridorkey_new.postprocessor.despill import remove_spill
from corridorkey_new.postprocessor.resize import resize_to_source

logger = logging.getLogger(__name__)


def postprocess_frame(
    result: InferenceResult,
    config: PostprocessConfig,
    stem: str = "",
) -> PostprocessedFrame:
    """Postprocess a single inference result into output-ready numpy arrays.

    Args:
        result: InferenceResult from the inference stage.
        config: Postprocessing options (despill, despeckle, checkerboard).
        stem: Filename stem for output naming (e.g. "frame_000001").
            Defaults to "frame_{frame_index:06d}" when empty.

    Returns:
        PostprocessedFrame with all arrays at source resolution, float32.
    """
    meta = result.meta

    # Step 1 — resize tensors back to source resolution and convert to numpy
    alpha_np, fg_np = resize_to_source(
        result.alpha,
        result.fg,
        meta.original_h,
        meta.original_w,
    )

    # Step 2 — green spill removal
    fg_np = remove_spill(fg_np, config.despill_strength)

    # Step 3 — alpha matte cleanup
    if config.auto_despeckle:
        alpha_np = despeckle_alpha(alpha_np, config.despeckle_size)

    # Step 4 — checkerboard preview composite
    comp_np = make_preview(fg_np, alpha_np, config.checkerboard_size)

    output_stem = stem or f"frame_{meta.frame_index:06d}"

    logger.debug(
        "postprocess_frame: frame=%d source=(%d,%d) stem=%s",
        meta.frame_index,
        meta.original_h,
        meta.original_w,
        output_stem,
    )

    # Step 5 — return
    return PostprocessedFrame(
        alpha=alpha_np,
        fg=fg_np,
        comp=comp_np,
        frame_index=meta.frame_index,
        source_h=meta.original_h,
        source_w=meta.original_w,
        stem=output_stem,
    )
