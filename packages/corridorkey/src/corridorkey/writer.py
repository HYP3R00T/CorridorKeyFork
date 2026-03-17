"""Frame output writer and mask generation stub.

write_outputs   - write all enabled output images for one processed frame.
generate_masks  - stage 2 placeholder; raises NotImplementedError until a
                  generator package is wired in.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from corridorkey.contracts import WriteConfig
from corridorkey.errors import WriteFailureError

if TYPE_CHECKING:
    from corridorkey_core.contracts import ProcessedFrame

# EXR compression codec IDs (cv2 constant names not available in all builds).
EXR_COMPRESSION_IDS: dict[str, int] = {
    "none": 0,
    "rle": 1,
    "zips": 2,
    "zip": 3,
    "piz": 4,
    "pxr24": 5,
    "dwaa": 6,
    "dwab": 7,
}


def exr_flags(compression: str) -> list[int]:
    """Return cv2.imwrite flags for EXR with the given compression codec."""
    codec = EXR_COMPRESSION_IDS.get(compression.lower(), EXR_COMPRESSION_IDS["dwaa"])
    return [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF, cv2.IMWRITE_EXR_COMPRESSION, codec]


def write_outputs(frame: ProcessedFrame, cfg: WriteConfig) -> None:
    """Write all enabled output images for one processed frame to disk.

    Handles dtype conversion: EXR expects float32, PNG expects uint8.
    All colour-space conversions happened in stage 5 - this function only writes.

    Args:
        frame: ProcessedFrame from stage_5_postprocess.
        cfg: WriteConfig controlling which outputs to write and where.

    Raises:
        WriteFailureError: If any cv2.imwrite call fails.
    """
    stem = frame.stem
    flags = exr_flags(cfg.exr_compression)

    def _write(img: np.ndarray, path: str, fmt: str, clip_name: str = "", frame_index: int = 0) -> None:
        if fmt == "exr":
            arr = img if img.dtype == np.float32 else img.astype(np.float32)
            ok = cv2.imwrite(path, arr, flags)
        else:
            arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8) if img.dtype != np.uint8 else img
            ok = cv2.imwrite(path, arr)
        if not ok:
            raise WriteFailureError(clip_name, frame_index, path)

    if cfg.fg_enabled and "fg" in cfg.dirs:
        fg_bgr = cv2.cvtColor(frame.fg, cv2.COLOR_RGB2BGR)
        _write(fg_bgr, os.path.join(cfg.dirs["fg"], f"{stem}.{cfg.fg_format}"), cfg.fg_format, stem)

    if cfg.matte_enabled and "matte" in cfg.dirs:
        alpha = frame.alpha[:, :, 0] if frame.alpha.ndim == 3 else frame.alpha
        _write(alpha, os.path.join(cfg.dirs["matte"], f"{stem}.{cfg.matte_format}"), cfg.matte_format, stem)

    if cfg.comp_enabled and "comp" in cfg.dirs:
        comp_bgr = cv2.cvtColor((np.clip(frame.comp, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _write(comp_bgr, os.path.join(cfg.dirs["comp"], f"{stem}.{cfg.comp_format}"), cfg.comp_format, stem)

    if cfg.processed_enabled and "processed" in cfg.dirs:
        proc_bgra = cv2.cvtColor(frame.processed, cv2.COLOR_RGBA2BGRA)
        _write(
            proc_bgra, os.path.join(cfg.dirs["processed"], f"{stem}.{cfg.processed_format}"), cfg.processed_format, stem
        )


def generate_masks(
    frames_dir: str,
    output_dir: str,
    generator: Any = None,
    **kwargs: Any,
) -> None:
    """Generate alpha hint masks for a frame sequence (stage 2 placeholder).

    Delegates to an external AlphaGenerator implementation. If the output
    directory is already populated the caller should skip this stage entirely.

    Supported generators (future - install as separate packages):
        corridorkey-gbm     Chroma key / GBM-based generator.
        corridorkey-sam2    SAM2 - strongest temporal consistency.
        corridorkey-gvm     GVM / VideoMaMa - handles difficult footage.

    Args:
        frames_dir: Directory containing the source frame sequence.
        output_dir: Directory where generated mask frames will be written.
        generator: An object implementing the AlphaGenerator protocol.
            None raises NotImplementedError.
        **kwargs: Forwarded to the generator's generate() method.

    Raises:
        NotImplementedError: Always, until a generator implementation is wired in.
    """
    if generator is None:
        raise NotImplementedError(
            "generate_masks: no generator provided. "
            "Install a mask generator package (e.g. corridorkey-gbm, corridorkey-sam2) "
            "and pass its generator instance here."
        )
    raise NotImplementedError(
        f"generate_masks: generator protocol not yet wired. Received generator of type '{type(generator).__name__}'."
    )
