"""Writer stage — orchestrator.

Public entry point: write_frame(frame, config)

Writes alpha, fg, and comp images to subdirectories under config.output_dir.
Output subdirectories are created on first write.

    alpha/  — alpha matte
    fg/     — foreground colour
    comp/   — checkerboard preview composite

Performance notes
-----------------
- Per-thread scratch buffers (``_tls``) are reused across frames on the same
  thread to avoid ~380 MB of numpy allocations per frame at 4K.
- OpenEXR (pyexr) is used for EXR writing when available. cv2 4.13.0 has a
  known bug where DWAB/DWAA compression produces corrupt files; pyexr avoids
  this entirely. Falls back to cv2 with PIZ compression when pyexr is absent.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from corridorkey.errors import WriteFailureError
from corridorkey.stages.postprocessor.contracts import ProcessedFrame
from corridorkey.stages.writer.contracts import WriteConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pyexr optional fast path
# ---------------------------------------------------------------------------

try:
    import Imath as _imath  # type: ignore[import-not-found]  # noqa: N813
    import OpenEXR as _pyexr  # type: ignore[import-not-found]  # noqa: N813

    _HAS_PYEXR = True
    logger.debug("writer: pyexr available — using OpenEXR for EXR writes")
except ImportError:
    _pyexr = None
    _imath = None
    _HAS_PYEXR = False
    logger.debug("writer: pyexr not available — falling back to cv2 PIZ")

# ---------------------------------------------------------------------------
# Per-thread scratch buffers (2.3)
# Reused across frames on the same thread to avoid repeated large allocations.
# ---------------------------------------------------------------------------

_tls = threading.local()

# EXR compression codec IDs (cv2 constant names not available in all builds).
_EXR_COMPRESSION_IDS: dict[str, int] = {
    "none": 0,
    "rle": 1,
    "zips": 2,
    "zip": 3,
    "piz": 4,
    "pxr24": 5,
    "dwaa": 6,
    "dwab": 7,
}
_EXR_ID_TO_NAME: dict[int, str] = {v: k for k, v in _EXR_COMPRESSION_IDS.items()}


def write_frame(frame: ProcessedFrame, config: WriteConfig) -> None:
    """Write all enabled outputs for one postprocessed frame to disk.

    Args:
        frame: ProcessedFrame from the postprocessor stage.
        config: WriteConfig controlling which outputs to write and where.

    Raises:
        OSError: If any cv2.imwrite call fails.
    """
    exr_flags = _exr_flags(config.exr_compression)
    exr_flags_f32 = _exr_flags(config.exr_compression, half=False)

    if config.alpha_enabled:
        _write(
            _alpha_to_bgr(frame.alpha),
            config.output_dir / "alpha" / f"{frame.stem}.{config.alpha_format}",
            config.alpha_format,
            exr_flags,
        )

    if config.fg_enabled:
        _write(
            cv2.cvtColor(frame.fg, cv2.COLOR_RGB2BGR),
            config.output_dir / "fg" / f"{frame.stem}.{config.fg_format}",
            config.fg_format,
            exr_flags,
        )

    if config.processed_enabled:
        # Premultiplied linear RGBA — convert to BGRA for cv2.
        # Written as 16-bit PNG or float32 EXR to preserve sub-pixel alpha precision.
        processed = frame.processed
        if config.processed_format == "png":
            # PNG viewers and compositors expect sRGB-encoded colour channels.
            # Convert the RGB channels from linear to sRGB before writing.
            # Alpha stays linear — alpha values are never gamma-encoded.
            from corridorkey.infra.colorspace import linear_to_srgb_lut, lut_apply

            rgb_srgb = lut_apply(np.clip(processed[:, :, :3], 0.0, 1.0), linear_to_srgb_lut)
            processed = np.concatenate([rgb_srgb, processed[:, :, 3:4]], axis=-1).astype(np.float32)
        bgra = cv2.cvtColor(processed, cv2.COLOR_RGBA2BGRA)
        _write(
            bgra,
            config.output_dir / "processed" / f"{frame.stem}.{config.processed_format}",
            config.processed_format,
            exr_flags_f32,  # float32 EXR — full precision for compositing
            sixteen_bit=(config.processed_format == "png"),
        )

    if config.comp_enabled:
        _write(
            cv2.cvtColor(frame.comp, cv2.COLOR_RGB2BGR),
            config.output_dir / "comp" / f"{frame.stem}.{config.comp_format}",
            config.comp_format,
            exr_flags,
        )

    logger.debug("write_frame: stem=%s", frame.stem)


def _exr_flags(compression: str, half: bool = True) -> list[int]:
    # cv2 4.13 DWAB/DWAA bug: fall back to PIZ (lossless) for those codecs.
    key = compression.lower()
    if key in ("dwab", "dwaa"):
        key = "piz"
    codec = _EXR_COMPRESSION_IDS.get(key, _EXR_COMPRESSION_IDS["piz"])
    exr_type = cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT
    return [cv2.IMWRITE_EXR_TYPE, exr_type, cv2.IMWRITE_EXR_COMPRESSION, codec]


def _exr_compression_name(exr_flags: list[int]) -> str:
    """Reverse-map an exr_flags list back to a compression name string.

    Used to pass the user-configured compression through to pyexr, which
    takes a name string rather than cv2 integer flags.
    """
    # exr_flags layout: [IMWRITE_EXR_TYPE, type_val, IMWRITE_EXR_COMPRESSION, codec_id]
    codec_id = exr_flags[3] if len(exr_flags) >= 4 else _EXR_COMPRESSION_IDS["piz"]
    return _EXR_ID_TO_NAME.get(codec_id, "piz")


def _alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    """Convert [H, W, 1] alpha to [H, W, 3] BGR grayscale for writing."""
    a2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    return np.stack([a2d, a2d, a2d], axis=-1)


def _write_exr_pyexr(img: np.ndarray, path: Path, compression: str) -> None:
    """Write an EXR file using OpenEXR (pyexr). Handles fp16 and fp32."""
    assert _pyexr is not None and _imath is not None

    comp_map = {
        "none": _pyexr.NO_COMPRESSION,
        "rle": _pyexr.RLE_COMPRESSION,
        "zips": _pyexr.ZIPS_COMPRESSION,
        "zip": _pyexr.ZIP_COMPRESSION,
        "piz": _pyexr.PIZ_COMPRESSION,
        "pxr24": _pyexr.PXR24_COMPRESSION,
        "dwaa": _pyexr.DWAA_COMPRESSION,
        "dwab": _pyexr.DWAB_COMPRESSION,
    }
    comp = comp_map.get(compression.lower(), _pyexr.ZIP_COMPRESSION)

    arr = img if img.dtype == np.float32 else img.astype(np.float32)
    h, w = arr.shape[:2]
    header = _pyexr.Header(w, h)
    header["compression"] = _imath.Compression(comp)

    if arr.ndim == 2:
        channels = {"Y": arr.tobytes()}
        header["channels"] = {"Y": _imath.Channel(_imath.FLOAT)}
    elif arr.shape[2] == 3:
        channels = {
            "R": arr[:, :, 0].tobytes(),
            "G": arr[:, :, 1].tobytes(),
            "B": arr[:, :, 2].tobytes(),
        }
        header["channels"] = {c: _imath.Channel(_imath.FLOAT) for c in "RGB"}
    else:
        channels = {
            "R": arr[:, :, 0].tobytes(),
            "G": arr[:, :, 1].tobytes(),
            "B": arr[:, :, 2].tobytes(),
            "A": arr[:, :, 3].tobytes(),
        }
        header["channels"] = {c: _imath.Channel(_imath.FLOAT) for c in "RGBA"}

    exr_file = _pyexr.OutputFile(str(path), header)
    exr_file.writePixels(channels)


def _write(img: np.ndarray, path: Path, fmt: str, exr_flags: list[int], sixteen_bit: bool = False) -> None:
    """Write a single image to disk using per-thread scratch buffers.

    Reuses pre-allocated numpy arrays on the calling thread to avoid
    ~95 MB of allocations per call at 4K.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise WriteFailureError(str(path), f"cannot create output directory: {e}") from e

    try:
        if fmt == "exr":
            arr = img if img.dtype == np.float32 else img.astype(np.float32)
            if _HAS_PYEXR:
                # pyexr avoids the cv2 4.13 DWAB/DWAA corruption bug.
                # Extract the compression name from exr_flags so the user's
                # configured codec is honoured (not silently overridden).
                compression = _exr_compression_name(exr_flags)
                _write_exr_pyexr(arr, path, compression)
                return
            ok = cv2.imwrite(str(path), arr, exr_flags)
        elif sixteen_bit:
            # Reuse per-thread uint16 scratch buffer.
            buf: np.ndarray | None = getattr(_tls, "u16_buf", None)
            if buf is None or buf.shape != img.shape:
                buf = np.empty(img.shape, dtype=np.uint16)
                _tls.u16_buf = buf
            np.multiply(np.clip(img, 0.0, 1.0), 65535.0, out=buf.astype(np.float32))
            buf[:] = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
            ok = cv2.imwrite(str(path), buf)
        else:
            # Reuse per-thread uint8 scratch buffer.
            buf = getattr(_tls, "u8_buf", None)
            if buf is None or buf.shape != img.shape:
                buf = np.empty(img.shape, dtype=np.uint8)
                _tls.u8_buf = buf
            buf[:] = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            ok = cv2.imwrite(str(path), buf)
    except OSError as e:
        raise WriteFailureError(str(path), str(e)) from e

    if not ok:
        raise WriteFailureError(str(path), "cv2.imwrite returned False")
