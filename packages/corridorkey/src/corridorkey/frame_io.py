"""Unified frame I/O - read images and video frames as float32 RGB.

All reading functions return float32 arrays in [0, 1] range with RGB channel
order. EXR files are read as-is (linear float); standard formats (PNG, JPG,
etc.) are normalised from uint8.

This module consolidates frame-reading patterns that were previously
duplicated across service.py methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from corridorkey_core.compositing import linear_to_srgb

from corridorkey.validators import normalize_mask_channels, normalize_mask_dtype

logger = logging.getLogger(__name__)

# EXR write flags - DWAA half-float: ~5x faster writes than PXR24, half the file size.
# DWAA is a lossy DCT-based compression standard used widely in VFX pipelines (Nuke, Resolve).
# Visually lossless at default quality for compositing work.
EXR_WRITE_FLAGS = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION,
    6,  # DWAA (cv2.IMWRITE_EXR_COMPRESSION_DWAA not available in all builds)
]


def read_image_frame(fpath: str, gamma_correct_exr: bool = False) -> np.ndarray | None:
    """Read an image file (EXR or standard) as float32 RGB in [0, 1].

    Args:
        fpath: Absolute path to the image file.
        gamma_correct_exr: Apply the piecewise sRGB transfer function to EXR
            data when True (converts linear to sRGB for models expecting sRGB).

    Returns:
        float32 array of shape [H, W, 3] in RGB order, or None if the read fails.
    """
    is_exr = fpath.lower().endswith(".exr")

    if is_exr:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read frame: %s", fpath)
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = np.maximum(img_rgb, 0.0).astype(np.float32)
        if gamma_correct_exr:
            srgb = linear_to_srgb(result)
            result = np.asarray(srgb, dtype=np.float32)
        return result
    else:
        img = cv2.imread(fpath)
        if img is None:
            logger.warning("Could not read frame: %s", fpath)
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32) / 255.0


def read_video_frame_at(video_path: str, frame_index: int) -> np.ndarray | None:
    """Read a single frame from a video by index as float32 RGB in [0, 1].

    Args:
        video_path: Path to the video file.
        frame_index: Zero-based frame index to seek to.

    Returns:
        float32 array of shape [H, W, 3] in RGB order, or None if the seek
        or read fails.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    finally:
        cap.release()


def read_video_frames(
    video_path: str,
    processor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Read all frames from a video, optionally applying a processor to each.

    Without a processor, frames are returned as float32 RGB in [0, 1].

    Args:
        video_path: Path to the video file.
        processor: Optional callable that receives a BGR uint8 frame and
            returns a processed array. When None, default conversion to
            float32 RGB [0, 1] is applied.

    Returns:
        List of processed frames.
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if processor is not None:
                frames.append(processor(frame))
            else:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(img_rgb)
    finally:
        cap.release()
    return frames


def read_mask_frame(
    fpath: str,
    clip_name: str = "",
    frame_index: int = 0,
) -> np.ndarray | None:
    """Read a mask frame as float32 [H, W] in [0, 1].

    Handles any channel count and dtype via normalize_mask_channels and
    normalize_mask_dtype.

    Args:
        fpath: Path to the mask image.
        clip_name: Clip name for error context.
        frame_index: Frame index for error context.

    Returns:
        float32 array of shape [H, W] in [0, 1], or None if the read fails.
    """
    mask_in = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if mask_in is None:
        return None
    # dtype normalisation MUST happen before channel extraction because
    # normalize_mask_channels casts to float32, which would leave a uint8
    # value of 255 as float32 255.0 and skip the /255 division.
    mask = normalize_mask_dtype(mask_in)
    mask = normalize_mask_channels(mask, clip_name, frame_index)
    return mask


def read_video_mask_at(video_path: str, frame_index: int) -> np.ndarray | None:
    """Read a single mask frame from a video by index as float32 [H, W] in [0, 1].

    Extracts the blue channel (index 2) from BGR, matching the convention
    used by alpha-channel video masks.

    Args:
        video_path: Path to the video file.
        frame_index: Zero-based frame index.

    Returns:
        float32 array of shape [H, W] in [0, 1], or None if the seek or
        read fails.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame[:, :, 2].astype(np.float32) / 255.0
    finally:
        cap.release()


@dataclass
class FrameData:
    """Output of load_frame. Input to the core engine's preprocessing stage.

    Attributes:
        image: RGB float32 [H, W, 3] sRGB, values 0-1.
        mask: Grayscale float32 [H, W, 1] linear, values 0-1.
            0.0 = definite background, 1.0 = definite foreground, 0.5 = unknown.
        source_h: Original frame height in pixels.
        source_w: Original frame width in pixels.
        is_linear: True if the source image was originally in linear light (e.g. EXR).
            The image field always contains sRGB - this flag records the origin.
        stem: Filename stem of the source frame (e.g. "frame_000001").
    """

    image: np.ndarray
    mask: np.ndarray
    source_h: int
    source_w: int
    is_linear: bool = False
    stem: str = ""


def load_frame(
    image_path: str | Path,
    mask_path: str | Path,
    input_is_linear: bool = False,
    stem: str = "",
) -> FrameData:
    """Read one image frame and its corresponding mask from disk.

    Normalises both to float32 in [0, 1]. If the source image is in linear
    light it is converted to sRGB here so all downstream stages always
    receive sRGB. The ``is_linear`` flag on the returned FrameData records
    the original colour space for reference.

    Args:
        image_path: Path to the source image (PNG, EXR, TIFF, etc.).
        mask_path: Path to the alpha hint mask (any single-channel image).
        input_is_linear: True if the source is in linear light (e.g. EXR).
        stem: Filename stem to carry through the pipeline for output naming.
            Defaults to the image filename stem when not provided.

    Returns:
        FrameData with image [H, W, 3] sRGB float32 and mask [H, W, 1] float32.

    Raises:
        FileNotFoundError: If either path does not exist.
        OSError: If either file cannot be decoded by OpenCV.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    is_exr = image_path.suffix.lower() == ".exr"
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Could not decode image: {image_path}")

    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    if is_exr:
        image = np.maximum(img_rgb, 0.0).astype(np.float32)
        if input_is_linear:
            image = np.asarray(linear_to_srgb(image), dtype=np.float32)
    else:
        image = img_rgb.astype(np.float32) / 255.0

    h, w = image.shape[:2]

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise OSError(f"Could not decode mask: {mask_path}")

    if mask_raw.dtype == np.uint8:
        mask_f = mask_raw.astype(np.float32) / 255.0
    elif mask_raw.dtype == np.uint16:
        mask_f = mask_raw.astype(np.float32) / 65535.0
    else:
        mask_f = mask_raw.astype(np.float32)

    if mask_f.ndim == 3:
        mask_f = mask_f[:, :, 0]
    mask_f = mask_f[:, :, np.newaxis]

    return FrameData(
        image=image,
        mask=mask_f,
        source_h=h,
        source_w=w,
        is_linear=input_is_linear,
        stem=stem or image_path.stem,
    )
