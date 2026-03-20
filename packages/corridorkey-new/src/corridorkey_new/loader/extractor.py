"""Stage 1 - video extraction.

Extracts video files to PNG image sequences using PyAV (Python bindings to
FFmpeg C libraries). PyAV bundles its own FFmpeg — no system install required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import cv2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})


def is_video(path: Path) -> bool:
    """Check if a path points to a video file."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def extract_video(video_path: Path, output_dir: Path, pattern: str = "frame_{:06d}.png") -> None:
    """Extract a video file to a PNG image sequence using PyAV.

    Frames are written as lossless PNGs. The output filenames follow
    ``pattern`` (default: ``frame_000000.png``, ``frame_000001.png``, ...).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frames into (created if needed).
        pattern: Python format string for frame filenames. Must contain one
            integer placeholder (e.g. ``"frame_{:06d}.png"``).

    Raises:
        RuntimeError: If the video cannot be opened or a frame cannot be decoded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting video: %s -> %s", video_path, output_dir)

    try:
        container = av.open(str(video_path))
    except av.AVError as e:
        raise RuntimeError(f"Cannot open video '{video_path}': {e}") from e

    stream = container.streams.video[0]
    stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO

    frame_index = 0
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                # Convert to BGR numpy array for cv2
                bgr = frame.to_ndarray(format="bgr24")
                out_path = output_dir / pattern.format(frame_index)
                cv2.imwrite(str(out_path), bgr)
                frame_index += 1
    except av.AVError as e:
        raise RuntimeError(f"Error decoding '{video_path}' at frame {frame_index}: {e}") from e
    finally:
        container.close()

    logger.info("Extracted %d frames to %s", frame_index, output_dir)
