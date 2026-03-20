"""Stage 1 - video extraction.

Extracts video files to PNG image sequences using ffmpeg.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})


def is_video(path: Path) -> bool:
    """Check if a path points to a video file."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def extract_video(video_path: Path, output_dir: Path, pattern: str = "frame_%06d.png") -> None:
    """Extract a video file to a PNG image sequence.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frames into (created if needed).
        pattern: Frame filename pattern in FFmpeg style (default: frame_%06d.png).

    Raises:
        RuntimeError: If ffmpeg is not found or extraction fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg to extract video files")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-start_number",
        "0",
        "-vsync",
        "passthrough",
        str(output_dir / pattern),
        "-y",
    ]

    logger.info("Extracting video: %s -> %s", video_path, output_dir)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Video extraction timed out after 300s: {video_path}") from e

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed for '{video_path}': {result.stderr[:500]}")

    extracted_count = len(list(output_dir.glob("*.png")))
    logger.info("Extracted %d frames to %s", extracted_count, output_dir)
