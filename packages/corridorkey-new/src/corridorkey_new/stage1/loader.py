"""Stage 1 - clip manifest builder.

Validates a Clip and returns a ClipManifest with paths and metadata.
If input or alpha is a video file, it is extracted to a PNG sequence first.
No pixel data is read beyond what ffmpeg writes to disk. Stage 3 handles
frame iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corridorkey_new.entrypoint import Clip
from corridorkey_new.stage1.contracts import ClipManifest
from corridorkey_new.stage1.extractor import extract_video, is_video
from corridorkey_new.stage1.validator import count_frames, detect_is_linear, validate

logger = logging.getLogger(__name__)


def load(clip: Clip) -> ClipManifest:
    """Validate a clip and return its manifest.

    If input or alpha is a video file, it is extracted to a PNG sequence
    inside a sibling ``Frames/`` or ``AlphaFrames/`` directory before
    validation runs.

    Args:
        clip: A Clip from stage 0.

    Returns:
        ClipManifest with validated sequence paths, needs_alpha flag,
        frame count, and is_linear.

    Raises:
        ValueError: If validation fails.
        RuntimeError: If video extraction fails.
    """
    input_path = _ensure_sequence(clip.input_path, "Frames")
    alpha_path = _ensure_sequence(clip.alpha_path, "AlphaFrames") if clip.alpha_path else None

    # Rebuild clip with resolved sequence paths for validation
    resolved = clip.model_copy(update={"input_path": input_path, "alpha_path": alpha_path})
    validate(resolved)

    return ClipManifest(
        clip_name=clip.name,
        input_path=input_path,
        alpha_path=alpha_path,
        needs_alpha=alpha_path is None,
        frame_count=count_frames(input_path),
        is_linear=detect_is_linear(input_path),
    )


def _ensure_sequence(path: Path, sibling_dir_name: str) -> Path:
    """Return a sequence directory for the given path.

    If path is already a directory, return it as-is.
    If path is a video file, extract it to a sibling directory and return that.

    Args:
        path: Input path — either a directory or a video file.
        sibling_dir_name: Name of the directory to create next to the video.

    Returns:
        Path to the image sequence directory.
    """
    if not is_video(path):
        return path

    output_dir = path.parent / sibling_dir_name
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("Sequence already exists, skipping extraction: %s", output_dir)
        return output_dir

    extract_video(path, output_dir)
    return output_dir
