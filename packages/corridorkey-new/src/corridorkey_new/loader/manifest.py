"""Stage 1 - clip manifest builder.

Validates a Clip and returns a ClipManifest with paths and metadata.
User files are never modified. For video inputs, frames are extracted into
a sibling Frames/ or AlphaFrames/ directory. For image sequences, the
existing directory is used directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corridorkey_new.entrypoint import Clip
from corridorkey_new.loader.contracts import ClipLayout, ClipManifest
from corridorkey_new.loader.extractor import extract_video, is_video
from corridorkey_new.loader.validator import count_frames, detect_is_linear, validate

logger = logging.getLogger(__name__)


def load(clip: Clip) -> ClipManifest:
    """Validate a clip and return its manifest.

    For image sequence inputs, reads directly from Input/ and AlphaHint/.
    For video inputs, extracts frames into Frames/ and AlphaFrames/ without
    touching the original files.

    Args:
        clip: A Clip from stage 0.

    Returns:
        ClipManifest with validated layout, needs_alpha flag,
        frame count, and is_linear.

    Raises:
        ValueError: If validation fails.
        RuntimeError: If video extraction fails.
    """
    frames_dir = _resolve_frames(clip.input_path, "Frames")
    alpha_frames_dir = _resolve_frames(clip.alpha_path, "AlphaFrames") if clip.alpha_path else None

    layout = ClipLayout(
        root=clip.root,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_frames_dir,
    )

    resolved = clip.model_copy(update={"input_path": frames_dir, "alpha_path": alpha_frames_dir})
    validate(resolved)

    return ClipManifest(
        clip_name=clip.name,
        layout=layout,
        needs_alpha=alpha_frames_dir is None,
        frame_count=count_frames(frames_dir),
        is_linear=detect_is_linear(frames_dir),
    )


def _resolve_frames(path: Path, extracted_dir_name: str) -> Path:
    """Resolve the frame sequence directory for a given input path.

    If path is a directory (image sequence), return it directly — no copy made.
    If path is a video file, extract frames into a sibling directory and return
    that. The video file is never moved or modified.

    Args:
        path: Input path — a frames directory or a video file.
        extracted_dir_name: Name for the extracted frames directory
            (``"Frames"`` or ``"AlphaFrames"``).

    Returns:
        Path to the frame sequence directory.
    """
    if not is_video(path):
        return path

    output_dir = path.parent.parent / extracted_dir_name
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("Frames already extracted, skipping: %s", output_dir)
        return output_dir

    extract_video(path, output_dir)
    return output_dir
