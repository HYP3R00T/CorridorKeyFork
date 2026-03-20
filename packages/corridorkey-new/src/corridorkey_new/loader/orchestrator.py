"""Loader stage — orchestrator.

Validates a Clip and returns a ClipManifest ready for downstream stages.
User files are never modified. For video inputs, frames are extracted into
a sibling Frames/ or AlphaFrames/ directory. For image sequences, the
existing directory is used directly.

Each stage in the pipeline has a corresponding orchestrator.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.extractor import extract_video, is_video, save_video_metadata
from corridorkey_new.loader.validator import count_frames, detect_is_linear, validate
from corridorkey_new.scanner.contracts import Clip

logger = logging.getLogger(__name__)


def load(clip: Clip) -> ClipManifest:
    """Validate a clip and return its manifest.

    For image sequence inputs, reads directly from Input/ and AlphaHint/.
    For video inputs, extracts frames into Frames/ and AlphaFrames/ without
    touching the original files.

    Args:
        clip: A Clip from stage 0.

    Returns:
        ClipManifest ready for preprocessing, or for the interface to generate
        alpha externally via resolve_alpha() if needs_alpha is True.

    Raises:
        ValueError: If validation fails.
        RuntimeError: If video extraction fails.
    """
    frames_dir = _resolve_frames(clip.input_path, "Frames")
    alpha_frames_dir = _resolve_frames(clip.alpha_path, "AlphaFrames") if clip.alpha_path else None

    validate(clip.name, frames_dir, alpha_frames_dir)

    output_dir = clip.root / "Output"
    output_dir.mkdir(exist_ok=True)

    frame_count = count_frames(frames_dir)

    # Carry the video metadata path so stage 6 can re-encode with matching
    # framerate, codec, colour space, etc. None for image sequence inputs.
    video_meta_path: Path | None = None
    if is_video(clip.input_path):
        candidate = clip.root / "video_meta.json"
        if candidate.exists():
            video_meta_path = candidate
        else:
            logger.warning("Video metadata missing for '%s' — stage 6 will not be able to re-encode.", clip.name)

    return ClipManifest(
        clip_name=clip.name,
        clip_root=clip.root,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_frames_dir,
        output_dir=output_dir,
        needs_alpha=alpha_frames_dir is None,
        frame_count=frame_count,
        frame_range=(0, frame_count),
        is_linear=detect_is_linear(frames_dir),
        video_meta_path=video_meta_path,
    )


def _resolve_frames(path: Path, extracted_dir_name: str) -> Path:
    """Resolve the frame sequence directory for a given input path.

    If path is a directory (image sequence), return it directly — no copy made.
    If path is a video file, extract frames into a sibling directory at the
    clip root level and return that. The video file is never moved or modified.

    Args:
        path: Input path — a frames directory or a video file.
        extracted_dir_name: Name for the extracted frames directory
            (``"Frames"`` or ``"AlphaFrames"``).

    Returns:
        Path to the frame sequence directory.
    """
    if not is_video(path):
        return path

    # path.parent is Input/ or AlphaHint/, path.parent.parent is the clip root
    output_dir = path.parent.parent / extracted_dir_name
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("Frames already extracted, skipping: %s", output_dir)
        return output_dir

    metadata = extract_video(path, output_dir)

    # Only save metadata for the main input video (not alpha video)
    if extracted_dir_name == "Frames":
        clip_root = path.parent.parent
        save_video_metadata(metadata, clip_root)

    return output_dir


def resolve_alpha(manifest: ClipManifest, alpha_frames_dir: Path) -> ClipManifest:
    """Update a manifest with an externally generated alpha sequence.

    Called by the interface layer (CLI, GUI, etc.) after it has generated alpha
    frames using an external tool. Alpha generation is not a pipeline stage —
    it is entirely the interface's responsibility.

    Validates the provided alpha directory matches the input frame count, then
    returns an updated manifest with ``needs_alpha=False`` and
    ``alpha_frames_dir`` set, ready for preprocessing.

    Args:
        manifest: A ClipManifest with ``needs_alpha=True``.
        alpha_frames_dir: Path to the directory containing the generated
            alpha frame sequence.

    Returns:
        Updated ClipManifest with ``needs_alpha=False``, ready for preprocessing.

    Raises:
        ValueError: If manifest already has alpha, the directory doesn't
            exist, or the frame count doesn't match.
    """
    if not manifest.needs_alpha:
        raise ValueError(f"Clip '{manifest.clip_name}' already has alpha — resolve_alpha should not be called.")

    validate(manifest.clip_name, manifest.frames_dir, alpha_frames_dir)

    return manifest.model_copy(
        update={
            "alpha_frames_dir": alpha_frames_dir,
            "needs_alpha": False,
        }
    )
