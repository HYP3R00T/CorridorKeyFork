from __future__ import annotations

import logging
from pathlib import Path

from corridorkey.errors import ClipLoadError, ExtractionError
from corridorkey.events import PipelineEvents
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.loader.extractor import (
    DEFAULT_PNG_COMPRESSION,
    extract_video,
    is_video,
    read_video_metadata,
    save_video_metadata,
)
from corridorkey.stages.loader.validator import scan_frames, validate
from corridorkey.stages.scanner.contracts import Clip

logger = logging.getLogger(__name__)


def load(
    clip: Clip,
    events: PipelineEvents | None = None,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
) -> ClipManifest:
    """Validate a clip and return its manifest.

    For image sequence inputs, reads directly from Input/ and AlphaHint/.
    For video inputs, extracts frames into Frames/ and AlphaFrames/ without
    touching the original files.

    Args:
        clip: A Clip from stage 0.
        events: Optional PipelineEvents for extraction progress reporting.
        png_compression: PNG compression level for video extraction (0–9).
            Default 1 is recommended for intermediate frames.

    Returns:
        ClipManifest ready for preprocessing, or for the interface to generate
        alpha externally via attach_alpha() if needs_alpha is True.

    Raises:
        ClipLoadError: If input has no frames or the output directory cannot be created.
        FrameMismatchError: If alpha frame count does not match input frame count.
        ExtractionError: If video extraction fails.
    """
    frames_dir = _resolve_frames(
        clip.input_path,
        clip.root,
        "Frames",
        save_metadata=True,
        events=events,
        png_compression=png_compression,
    )
    alpha_frames_dir = (
        _resolve_frames(
            clip.alpha_path,
            clip.root,
            "AlphaFrames",
            save_metadata=False,
            events=events,
            png_compression=png_compression,
        )
        if clip.alpha_path
        else None
    )

    # Single scan pass per directory — validate() returns FrameScan results
    # so we reuse them for frame_count and is_linear without re-scanning.
    input_scan, _ = validate(clip.name, frames_dir, alpha_frames_dir)
    assert input_scan is not None  # noqa: S101 — only None when expected_frame_count is passed

    output_dir = clip.root / "Output"
    try:
        output_dir.mkdir(exist_ok=True)
    except OSError as e:
        raise ClipLoadError(clip.name, f"cannot create output directory '{output_dir}': {e}") from e

    video_meta_path: Path | None = None
    if is_video(clip.input_path):
        candidate = clip.root / "video_meta.json"
        if candidate.exists():
            video_meta_path = candidate
        else:
            logger.warning(
                "Video metadata missing for '%s' — stage 6 will not be able to re-encode.",
                clip.name,
            )

    return ClipManifest(
        clip_name=clip.name,
        clip_root=clip.root,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_frames_dir,
        output_dir=output_dir,
        needs_alpha=alpha_frames_dir is None,
        frame_count=input_scan.count,
        frame_range=(0, input_scan.count),
        is_linear=input_scan.is_linear,
        video_meta_path=video_meta_path,
        png_compression=png_compression,
    )


def _check_extraction_cache(video_path: Path, output_dir: Path) -> bool:
    # Returns True if output_dir already contains a complete extraction.
    # Verifies frame count against container metadata — not just non-empty —
    # to catch partial extractions from crashed runs.
    if not output_dir.exists():
        return False
    existing = scan_frames(output_dir)
    if existing.count == 0:
        return False
    try:
        meta = read_video_metadata(video_path)
        expected = meta.estimated_frame_count
    except RuntimeError:
        expected = 0
    if expected == 0 or existing.count == expected:
        # expected == 0 means the container doesn't report a frame count —
        # we can't verify completeness, so accept whatever is on disk.
        logger.info("Frames already extracted (%d frames), skipping: %s", existing.count, output_dir)
        return True
    logger.warning(
        "Incomplete extraction detected for '%s': %d frames on disk, %d expected — re-extracting.",
        output_dir,
        existing.count,
        expected,
    )
    return False


def _resolve_frames(
    path: Path,
    clip_root: Path,
    extracted_dir_name: str,
    save_metadata: bool = False,
    events: PipelineEvents | None = None,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
) -> Path:
    # Resolve the frame sequence directory for a given input path.
    # clip_root is passed explicitly rather than derived from path arithmetic
    # (path.parent.parent) to avoid fragile assumptions about directory depth.
    # save_metadata controls whether video metadata is written to clip_root
    # after extraction — True for the input video, False for alpha hint videos.
    if not is_video(path):
        return path

    output_dir = clip_root / extracted_dir_name

    if _check_extraction_cache(path, output_dir):
        return output_dir

    # Pre-open the container to get an accurate frame count for the progress
    # callback before extraction starts, so the GUI shows a real total.
    total_frames = 0
    try:
        meta = read_video_metadata(path)
        total_frames = meta.estimated_frame_count
    except RuntimeError:
        pass

    if events:
        events.stage_start("extract", total_frames)

    try:
        metadata = extract_video(
            path,
            output_dir,
            png_compression=png_compression,
            on_frame=events.extract_frame if events else None,
        )
    except RuntimeError as e:
        raise ExtractionError(clip_root.name, str(e)) from e

    if events:
        events.stage_done("extract")

    if save_metadata:
        save_video_metadata(metadata, clip_root)

    return output_dir


def attach_alpha(manifest: ClipManifest, alpha_frames_dir: Path) -> ClipManifest:
    """Update a manifest with an externally generated alpha sequence.

    Called by the interface layer (CLI, GUI, etc.) after it has generated alpha
    frames using an external tool. Alpha generation is not a pipeline stage —
    it is entirely the interface's responsibility.

    Validates the provided alpha directory matches the stored frame count
    (using the count already in the manifest — no re-scan of the input
    directory), then returns an updated manifest with ``needs_alpha=False``
    and ``alpha_frames_dir`` set, ready for preprocessing.

    Args:
        manifest: A ClipManifest with ``needs_alpha=True``.
        alpha_frames_dir: Path to the directory containing the generated
            alpha frame sequence.

    Returns:
        Updated ClipManifest with ``needs_alpha=False``, ready for preprocessing.

    Raises:
        ValueError: If manifest already has alpha or the directory doesn't exist.
        FrameMismatchError: If the alpha frame count doesn't match manifest.frame_count.
    """
    if not manifest.needs_alpha:
        raise ValueError(f"Clip '{manifest.clip_name}' already has alpha — attach_alpha should not be called.")

    if not alpha_frames_dir.exists():
        raise ValueError(f"Clip '{manifest.clip_name}': alpha_frames_dir does not exist: {alpha_frames_dir}")

    # Pass expected_frame_count so validate() skips re-scanning the input
    # directory — the count is already known from the manifest.
    validate(
        manifest.clip_name,
        manifest.frames_dir,
        alpha_frames_dir,
        expected_frame_count=manifest.frame_count,
    )

    return manifest.model_copy(
        update={
            "alpha_frames_dir": alpha_frames_dir,
            "needs_alpha": False,
        }
    )
