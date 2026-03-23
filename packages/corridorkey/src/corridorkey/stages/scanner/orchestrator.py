"""Scanner stage — orchestrator (stage 0).

Accepts a path from the external interface (CLI, GUI, or API) and produces
a ScanResult containing valid Clip objects and any skipped paths with reasons.

This is the only place that touches the filesystem for discovery purposes,
and the only place that reorganises the user's files (video normalisation).

Each stage in the pipeline has a corresponding orchestrator.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corridorkey.errors import ClipScanError
from corridorkey.events import PipelineEvents
from corridorkey.infra.utils import VIDEO_EXTENSIONS
from corridorkey.stages.scanner.contracts import Clip, ScanResult, SkippedPath
from corridorkey.stages.scanner.normaliser import normalise_video, try_build_clip

logger = logging.getLogger(__name__)


def scan(
    path: str | Path,
    reorganise: bool = True,
    events: PipelineEvents | None = None,
) -> ScanResult:
    """Scan a path for processable clips.

    Accepts:
    - A clips directory containing multiple clip subfolders
    - A single clip folder (must contain Input/ and optionally AlphaHint/)
    - A single video file (reorganised in-place into a clip folder structure)

    Args:
        path: Path to a clips directory, a single clip folder, or a video file.
        reorganise: If True (default), loose video files are moved into an
            Input/ subfolder in-place. If False, loose videos are reported as
            skipped rather than silently ignored.
        events: Optional PipelineEvents for streaming clip discovery to a GUI.
            on_clip_found fires for each valid clip as it is discovered.
            on_clip_skipped fires for each path that could not be used.

    Returns:
        ScanResult with clips ready for the loader stage and any skipped paths.

    Raises:
        ClipScanError: If the path does not exist or is an unrecognised file type.
        PermissionError: If the top-level directory cannot be read.
        OSError: If video reorganisation fails.
    """
    path = Path(path)

    if not path.exists():
        raise ClipScanError(f"Path does not exist: {path}")

    clips: list[Clip] = []
    skipped: list[SkippedPath] = []

    # --- Single video file ---
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ClipScanError(f"File is not a recognised video format: {path}")
        if reorganise:
            clip = normalise_video(path)
            clips.append(clip)
            if events:
                events.clip_found(clip.name, clip.root)
        else:
            reason = "reorganise=False — loose video files are not processed without reorganisation"
            logger.warning("Skipping '%s': %s", path, reason)
            skipped.append(SkippedPath(path=path, reason=reason))
            if events:
                events.clip_skipped(reason, path)
        return ScanResult(clips=tuple(clips), skipped=tuple(skipped))

    # --- Single clip folder ---
    clip, skip = try_build_clip(path)
    if clip is not None:
        clips.append(clip)
        if events:
            events.clip_found(clip.name, clip.root)
        return ScanResult(clips=tuple(clips), skipped=tuple(skipped))
    if skip is not None:
        skipped.append(skip)
        if events:
            events.clip_skipped(skip.reason, skip.path)
        return ScanResult(clips=tuple(clips), skipped=tuple(skipped))

    # --- Clips directory ---
    try:
        items = sorted(path.iterdir())
    except PermissionError as e:
        raise PermissionError(f"Cannot read directory: {path}") from e

    for item in items:
        if item.is_file():
            if item.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if reorganise:
                clip = normalise_video(item)
                clips.append(clip)
                if events:
                    events.clip_found(clip.name, clip.root)
            else:
                reason = "reorganise=False — loose video files are not processed without reorganisation"
                logger.debug("Skipping loose video '%s': %s", item, reason)
                skipped.append(SkippedPath(path=item, reason=reason))
                if events:
                    events.clip_skipped(reason, item)
            continue

        if not item.is_dir():
            continue

        clip, skip = try_build_clip(item)
        if clip is not None:
            clips.append(clip)
            if events:
                events.clip_found(clip.name, clip.root)
        elif skip is not None:
            skipped.append(skip)
            if events:
                events.clip_skipped(skip.reason, skip.path)

    if not clips and not skipped:
        logger.warning("No clips found in '%s'", path)

    return ScanResult(clips=tuple(clips), skipped=tuple(skipped))
