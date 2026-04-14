from __future__ import annotations

import logging
from pathlib import Path

from corridorkey.errors import ClipScanError
from corridorkey.events import PipelineEvents
from corridorkey.infra.utils import VIDEO_EXTENSIONS
from corridorkey.stages.scanner.contracts import Clip, ScanResult, SkippedClip
from corridorkey.stages.scanner.normaliser import normalise_video, try_build_clip

logger = logging.getLogger(__name__)


def scan(
    paths: list[Path] | Path | str,
    reorganise: bool = True,
    events: PipelineEvents | None = None,
) -> ScanResult:
    """Scan one or more paths for processable clips.

    Accepts a single path or a list of paths. Each path may be:
    - A clips directory containing multiple clip subfolders
    - A single clip folder (must contain Input/ and optionally AlphaHint/)
    - A single video file (reorganised in-place into a clip folder structure)

    Passing a list allows scanning clips that live at unrelated locations on
    disk (different drives, different projects) in a single call.

    Args:
        paths: A single path or list of paths to scan.
        reorganise: If True (default), loose video files are moved into an
            Input/ subfolder in-place. If False, loose videos are reported as
            skipped rather than silently ignored.
        events: Optional PipelineEvents for streaming clip discovery to a GUI.
            on_clip_found fires for each valid clip as it is discovered.
            on_clip_skipped fires for each path that could not be used.

    Returns:
        ScanResult with clips ready for the loader stage and any skipped paths.

    Raises:
        ClipScanError: If a path does not exist, is an unrecognised file type,
            a directory cannot be read, or video reorganisation fails.
    """
    # Normalise to list
    path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]

    all_clips: list[Clip] = []
    all_skipped: list[SkippedClip] = []

    for path in path_list:
        result = _scan_one(path, reorganise=reorganise, events=events)
        all_clips.extend(result.clips)
        all_skipped.extend(result.skipped)

    return ScanResult(clips=tuple(all_clips), skipped=tuple(all_skipped))


def _scan_one(
    path: Path,
    reorganise: bool,
    events: PipelineEvents | None = None,
) -> ScanResult:
    if not path.exists():
        raise ClipScanError(f"Path does not exist: {path}")

    if path.is_file():
        return _scan_video_file(path, reorganise=reorganise, events=events)

    clip, skip = try_build_clip(path)
    if clip is not None or skip is not None:
        return _scan_clip_folder(clip, skip, events=events)

    return _scan_clips_directory(path, reorganise=reorganise, events=events)


def _scan_video_file(
    path: Path,
    reorganise: bool,
    events: PipelineEvents | None,
) -> ScanResult:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ClipScanError(f"File is not a recognised video format: {path}")

    if not reorganise:
        reason = "reorganise=False — loose video files are not processed without reorganisation"
        logger.warning("Skipping '%s': %s", path, reason)
        if events:
            events.clip_skipped(reason, path)
        return ScanResult(clips=(), skipped=(SkippedClip(path=path, reason=reason),))

    clip = normalise_video(path)
    if events:
        events.clip_found(clip.name, clip.root)
    return ScanResult(clips=(clip,), skipped=())


def _scan_clip_folder(
    clip: Clip | None,
    skip: SkippedClip | None,
    events: PipelineEvents | None,
) -> ScanResult:
    if clip is not None:
        if events:
            events.clip_found(clip.name, clip.root)
        return ScanResult(clips=(clip,), skipped=())

    # skip is non-None here: try_build_clip returns (None, SkippedClip) or (Clip, None)
    assert skip is not None  # noqa: S101
    if events:
        events.clip_skipped(skip.reason, skip.path)
    return ScanResult(clips=(), skipped=(skip,))


def _scan_clips_directory(
    path: Path,
    reorganise: bool,
    events: PipelineEvents | None,
) -> ScanResult:
    try:
        items = sorted(path.iterdir())
    except PermissionError as e:
        raise ClipScanError(f"Cannot read directory: {path}") from e

    clips: list[Clip] = []
    skipped: list[SkippedClip] = []

    for item in items:
        if item.is_file():
            _collect_loose_video(item, reorganise=reorganise, events=events, clips=clips, skipped=skipped)
        elif item.is_dir():
            _collect_clip_folder(item, events=events, clips=clips, skipped=skipped)

    if not clips and not skipped:
        logger.warning("No clips found in '%s'", path)

    return ScanResult(clips=tuple(clips), skipped=tuple(skipped))


def _collect_loose_video(
    item: Path,
    reorganise: bool,
    events: PipelineEvents | None,
    clips: list[Clip],
    skipped: list[SkippedClip],
) -> None:
    if item.suffix.lower() not in VIDEO_EXTENSIONS:
        return
    if reorganise:
        clip = normalise_video(item)
        clips.append(clip)
        if events:
            events.clip_found(clip.name, clip.root)
    else:
        reason = "reorganise=False — loose video files are not processed without reorganisation"
        logger.debug("Skipping loose video '%s': %s", item, reason)
        skipped.append(SkippedClip(path=item, reason=reason))
        if events:
            events.clip_skipped(reason, item)


def _collect_clip_folder(
    item: Path,
    events: PipelineEvents | None,
    clips: list[Clip],
    skipped: list[SkippedClip],
) -> None:
    clip, skip = try_build_clip(item)
    if clip is not None:
        clips.append(clip)
        if events:
            events.clip_found(clip.name, clip.root)
    elif skip is not None:
        skipped.append(skip)
        if events:
            events.clip_skipped(skip.reason, skip.path)
