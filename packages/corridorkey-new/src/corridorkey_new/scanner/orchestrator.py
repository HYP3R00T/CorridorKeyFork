"""Scanner stage — orchestrator (stage 0).

Accepts a path from the external interface (CLI, GUI, or API) and produces
a list of Clip objects ready for the loader stage to consume.

This is the only place that touches the filesystem for discovery purposes,
and the only place that reorganises the user's files (video normalisation).

Each stage in the pipeline has a corresponding orchestrator.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corridorkey_new.scanner.contracts import Clip
from corridorkey_new.scanner.normaliser import VIDEO_EXTENSIONS, normalise_video, try_build_clip

logger = logging.getLogger(__name__)


def scan(path: str | Path, reorganise: bool = True) -> list[Clip]:
    """Scan a path for processable clips.

    Accepts:
    - A clips directory containing multiple clip subfolders
    - A single clip folder (must contain Input/ and optionally AlphaHint/)
    - A single video file (reorganised in-place into a clip folder structure)

    Args:
        path: Path to a clips directory, a single clip folder, or a video file.
        reorganise: If True (default), loose video files are moved into an
            Input/ subfolder in-place. If False, loose videos are skipped.

    Returns:
        List of Clip objects ready for the loader stage.

    Raises:
        ValueError: If the path does not exist or an unrecognised file type is given.
        PermissionError: If a directory cannot be read.
        OSError: If video reorganisation fails.
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"File is not a recognised video format: {path}")
        if reorganise:
            return [normalise_video(path)]
        return []

    clip = try_build_clip(path)
    if clip is not None:
        return [clip]

    clips = []
    try:
        items = sorted(path.iterdir())
    except PermissionError as e:
        raise PermissionError(f"Cannot read directory: {path}") from e

    for item in items:
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
            if reorganise:
                clips.append(normalise_video(item))
            continue
        if not item.is_dir():
            continue
        clip = try_build_clip(item)
        if clip is not None:
            clips.append(clip)

    return clips
