"""Scanner stage — video normaliser and clip discovery helpers.

Handles reorganising loose video files into the expected clip folder structure,
and locating Input/ and AlphaHint/ assets inside a clip directory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import ValidationError

from corridorkey_new.scanner.contracts import Clip

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})


def normalise_video(video_path: Path) -> Clip:
    """Reorganise a loose video file into a clip folder structure in-place.

    Given ``parent/random_name.mp4``, produces:
        parent/
          Input/
            random_name.mp4   <- moved here
          AlphaHint/          <- created empty

    If the destination file already exists inside ``Input/``, the video is not moved again.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Clip with root=parent, input_path=parent/Input/video, alpha_path=None.

    Raises:
        OSError: If directory creation or file move fails.
    """
    clip_root = video_path.parent
    input_dir = clip_root / "Input"
    alpha_dir = clip_root / "AlphaHint"

    try:
        input_dir.mkdir(exist_ok=True)
        alpha_dir.mkdir(exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create clip structure in '{clip_root}': {e}") from e

    dest = input_dir / video_path.name
    if not dest.exists():
        try:
            shutil.move(str(video_path), str(dest))
            logger.info("Moved video '%s' -> '%s'", video_path, dest)
        except OSError as e:
            raise OSError(f"Failed to move '{video_path}' to '{dest}': {e}") from e
    else:
        logger.debug("Video already in place, skipping move: '%s'", dest)

    try:
        return Clip(name=clip_root.name, root=clip_root, input_path=dest, alpha_path=None)
    except ValidationError as e:
        raise OSError(f"Clip validation failed after reorganising '{video_path}': {e}") from e


def try_build_clip(clip_dir: Path) -> Clip | None:
    """Attempt to build a Clip from a directory. Returns None if not a valid clip."""
    try:
        input_path = find_input(clip_dir)
    except PermissionError as e:
        logger.warning("Skipping '%s': cannot read directory: %s", clip_dir, e)
        return None

    if input_path is None:
        return None

    alpha_path = find_alpha(clip_dir)

    try:
        return Clip(name=clip_dir.name, root=clip_dir, input_path=input_path, alpha_path=alpha_path)
    except ValidationError as e:
        logger.warning("Skipping '%s': validation failed: %s", clip_dir, e)
        return None


def find_input(clip_dir: Path) -> Path | None:
    """Locate the input asset inside a clip folder (case-insensitive).

    Returns the video path if Input/ contains a video, otherwise the directory.

    Raises:
        PermissionError: If clip_dir cannot be read.
    """
    input_dir = _find_icase(clip_dir, "Input")
    if input_dir is None:
        return None
    video = _find_video_in(input_dir)
    return video if video is not None else input_dir


def find_alpha(clip_dir: Path) -> Path | None:
    """Locate the AlphaHint asset inside a clip folder (case-insensitive).

    Returns the video path if AlphaHint/ contains a video, otherwise the directory.
    """
    alpha_dir = _find_icase(clip_dir, "AlphaHint")
    if alpha_dir is None:
        return None
    video = _find_video_in(alpha_dir)
    return video if video is not None else alpha_dir


def _find_video_in(directory: Path) -> Path | None:
    """Return the first video file found in a directory, or None."""
    try:
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                return child
    except PermissionError:
        pass
    return None


def _find_icase(parent: Path, name: str) -> Path | None:
    """Case-insensitive lookup of a child entry inside parent.

    Raises:
        PermissionError: If parent cannot be read.
    """
    try:
        for child in parent.iterdir():
            if child.name.lower() == name.lower():
                return child
    except PermissionError as e:
        raise PermissionError(f"Cannot read directory: {parent}") from e
    return None
