"""Stage 0 - entrypoint.

Accepts a path from the external interface (CLI, GUI, or API) and produces
a list of Clip objects ready for stage 1 to consume.

This is the only place that touches the filesystem for discovery purposes,
and the only place that reorganises the user's files (video normalisation).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})


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
        List of Clip objects ready for stage 1.

    Raises:
        ValueError: If the path does not exist, or if a file path is given with
            an unrecognised extension.
        PermissionError: If a directory cannot be read.
        OSError: If video reorganisation fails (e.g. file locked, read-only fs).
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    # Single video file - reorganise into a clip folder structure.
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"File is not a recognised video format: {path}")
        if reorganise:
            return [_normalise_video(path)]
        return []

    # If the directory itself looks like a clip, treat it as one.
    clip = _try_build_clip(path)
    if clip is not None:
        return [clip]

    # Otherwise scan subdirectories for clips.
    clips = []
    try:
        items = sorted(path.iterdir())
    except PermissionError as e:
        raise PermissionError(f"Cannot read directory: {path}") from e

    for item in items:
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
            if reorganise:
                clips.append(_normalise_video(item))
            continue
        if not item.is_dir():
            continue
        clip = _try_build_clip(item)
        if clip is not None:
            clips.append(clip)

    return clips


def _normalise_video(video_path: Path) -> Clip:
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
        return Clip(
            name=clip_root.name,
            root=clip_root,
            input_path=dest,
            alpha_path=None,
        )
    except ValidationError as e:
        raise OSError(f"Clip validation failed after reorganising '{video_path}': {e}") from e


def _try_build_clip(clip_dir: Path) -> Clip | None:
    """Attempt to build a Clip from a directory. Returns None if not a valid clip."""
    try:
        input_path = _find_input(clip_dir)
    except PermissionError as e:
        logger.warning("Skipping '%s': cannot read directory: %s", clip_dir, e)
        return None

    if input_path is None:
        return None

    alpha_path = _find_alpha(clip_dir)  # None if absent - stage 2 will generate it

    try:
        return Clip(
            name=clip_dir.name,
            root=clip_dir,
            input_path=input_path,
            alpha_path=alpha_path,
        )
    except ValidationError as e:
        logger.warning("Skipping '%s': validation failed: %s", clip_dir, e)
        return None


def _find_input(clip_dir: Path) -> Path | None:
    """Locate the Input/ directory inside a clip folder (case-insensitive).

    Raises:
        PermissionError: If clip_dir cannot be read.
    """
    return _find_icase(clip_dir, "Input")


def _find_alpha(clip_dir: Path) -> Path | None:
    """Locate the AlphaHint/ asset inside a clip folder (case-insensitive).

    Raises:
        PermissionError: If clip_dir cannot be read.
    """
    return _find_icase(clip_dir, "AlphaHint")


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


class Clip(BaseModel):
    """A clip ready for stage 1. Output contract of stage 0.

    Attributes:
        name: Human-readable clip name derived from the folder name.
        root: Absolute path to the clip folder.
        input_path: Path to the input asset. Either the Input/ directory (for
            pre-structured clips) or a video file inside Input/ (for normalised videos).
        alpha_path: Path to the alpha hint asset. None if absent (stage 2 required).
    """

    name: str
    root: Path
    input_path: Path
    alpha_path: Path | None

    @field_validator("root", "input_path", "alpha_path")
    @classmethod
    def must_exist(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @model_validator(mode="after")
    def root_must_be_directory(self) -> Clip:
        if not self.root.is_dir():
            raise ValueError(f"Clip root is not a directory: {self.root}")
        return self

    def __repr__(self) -> str:
        return f"Clip(name={self.name!r}, input={self.input_path}, alpha={self.alpha_path})"
