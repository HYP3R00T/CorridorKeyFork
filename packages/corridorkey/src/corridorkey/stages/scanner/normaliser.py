"""Scanner stage — video normaliser and clip discovery helpers.

Handles reorganising loose video files into the expected clip folder structure,
and locating Input/ and AlphaHint/ assets inside a clip directory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import ValidationError

from corridorkey.infra.utils import VIDEO_EXTENSIONS
from corridorkey.stages.scanner.contracts import Clip, SkippedClip

logger = logging.getLogger(__name__)


def normalise_video(video_path: Path) -> Clip:
    """Reorganise a loose video file into a clip folder structure in-place.

    Given ``parent/random_name.mp4``, produces::

        parent/
          Input/
            random_name.mp4   <- moved here (safe copy-verify-delete)
          AlphaHint/          <- created empty

    If the destination file already exists inside ``Input/`` with the same
    size, the source is not moved again (idempotent).

    The move is performed as copy → size-verify → delete to avoid data loss
    on cross-filesystem moves where ``shutil.move`` would fall back to a
    non-atomic copy-then-delete without verification.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Clip with root=parent, input_path=parent/Input/video, alpha_path=None.

    Raises:
        OSError: If directory creation, copy, or delete fails.
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
    if dest.exists() and dest.stat().st_size == video_path.stat().st_size:
        logger.debug("Video already in place, skipping move: '%s'", dest)
    else:
        _safe_move(video_path, dest)

    try:
        return Clip(name=clip_root.name, root=clip_root, input_path=dest, alpha_path=None)
    except ValidationError as e:
        raise OSError(f"Clip validation failed after reorganising '{video_path}': {e}") from e


def try_build_clip(clip_dir: Path) -> tuple[Clip | None, SkippedClip | None]:
    """Attempt to build a Clip from a directory.

    Returns:
        (Clip, None) on success.
        (None, SkippedPath) if the directory was a recognisable clip structure
            that failed validation.
        (None, None) if the directory is simply not a clip (no Input/ folder).
    """
    try:
        input_path, input_skip = find_input(clip_dir)
    except PermissionError as e:
        reason = f"cannot read directory: {e}"
        logger.warning("Skipping '%s': %s", clip_dir, reason)
        return None, SkippedClip(path=clip_dir, reason=reason)

    if input_path is None:
        if input_skip is not None:
            # Has an Input/ folder but it's ambiguous or empty — report it.
            return None, input_skip
        # No Input/ folder at all — not a clip, silently ignore.
        return None, None

    alpha_path, alpha_skip = find_alpha(clip_dir)
    if alpha_skip is not None:
        logger.warning("Clip '%s': alpha hint skipped — %s", clip_dir.name, alpha_skip.reason)

    try:
        clip = Clip(name=clip_dir.name, root=clip_dir, input_path=input_path, alpha_path=alpha_path)
        return clip, None
    except ValidationError as e:
        reason = f"validation failed: {e}"
        logger.warning("Skipping '%s': %s", clip_dir, reason)
        return None, SkippedClip(path=clip_dir, reason=reason)


def find_input(clip_dir: Path) -> tuple[Path | None, SkippedClip | None]:
    """Locate the input asset inside a clip folder (case-insensitive).

    Returns:
        (video_path, None) if Input/ contains exactly one video.
        (input_dir, None) if Input/ contains no video (image sequence).
        (None, SkippedPath) if Input/ contains multiple videos (ambiguous).
        (None, None) if no Input/ folder exists.

    Raises:
        PermissionError: If clip_dir cannot be read.
    """
    input_dir = _find_icase(clip_dir, "Input")
    if input_dir is None:
        return None, None

    videos = _find_videos_in(input_dir)
    if len(videos) == 0:
        return input_dir, None
    if len(videos) == 1:
        return videos[0], None

    # Multiple videos — ambiguous, report as skipped.
    names = ", ".join(v.name for v in videos)
    reason = f"Input/ contains multiple video files ({names}) — keep exactly one"
    logger.warning("Skipping '%s': %s", clip_dir, reason)
    return None, SkippedClip(path=clip_dir, reason=reason)


def find_alpha(clip_dir: Path) -> tuple[Path | None, SkippedClip | None]:
    """Locate the AlphaHint asset inside a clip folder (case-insensitive).

    Returns:
        (video_path, None) if AlphaHint/ contains exactly one video.
        (alpha_dir, None) if AlphaHint/ contains no video (image sequence).
        (None, SkippedPath) if AlphaHint/ contains multiple videos (ambiguous).
        (None, None) if no AlphaHint/ folder exists.
    """
    alpha_dir = _find_icase(clip_dir, "AlphaHint")
    if alpha_dir is None:
        return None, None

    videos = _find_videos_in(alpha_dir)
    if len(videos) == 0:
        return alpha_dir, None
    if len(videos) == 1:
        return videos[0], None

    names = ", ".join(v.name for v in videos)
    reason = f"AlphaHint/ contains multiple video files ({names}) — keep exactly one"
    return None, SkippedClip(path=clip_dir, reason=reason)


def _find_videos_in(directory: Path) -> list[Path]:
    """Return all video files found in a directory, sorted by name."""
    try:
        return sorted(
            (child for child in directory.iterdir() if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        return []


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


def _safe_move(src: Path, dst: Path) -> None:
    """Copy src to dst, verify size matches, then delete src.

    Safer than shutil.move for cross-filesystem moves: shutil.move falls back
    to copy-then-delete without verifying the copy succeeded. This function
    verifies the destination size matches before removing the source.

    Raises:
        OSError: If copy, size verification, or delete fails.
    """
    src_size = src.stat().st_size
    try:
        shutil.copy2(str(src), str(dst))
    except OSError as e:
        raise OSError(f"Failed to copy '{src}' to '{dst}': {e}") from e

    dst_size = dst.stat().st_size
    if dst_size != src_size:
        dst.unlink(missing_ok=True)
        raise OSError(
            f"Copy verification failed for '{src}' -> '{dst}': source size {src_size} != destination size {dst_size}"
        )

    try:
        src.unlink()
        logger.info("Moved video '%s' -> '%s'", src, dst)
    except OSError as e:
        raise OSError(f"Copied '{src}' to '{dst}' but failed to delete source: {e}") from e
