from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import ValidationError

from corridorkey.errors import ClipScanError
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
        ClipScanError: If directory creation, copy, size verification, or delete fails.
    """
    clip_root = video_path.parent
    input_dir = clip_root / "Input"
    alpha_dir = clip_root / "AlphaHint"

    try:
        input_dir.mkdir(exist_ok=True)
        alpha_dir.mkdir(exist_ok=True)
    except OSError as e:
        raise ClipScanError(f"Failed to create clip structure in '{clip_root}': {e}") from e

    dest = input_dir / video_path.name
    if dest.exists() and dest.stat().st_size == video_path.stat().st_size:
        logger.debug("Video already in place, skipping move: '%s'", dest)
    else:
        _safe_move(video_path, dest)

    try:
        return Clip(name=clip_root.name, root=clip_root, input_path=dest, alpha_path=None)
    except ValidationError as e:
        raise ClipScanError(f"Clip validation failed after reorganising '{video_path}': {e}") from e


def try_build_clip(clip_dir: Path) -> tuple[Clip | None, SkippedClip | None]:
    """Attempt to build a Clip from a directory.

    Args:
        clip_dir: Directory to inspect for Input/ and AlphaHint/ assets.

    Returns:
        (Clip, None) on success.
        (None, SkippedClip) if the directory has an Input/ folder but it is
            ambiguous, empty of videos, or fails Clip validation.
        (None, None) if the directory has no Input/ folder at all.
    """
    try:
        input_path, input_skip = _find_asset(clip_dir, "Input")
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

    alpha_path, alpha_skip = _find_asset(clip_dir, "AlphaHint")
    if alpha_skip is not None:
        # AlphaHint ambiguity is recoverable — the clip can still be built without alpha
        # and alpha will be generated externally. Input ambiguity is not recoverable
        # (handled above) because there is no way to know which video is the source.
        logger.warning(
            "Clip '%s': AlphaHint skipped (%s) — clip will need alpha generated", clip_dir.name, alpha_skip.reason
        )
    try:
        clip = Clip(name=clip_dir.name, root=clip_dir, input_path=input_path, alpha_path=alpha_path)
        return clip, None
    except ValidationError as e:
        reason = f"validation failed: {e}"
        logger.warning("Skipping '%s': %s", clip_dir, reason)
        return None, SkippedClip(path=clip_dir, reason=reason)


def _find_asset(
    clip_dir: Path,
    folder_name: str,
) -> tuple[Path | None, SkippedClip | None]:
    # Locate a named asset subfolder (Input/ or AlphaHint/) and resolve its content.
    # Returns (asset_path, None), (None, SkippedClip), or (None, None).
    asset_dir = _find_icase(clip_dir, folder_name)
    if asset_dir is None:
        return None, None

    videos = _find_videos_in(asset_dir)
    if len(videos) == 0:
        return asset_dir, None
    if len(videos) == 1:
        return videos[0], None

    names = ", ".join(v.name for v in videos)
    reason = f"{folder_name}/ contains multiple video files ({names}) — keep exactly one"
    return None, SkippedClip(path=clip_dir, reason=reason)


def _find_videos_in(directory: Path) -> list[Path]:
    try:
        return sorted(
            (child for child in directory.iterdir() if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        logger.warning("Cannot read directory '%s' — treating as empty", directory)
        return []


def _find_icase(parent: Path, name: str) -> Path | None:
    try:
        for child in parent.iterdir():
            if child.name.lower() == name.lower():
                return child
    except PermissionError as e:
        raise PermissionError(f"Cannot read directory: {parent}") from e
    return None


def _safe_move(src: Path, dst: Path) -> None:
    src_size = src.stat().st_size
    try:
        shutil.copy2(str(src), str(dst))
    except OSError as e:
        raise ClipScanError(f"Failed to copy '{src}' to '{dst}': {e}") from e

    dst_size = dst.stat().st_size
    if dst_size != src_size:
        dst.unlink(missing_ok=True)
        raise ClipScanError(
            f"Copy verification failed for '{src}' -> '{dst}': source size {src_size} != destination size {dst_size}"
        )

    try:
        src.unlink()
        logger.info("Moved video '%s' -> '%s'", src, dst)
    except OSError as e:
        raise ClipScanError(f"Copied '{src}' to '{dst}' but failed to delete source: {e}") from e
