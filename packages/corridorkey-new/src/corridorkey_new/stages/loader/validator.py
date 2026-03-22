"""Stage 1 - validation.

Checks that input frames exist and, if alpha is present, that counts match.
"""

from __future__ import annotations

from pathlib import Path

from corridorkey_new.errors import FrameMismatchError
from corridorkey_new.infra.utils import natural_sort_key

IMAGE_EXTENSIONS = frozenset({".exr", ".png", ".jpg", ".jpeg", ".tiff", ".tif"})
LINEAR_EXTENSIONS = frozenset({".exr"})


def get_frame_files(path: Path) -> list[Path]:
    """Return naturally sorted image files in a directory.

    Args:
        path: Directory to scan.

    Returns:
        Naturally sorted list of image file paths.
    """
    if not path.is_dir():
        return []
    files = [p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def count_frames(path: Path) -> int:
    """Count image files in a directory."""
    return len(get_frame_files(path))


def detect_is_linear(path: Path) -> bool:
    """Detect whether input frames are in linear light from the first file's extension.

    Args:
        path: Input frames directory.

    Returns:
        True if the first frame has a linear extension (e.g. .exr), False otherwise.
    """
    frames = get_frame_files(path)
    if not frames:
        return False
    return frames[0].suffix.lower() in LINEAR_EXTENSIONS


def validate(clip_name: str, frames_dir: Path, alpha_frames_dir: Path | None) -> None:
    """Validate resolved frame sequence directories.

    Args:
        clip_name: Clip name for error messages.
        frames_dir: Resolved input frames directory.
        alpha_frames_dir: Resolved alpha frames directory, or None.

    Raises:
        ValueError: If input has no frames.
        FrameMismatchError: If alpha frame count mismatches input.
    """
    input_count = count_frames(frames_dir)
    if input_count == 0:
        raise ValueError(f"Clip '{clip_name}': no image frames found in {frames_dir}")

    if alpha_frames_dir is not None:
        alpha_count = count_frames(alpha_frames_dir)
        if input_count != alpha_count:
            raise FrameMismatchError(clip_name, input_count, alpha_count)
