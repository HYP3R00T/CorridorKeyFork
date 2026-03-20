"""Stage 1 - validation.

Checks that input frames exist and, if alpha is present, that counts match.
"""

from __future__ import annotations

from pathlib import Path

from corridorkey_new.entrypoint import Clip
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


def validate(clip: Clip) -> None:
    """Validate a clip's assets.

    Raises:
        ValueError: If input has no frames or alpha frame count mismatches input.
    """
    input_count = count_frames(clip.input_path)
    if input_count == 0:
        raise ValueError(f"Clip '{clip.name}': no image frames found in {clip.input_path}")

    if clip.alpha_path is not None:
        alpha_count = count_frames(clip.alpha_path)
        if input_count != alpha_count:
            raise ValueError(
                f"Clip '{clip.name}': frame count mismatch — {input_count} input frames vs {alpha_count} alpha frames"
            )
